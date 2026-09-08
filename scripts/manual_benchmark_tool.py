"""Standalone Manual Benchmark & Annotation Tool for Menipy.

Enables researchers to:
1. Interactively or via CLI select and adjust substrate baselines (sessile) and needle dimensions (pendant).
2. Override physical properties (densities, gravity) and define ground-truth values.
3. Run Menipy pipelines headlessly with exact manual or auto-detected geometry.
4. Export and import extracted contours, regions, and metrics as portable JSON files.
5. Ingest reported ground-truth from original benchmark code sources (Conan-ML, OpenDrop, UCLA, EPFL, pyDSA).

This tool is completely independent of the main Menipy GUI and can be used for
reproducibility studies, regression verification, and dataset validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.pipelines.runner import PipelineRunner

logger = logging.getLogger("manual_benchmark")
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
DEFAULT_ANNOTATION_DIR = BENCHMARK_DIR / "annotations"


# ==============================================================================
# 1. Annotation Data Schema & Serialization
# ==============================================================================


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder handling numpy scalars, arrays, and tuples."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32, np.uint8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


def _sanitize_for_json(val: Any) -> Any:
    """Recursively convert numpy types, tuples, and custom structures to native JSON-serializable types."""
    if isinstance(val, (tuple, list)):
        return [_sanitize_for_json(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer, np.int64, np.int32, np.uint8)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    return val


@dataclass
class BenchmarkAnnotation:
    """Portable annotation container for manual benchmark testing."""

    dataset: str = "custom"
    image_file: str = ""
    pipeline: str = "sessile"
    # Geometry overrides
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    needle_rect: tuple[int, int, int, int] | None = None
    roi_rect: tuple[int, int, int, int] | None = None
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None
    apex_point: tuple[int, int] | None = None
    px_per_mm: float | None = None
    needle_diameter_mm: float | None = None
    # Physical properties
    physics: dict[str, float] = field(
        default_factory=lambda: {"rho1": 998.2, "rho2": 1.2, "g": 9.80665}
    )
    # Ground truth
    ground_truth: dict[str, Any] = field(default_factory=dict)
    # Contour serialization (Nx2 float coordinates)
    drop_contour: list[list[float]] | None = None
    # Execution results cache
    results: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert annotation to JSON-serializable dictionary."""
        d = asdict(self)
        return _sanitize_for_json(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkAnnotation:
        """Construct BenchmarkAnnotation from a dictionary."""
        # Convert nested substrate tuples if present
        sub = data.get("substrate_line")
        if sub and len(sub) == 2:
            data["substrate_line"] = (
                (float(sub[0][0]), float(sub[0][1])),
                (float(sub[1][0]), float(sub[1][1])),
            )
        nr = data.get("needle_rect")
        if nr and len(nr) == 4:
            data["needle_rect"] = (int(nr[0]), int(nr[1]), int(nr[2]), int(nr[3]))
        roi = data.get("roi_rect")
        if roi and len(roi) == 4:
            data["roi_rect"] = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        cp = data.get("contact_points")
        if cp and len(cp) == 2:
            data["contact_points"] = (
                (int(cp[0][0]), int(cp[0][1])),
                (int(cp[1][0]), int(cp[1][1])),
            )
        ap = data.get("apex_point")
        if ap and len(ap) == 2:
            data["apex_point"] = (int(ap[0]), int(ap[1]))

        # Filter out unknown keys for forward compatibility
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, filepath: Path | str) -> None:
        """Save annotation to a JSON file atomically."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        clean_dict = self.to_dict()
        tmp_p = p.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(clean_dict, f, indent=2, cls=NumpyEncoder)
        tmp_p.replace(p)

    @classmethod
    def load(cls, filepath: Path | str) -> BenchmarkAnnotation:
        """Load annotation from a JSON file."""
        p = Path(filepath)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ==============================================================================
# 2. Ground-Truth Presets & Source Importers
# ==============================================================================


def get_known_benchmark_preset(image_path: Path | str) -> BenchmarkAnnotation:
    """Derive known ground truth and geometry presets from benchmark dataset sources."""
    path = Path(image_path)
    filename = path.name
    parent_dir = path.parent.name.lower()

    # 1. Conan-ML dataset (Langmuir 2024)
    # Filenames typically encode contact angle, e.g. "111.031693.bmp" -> 111.03
    if "conan" in parent_dir or parent_dir == "conan_ml":
        expected_ca = None
        clean_name = filename.replace(".bmp", "").replace(".png", "")
        # Handle formats like "2-s@M@Z-120.321945190429" or "111.031693" or "359 CA162"
        for part in clean_name.replace("CA", " ").replace("-", " ").split():
            try:
                val = float(part)
                if 10.0 <= val <= 180.0:
                    expected_ca = round(val, 2)
                    break
            except ValueError:
                continue

        return BenchmarkAnnotation(
            dataset="conan_ml",
            image_file=filename,
            pipeline="sessile",
            ground_truth={
                "contact_angle_deg": expected_ca,
                "source": "Berry et al., Langmuir 2024 (Conan-ML)",
            },
        )

    # 2. OpenDrop Pendant dataset (Berry et al., JCIS 2015)
    # 22G needle, outer diameter = 0.718 mm, IUPAC water at 20C gamma = 72.80 mN/m
    if "pendant" == parent_dir and "water_in_air" in filename:
        return BenchmarkAnnotation(
            dataset="opendrop_pendant",
            image_file=filename,
            pipeline="pendant",
            needle_diameter_mm=0.718,
            physics={"rho1": 998.2, "rho2": 1.2, "g": 9.80665},
            ground_truth={
                "surface_tension_mN_m": 72.80,
                "source": "Berry et al., JCIS 2015 (OpenDrop 22G needle)",
            },
        )

    # 3. UCLA Pendant Drop dataset (Pirouz Kavehpour et al.)
    if "ucla" in parent_dir or parent_dir == "ucla_pendant":
        if "h2o" in filename.lower():
            return BenchmarkAnnotation(
                dataset="ucla_pendant",
                image_file=filename,
                pipeline="pendant",
                needle_diameter_mm=0.51,  # 25G needle
                physics={"rho1": 998.2, "rho2": 1.2, "g": 9.80665},
                ground_truth={
                    "surface_tension_mN_m": 72.80,
                    "source": "UCLA Pendant Drop (25G needle, water)",
                },
            )
        elif "hexadecane" in filename.lower():
            return BenchmarkAnnotation(
                dataset="ucla_pendant",
                image_file=filename,
                pipeline="pendant",
                needle_diameter_mm=1.95,  # 14G needle
                physics={"rho1": 773.0, "rho2": 1.2, "g": 9.80665},
                ground_truth={
                    "surface_tension_mN_m": 27.50,
                    "source": "UCLA Pendant Drop (14G needle, hexadecane)",
                },
            )

    # 4. EPFL LB-ADSA dataset (Stalder et al., Colloids Surf. A 2006)
    if "epfl" in parent_dir or "epfl_drop_analysis" in parent_dir:
        return BenchmarkAnnotation(
            dataset="epfl_drop_analysis",
            image_file=filename,
            pipeline="sessile",
            px_per_mm=191.82,
            physics={"rho1": 998.2, "rho2": 1.2, "g": 9.80665},
            ground_truth={
                "optical_scale_px_mm": 191.82,
                "capillary_constant_mm2": 13.4752,
                "source": "Stalder et al., EPFL LB-ADSA (sample.jpg)",
            },
        )

    # 5. pyDSA dataset (Gaby Launay, INSA-Lyon)
    if "pydsa" in parent_dir:
        return BenchmarkAnnotation(
            dataset="pydsa",
            image_file=filename,
            pipeline="sessile",
            substrate_line=((100.0, 200.0), (1500.0, 200.0)),
            ground_truth={
                "baseline": [[100, 200], [1500, 200]],
                "scale_mm": 0.44,
                "source": "Launay, pyDSA benchmark",
            },
        )

    # Default fallback
    return BenchmarkAnnotation(
        dataset="custom",
        image_file=filename,
        pipeline="sessile",
    )


# ==============================================================================
# 3. Execution Engine
# ==============================================================================


def run_pipeline_with_annotation(
    image_path: Path | str,
    annotation: BenchmarkAnnotation,
    *,
    use_auto_calibration_seed: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Execute pipeline using manual or annotated geometry overrides.

    Returns:
        tuple of (Context, metrics_dictionary)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    # Seed with AutoCalibrator if manual features not provided and seed enabled
    if use_auto_calibration_seed:
        if (
            annotation.pipeline == "sessile" and annotation.substrate_line is None
        ) or (annotation.pipeline == "pendant" and annotation.needle_rect is None):
            try:
                cal = AutoCalibrator(img, pipeline_name=annotation.pipeline)
                res = cal.detect_all()
                if annotation.pipeline == "sessile" and res.substrate_line:
                    annotation.substrate_line = (
                        (float(res.substrate_line[0][0]), float(res.substrate_line[0][1])),
                        (float(res.substrate_line[1][0]), float(res.substrate_line[1][1])),
                    )
                if annotation.pipeline == "pendant" and res.needle_rect:
                    annotation.needle_rect = res.needle_rect
                if res.contact_points and annotation.contact_points is None:
                    annotation.contact_points = res.contact_points
                if res.apex_point and annotation.apex_point is None:
                    annotation.apex_point = res.apex_point
            except Exception as e:
                logger.warning(f"AutoCalibrator seed failed: {e}")

    runner = PipelineRunner(annotation.pipeline)

    # Assemble execution kwargs
    kwargs: dict[str, Any] = {
        "image": img,
        "image_path": str(path),
    }

    if annotation.substrate_line is not None:
        kwargs["substrate_line"] = annotation.substrate_line
    if annotation.needle_rect is not None:
        kwargs["needle_rect"] = annotation.needle_rect
    if annotation.needle_diameter_mm is not None:
        kwargs["needle_diameter_mm"] = annotation.needle_diameter_mm
    if annotation.roi_rect is not None:
        kwargs["roi_rect"] = annotation.roi_rect
    if annotation.contact_points is not None:
        kwargs["contact_points"] = annotation.contact_points
    if annotation.apex_point is not None:
        kwargs["apex_point"] = annotation.apex_point
    if annotation.px_per_mm is not None:
        kwargs["px_per_mm"] = annotation.px_per_mm
    if annotation.physics:
        kwargs["physics"] = annotation.physics
    if annotation.drop_contour is not None:
        kwargs["drop_contour"] = np.array(annotation.drop_contour, dtype=np.float64)

    t0 = time.perf_counter()
    ctx = runner.run(**kwargs)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Extract results and contour
    res_dict: dict[str, Any] = dict(ctx.results) if ctx.results else {}
    res_dict["runtime_ms"] = dt_ms

    # Extract detected contour for export
    if ctx.contour is not None and hasattr(ctx.contour, "xy"):
        annotation.drop_contour = ctx.contour.xy.tolist()
    elif ctx.drop_contour is not None:
        if isinstance(ctx.drop_contour, np.ndarray):
            annotation.drop_contour = ctx.drop_contour.reshape(-1, 2).tolist()

    # Extract computed contact points / apex if found
    if hasattr(ctx, "contact_points") and ctx.contact_points:
        annotation.contact_points = ctx.contact_points
    if hasattr(ctx, "apex_point") and ctx.apex_point:
        annotation.apex_point = ctx.apex_point

    # Evaluate against ground truth
    metrics: dict[str, Any] = {
        "pipeline": annotation.pipeline,
        "runtime_ms": dt_ms,
        "status": "UNKNOWN",
    }

    if annotation.pipeline == "sessile":
        ca = res_dict.get("contact_angle_deg")
        if ca is None:
            tl = res_dict.get("theta_left_deg")
            tr = res_dict.get("theta_right_deg")
            if tl is not None and tr is not None:
                ca = (tl + tr) / 2.0
        metrics["contact_angle_deg"] = ca
        expected_ca = annotation.ground_truth.get("contact_angle_deg")
        if expected_ca is not None and ca is not None:
            err = abs(ca - expected_ca)
            metrics["expected_ca"] = expected_ca
            metrics["error_deg"] = err
            metrics["status"] = "PASS" if err < 5.0 else ("FAIR" if err < 10.0 else "DEVIATED")

    elif annotation.pipeline == "pendant":
        gamma = res_dict.get("surface_tension_mN_m")
        metrics["surface_tension_mN_m"] = gamma
        expected_gamma = annotation.ground_truth.get("surface_tension_mN_m")
        if expected_gamma is not None and gamma is not None:
            dev_pct = abs(gamma - expected_gamma) / expected_gamma * 100.0
            metrics["expected_gamma"] = expected_gamma
            metrics["dev_pct"] = dev_pct
            metrics["status"] = "PASS" if dev_pct < 5.0 else ("FAIR" if dev_pct < 10.0 else "DEVIATED")

    annotation.results = res_dict
    return ctx, metrics


# ==============================================================================
# 4. Interactive PySide6 Canvas & Standalone Tool
# ==============================================================================

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QFont,
        QImage,
        QMouseEvent,
        QPainter,
        QPen,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


if HAS_PYSIDE6:

    class InteractiveImageCanvas(QWidget):
        """Interactive widget displaying image with draggable substrate line and needle box."""

        geometryChanged = Signal()

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.image_bgr: np.ndarray | None = None
            self.qpixmap: QPixmap | None = None

            # Geometry state
            self.substrate_p1: QPointF | None = None
            self.substrate_p2: QPointF | None = None
            self.needle_rect: QRectF | None = None
            self.drop_contour: list[tuple[float, float]] | None = None
            self.contact_points: list[tuple[float, float]] | None = None
            self.apex_point: tuple[float, float] | None = None

            # Interaction mode
            self.active_handle: str | None = None
            self.drag_start: QPointF | None = None
            self.pipeline_mode: str = "sessile"
            self.handle_radius = 8.0

            self.setMouseTracking(True)

        def set_image(self, img_bgr: np.ndarray) -> None:
            """Load image into canvas."""
            self.image_bgr = img_bgr
            h, w = img_bgr.shape[:2]
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.qpixmap = QPixmap.fromImage(qimg)
            self.setMinimumSize(w, h)
            self.update()

        def set_geometry(self, annotation: BenchmarkAnnotation) -> None:
            """Populate geometry from annotation."""
            self.pipeline_mode = annotation.pipeline
            if annotation.substrate_line:
                (x1, y1), (x2, y2) = annotation.substrate_line
                self.substrate_p1 = QPointF(x1, y1)
                self.substrate_p2 = QPointF(x2, y2)
            else:
                self.substrate_p1 = None
                self.substrate_p2 = None

            if annotation.needle_rect:
                x, y, w, h = annotation.needle_rect
                self.needle_rect = QRectF(x, y, w, h)
            else:
                self.needle_rect = None

            if annotation.drop_contour:
                self.drop_contour = [(float(pt[0]), float(pt[1])) for pt in annotation.drop_contour]
            else:
                self.drop_contour = None

            if annotation.contact_points:
                self.contact_points = [
                    (float(pt[0]), float(pt[1])) for pt in annotation.contact_points
                ]
            else:
                self.contact_points = None

            if annotation.apex_point:
                self.apex_point = (float(annotation.apex_point[0]), float(annotation.apex_point[1]))
            else:
                self.apex_point = None

            self.update()

        def get_substrate_line(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
            if self.substrate_p1 and self.substrate_p2:
                return (
                    (self.substrate_p1.x(), self.substrate_p1.y()),
                    (self.substrate_p2.x(), self.substrate_p2.y()),
                )
            return None

        def get_needle_rect(self) -> tuple[int, int, int, int] | None:
            if self.needle_rect:
                r = self.needle_rect.normalized()
                return (int(r.x()), int(r.y()), int(r.width()), int(r.height()))
            return None

        def paintEvent(self, event: Any) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw background image
            if self.qpixmap:
                painter.drawPixmap(0, 0, self.qpixmap)
            else:
                painter.fillRect(self.rect(), QColor(40, 40, 40))
                painter.setPen(QColor(180, 180, 180))
                painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
                return

            # Draw drop contour if available
            if self.drop_contour and len(self.drop_contour) > 2:
                pen_contour = QPen(QColor(255, 235, 59, 200), 2.0)
                painter.setPen(pen_contour)
                for i in range(len(self.drop_contour) - 1):
                    p_a = QPointF(self.drop_contour[i][0], self.drop_contour[i][1])
                    p_b = QPointF(self.drop_contour[i + 1][0], self.drop_contour[i + 1][1])
                    painter.drawLine(p_a, p_b)

            # Draw sessile substrate baseline
            if self.pipeline_mode == "sessile" and self.substrate_p1 and self.substrate_p2:
                pen_line = QPen(QColor(0, 230, 118), 2.5)
                painter.setPen(pen_line)
                painter.drawLine(self.substrate_p1, self.substrate_p2)

                # Draw endpoint handles
                brush_handle = QBrush(QColor(0, 230, 118, 220))
                painter.setBrush(brush_handle)
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.drawEllipse(self.substrate_p1, self.handle_radius, self.handle_radius)
                painter.drawEllipse(self.substrate_p2, self.handle_radius, self.handle_radius)

            # Draw pendant needle box
            if self.pipeline_mode == "pendant" and self.needle_rect:
                pen_needle = QPen(QColor(33, 150, 243), 2.0, Qt.DashLine)
                painter.setPen(pen_needle)
                painter.setBrush(QBrush(QColor(33, 150, 243, 40)))
                painter.drawRect(self.needle_rect)

                # Corner handles
                painter.setBrush(QBrush(QColor(33, 150, 243, 200)))
                painter.setPen(QPen(QColor(255, 255, 255), 1.0))
                for pt in [
                    self.needle_rect.topLeft(),
                    self.needle_rect.topRight(),
                    self.needle_rect.bottomLeft(),
                    self.needle_rect.bottomRight(),
                ]:
                    painter.drawRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8))

            # Draw contact points
            if self.contact_points:
                painter.setBrush(QBrush(QColor(233, 30, 99)))
                painter.setPen(QPen(QColor(255, 255, 255), 1.0))
                for pt in self.contact_points:
                    painter.drawEllipse(QPointF(pt[0], pt[1]), 5.0, 5.0)

            # Draw apex
            if self.apex_point:
                painter.setBrush(QBrush(QColor(156, 39, 176)))
                painter.setPen(QPen(QColor(255, 255, 255), 1.0))
                painter.drawEllipse(QPointF(self.apex_point[0], self.apex_point[1]), 5.0, 5.0)

        def _dist(self, p1: QPointF, p2: QPointF) -> float:
            return float(np.hypot(p1.x() - p2.x(), p1.y() - p2.y()))

        def mousePressEvent(self, event: QMouseEvent) -> None:
            if event.button() != Qt.LeftButton:
                return

            pos = event.position()
            self.active_handle = None

            if self.pipeline_mode == "sessile":
                if self.substrate_p1 and self._dist(pos, self.substrate_p1) <= self.handle_radius * 1.5:
                    self.active_handle = "sub_p1"
                elif self.substrate_p2 and self._dist(pos, self.substrate_p2) <= self.handle_radius * 1.5:
                    self.active_handle = "sub_p2"
                elif not self.substrate_p1:
                    self.substrate_p1 = pos
                    self.update()
                elif not self.substrate_p2:
                    self.substrate_p2 = pos
                    self.geometryChanged.emit()
                    self.update()

            elif self.pipeline_mode == "pendant":
                if self.needle_rect:
                    r = self.needle_rect
                    if self._dist(pos, r.bottomLeft()) <= 10.0:
                        self.active_handle = "needle_bl"
                    elif self._dist(pos, r.bottomRight()) <= 10.0:
                        self.active_handle = "needle_br"
                    elif r.contains(pos):
                        self.active_handle = "needle_move"
                        self.drag_start = pos
                else:
                    self.needle_rect = QRectF(pos.x() - 50, pos.y() - 100, 100, 100)
                    self.geometryChanged.emit()
                    self.update()

        def mouseMoveEvent(self, event: QMouseEvent) -> None:
            pos = event.position()
            if not (event.buttons() & Qt.LeftButton) or not self.active_handle:
                return

            if self.active_handle == "sub_p1":
                self.substrate_p1 = pos
                self.geometryChanged.emit()
                self.update()
            elif self.active_handle == "sub_p2":
                self.substrate_p2 = pos
                self.geometryChanged.emit()
                self.update()
            elif self.active_handle == "needle_bl" and self.needle_rect:
                r = self.needle_rect
                self.needle_rect = QRectF(pos.x(), r.y(), r.right() - pos.x(), pos.y() - r.y())
                self.geometryChanged.emit()
                self.update()
            elif self.active_handle == "needle_br" and self.needle_rect:
                r = self.needle_rect
                self.needle_rect = QRectF(r.left(), r.y(), pos.x() - r.left(), pos.y() - r.y())
                self.geometryChanged.emit()
                self.update()
            elif self.active_handle == "needle_move" and self.needle_rect and self.drag_start:
                dx = pos.x() - self.drag_start.x()
                dy = pos.y() - self.drag_start.y()
                self.needle_rect.translate(dx, dy)
                self.drag_start = pos
                self.geometryChanged.emit()
                self.update()

        def mouseReleaseEvent(self, event: QMouseEvent) -> None:
            if self.active_handle:
                self.active_handle = None
                self.drag_start = None
                self.geometryChanged.emit()


    class ManualBenchmarkWindow(QMainWindow):
        """Standalone GUI Window for manual benchmark inspection and contour export."""

        def __init__(self, initial_image: Path | str | None = None) -> None:
            super().__init__()
            self.setWindowTitle("Menipy - Standalone Manual Benchmark & Annotation Tool")
            self.resize(1300, 850)

            self.current_image_path: Path | None = None
            self.annotation = BenchmarkAnnotation()

            self._init_ui()

            if initial_image:
                self.load_image(initial_image)

        def _init_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)

            splitter = QSplitter(Qt.Horizontal)
            main_layout.addWidget(splitter)

            # Left Control Panel
            control_panel = QWidget()
            control_layout = QVBoxLayout(control_panel)
            control_layout.setContentsMargins(10, 10, 10, 10)

            # File & Dataset Group
            grp_file = QGroupBox("Benchmark Dataset & Image")
            form_file = QFormLayout(grp_file)

            btn_open = QPushButton("Open Image...")
            btn_open.clicked.connect(self._on_open_image)
            form_file.addRow(btn_open)

            self.lbl_filename = QLabel("No image loaded")
            self.lbl_filename.setWordWrap(True)
            form_file.addRow("File:", self.lbl_filename)

            self.txt_dataset = QLineEdit("custom")
            form_file.addRow("Dataset:", self.txt_dataset)

            self.txt_pipeline = QLineEdit("sessile")
            form_file.addRow("Pipeline:", self.txt_pipeline)
            control_layout.addWidget(grp_file)

            # Substrate & Needle Geometry Group
            self.grp_geom = QGroupBox("Manual Geometry Overrides")
            form_geom = QFormLayout(self.grp_geom)

            btn_auto = QPushButton("Seed with AutoCalibrator")
            btn_auto.clicked.connect(self._on_auto_calibrate)
            form_geom.addRow(btn_auto)

            self.txt_sub_x1 = QSpinBox()
            self.txt_sub_x1.setRange(0, 10000)
            self.txt_sub_y1 = QSpinBox()
            self.txt_sub_y1.setRange(0, 10000)
            self.txt_sub_x2 = QSpinBox()
            self.txt_sub_x2.setRange(0, 10000)
            self.txt_sub_y2 = QSpinBox()
            self.txt_sub_y2.setRange(0, 10000)

            sub_row = QHBoxLayout()
            sub_row.addWidget(QLabel("P1:"))
            sub_row.addWidget(self.txt_sub_x1)
            sub_row.addWidget(self.txt_sub_y1)
            sub_row.addWidget(QLabel("P2:"))
            sub_row.addWidget(self.txt_sub_x2)
            sub_row.addWidget(self.txt_sub_y2)
            form_geom.addRow("Substrate Line:", sub_row)

            for sb in (self.txt_sub_x1, self.txt_sub_y1, self.txt_sub_x2, self.txt_sub_y2):
                sb.valueChanged.connect(self._on_sub_spin_changed)

            self.txt_needle_dia = QLineEdit("0.718")
            form_geom.addRow("Needle Diam (mm):", self.txt_needle_dia)

            self.txt_px_mm = QLineEdit("")
            form_geom.addRow("Scale (px/mm):", self.txt_px_mm)

            control_layout.addWidget(self.grp_geom)

            # Physics & Ground Truth Group
            grp_phys = QGroupBox("Physical Properties & Expected Truth")
            form_phys = QFormLayout(grp_phys)

            self.txt_rho1 = QLineEdit("998.2")
            form_phys.addRow("Drop Density (kg/m³):", self.txt_rho1)
            self.txt_rho2 = QLineEdit("1.2")
            form_phys.addRow("Ambient Density (kg/m³):", self.txt_rho2)

            self.txt_expected_ca = QLineEdit("")
            form_phys.addRow("Expected CA (°):", self.txt_expected_ca)

            self.txt_expected_gamma = QLineEdit("72.80")
            form_phys.addRow("Expected γ (mN/m):", self.txt_expected_gamma)

            control_layout.addWidget(grp_phys)

            # Execution & Export Group
            grp_actions = QGroupBox("Actions & Persistence")
            vbox_actions = QVBoxLayout(grp_actions)

            self.btn_run = QPushButton("▶ Run Pipeline with Overrides")
            self.btn_run.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white; padding: 6px;")
            self.btn_run.clicked.connect(self._on_run_pipeline)
            vbox_actions.addWidget(self.btn_run)

            btn_export = QPushButton("Export Annotation & Contours (JSON)")
            btn_export.clicked.connect(self._on_export_annotation)
            vbox_actions.addWidget(btn_export)

            btn_import = QPushButton("Import Annotation / Contour (JSON)")
            btn_import.clicked.connect(self._on_import_annotation)
            vbox_actions.addWidget(btn_import)

            control_layout.addWidget(grp_actions)
            control_layout.addStretch()

            # Results Table
            self.tbl_results = QTableWidget(5, 2)
            self.tbl_results.setHorizontalHeaderLabels(["Metric", "Value"])
            self.tbl_results.verticalHeader().setVisible(False)
            self.tbl_results.setMaximumHeight(160)
            control_layout.addWidget(self.tbl_results)

            splitter.addWidget(control_panel)

            # Right View: Canvas in Scroll Area
            scroll = QScrollArea()
            self.canvas = InteractiveImageCanvas()
            self.canvas.geometryChanged.connect(self._on_canvas_geometry_changed)
            scroll.setWidget(self.canvas)
            scroll.setWidgetResizable(True)
            splitter.addWidget(scroll)

            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 3)

        def load_image(self, image_path: Path | str) -> None:
            """Load image and apply initial benchmark presets."""
            p = Path(image_path)
            self.current_image_path = p
            self.lbl_filename.setText(p.name)

            img = cv2.imread(str(p))
            if img is None:
                QMessageBox.critical(self, "Error", f"Failed to load image: {p}")
                return

            self.canvas.set_image(img)

            # Load preset or default
            self.annotation = get_known_benchmark_preset(p)
            self.txt_dataset.setText(self.annotation.dataset)
            self.txt_pipeline.setText(self.annotation.pipeline)

            if self.annotation.needle_diameter_mm:
                self.txt_needle_dia.setText(str(self.annotation.needle_diameter_mm))
            if self.annotation.px_per_mm:
                self.txt_px_mm.setText(str(self.annotation.px_per_mm))

            gt = self.annotation.ground_truth
            if "contact_angle_deg" in gt and gt["contact_angle_deg"] is not None:
                self.txt_expected_ca.setText(f"{gt['contact_angle_deg']:.2f}")
            if "surface_tension_mN_m" in gt and gt["surface_tension_mN_m"] is not None:
                self.txt_expected_gamma.setText(f"{gt['surface_tension_mN_m']:.2f}")

            # Auto-seed geometry if empty
            if not self.annotation.substrate_line and not self.annotation.needle_rect:
                self._on_auto_calibrate()
            else:
                self.canvas.set_geometry(self.annotation)
                self._sync_inputs_from_annotation()

        def _sync_inputs_from_annotation(self) -> None:
            if self.annotation.substrate_line:
                (x1, y1), (x2, y2) = self.annotation.substrate_line
                self.txt_sub_x1.blockSignals(True)
                self.txt_sub_y1.blockSignals(True)
                self.txt_sub_x2.blockSignals(True)
                self.txt_sub_y2.blockSignals(True)
                self.txt_sub_x1.setValue(int(x1))
                self.txt_sub_y1.setValue(int(y1))
                self.txt_sub_x2.setValue(int(x2))
                self.txt_sub_y2.setValue(int(y2))
                self.txt_sub_x1.blockSignals(False)
                self.txt_sub_y1.blockSignals(False)
                self.txt_sub_x2.blockSignals(False)
                self.txt_sub_y2.blockSignals(False)

        def _on_canvas_geometry_changed(self) -> None:
            sub = self.canvas.get_substrate_line()
            if sub:
                self.annotation.substrate_line = sub
                self._sync_inputs_from_annotation()

            needle = self.canvas.get_needle_rect()
            if needle:
                self.annotation.needle_rect = needle

        def _on_sub_spin_changed(self) -> None:
            p1 = (float(self.txt_sub_x1.value()), float(self.txt_sub_y1.value()))
            p2 = (float(self.txt_sub_x2.value()), float(self.txt_sub_y2.value()))
            self.annotation.substrate_line = (p1, p2)
            self.canvas.set_geometry(self.annotation)

        def _on_auto_calibrate(self) -> None:
            if not self.current_image_path:
                return
            img = self.canvas.image_bgr
            if img is None:
                return
            pipe = self.txt_pipeline.text().strip().lower()
            try:
                cal = AutoCalibrator(img, pipeline_name=pipe)
                res = cal.detect_all()
                if pipe == "sessile" and res.substrate_line:
                    (x1, y1), (x2, y2) = res.substrate_line
                    self.annotation.substrate_line = ((float(x1), float(y1)), (float(x2), float(y2)))
                elif pipe == "pendant" and res.needle_rect:
                    self.annotation.needle_rect = res.needle_rect
                if res.contact_points:
                    self.annotation.contact_points = res.contact_points
                if res.apex_point:
                    self.annotation.apex_point = res.apex_point

                self.canvas.set_geometry(self.annotation)
                self._sync_inputs_from_annotation()
            except Exception as e:
                QMessageBox.warning(self, "Auto-Calibrate", f"Auto-calibration failed: {e}")

        def _on_open_image(self) -> None:
            fn, _ = QFileDialog.getOpenFileName(
                self, "Open Droplet Image", str(BENCHMARK_DIR), "Images (*.png *.bmp *.jpg *.jpeg *.tif)"
            )
            if fn:
                self.load_image(fn)

        def _on_run_pipeline(self) -> None:
            if not self.current_image_path:
                QMessageBox.warning(self, "Warning", "Please load an image first.")
                return

            # Sync GUI fields to annotation
            self.annotation.dataset = self.txt_dataset.text().strip()
            self.annotation.pipeline = self.txt_pipeline.text().strip().lower()
            if self.txt_needle_dia.text().strip():
                try:
                    self.annotation.needle_diameter_mm = float(self.txt_needle_dia.text().strip())
                except ValueError:
                    pass
            if self.txt_px_mm.text().strip():
                try:
                    self.annotation.px_per_mm = float(self.txt_px_mm.text().strip())
                except ValueError:
                    self.annotation.px_per_mm = None
            else:
                self.annotation.px_per_mm = None

            try:
                rho1 = float(self.txt_rho1.text().strip())
                rho2 = float(self.txt_rho2.text().strip())
                self.annotation.physics = {"rho1": rho1, "rho2": rho2, "g": 9.80665}
            except ValueError:
                pass

            gt_ca = None
            if self.txt_expected_ca.text().strip():
                try:
                    gt_ca = float(self.txt_expected_ca.text().strip())
                except ValueError:
                    pass

            gt_gamma = None
            if self.txt_expected_gamma.text().strip():
                try:
                    gt_gamma = float(self.txt_expected_gamma.text().strip())
                except ValueError:
                    pass

            self.annotation.ground_truth = {
                "contact_angle_deg": gt_ca,
                "surface_tension_mN_m": gt_gamma,
            }

            try:
                _ctx, metrics = run_pipeline_with_annotation(
                    self.current_image_path, self.annotation, use_auto_calibration_seed=False
                )
                self.canvas.set_geometry(self.annotation)
                self._display_results(metrics)
            except Exception as e:
                QMessageBox.critical(self, "Execution Error", f"Pipeline execution failed:\n{e}")

        def _display_results(self, metrics: dict[str, Any]) -> None:
            rows = []
            if "contact_angle_deg" in metrics and metrics["contact_angle_deg"] is not None:
                rows.append(("Contact Angle", f"{metrics['contact_angle_deg']:.2f}°"))
                if "expected_ca" in metrics and metrics["expected_ca"] is not None:
                    rows.append(("Expected CA", f"{metrics['expected_ca']:.2f}°"))
                    rows.append(("Error", f"{metrics.get('error_deg', 0.0):.2f}°"))
            if "surface_tension_mN_m" in metrics and metrics["surface_tension_mN_m"] is not None:
                rows.append(("Surface Tension", f"{metrics['surface_tension_mN_m']:.2f} mN/m"))
                if "expected_gamma" in metrics and metrics["expected_gamma"] is not None:
                    rows.append(("Expected γ", f"{metrics['expected_gamma']:.2f} mN/m"))
                    rows.append(("Deviation", f"{metrics.get('dev_pct', 0.0):.2f}%"))

            rows.append(("Status", metrics.get("status", "UNKNOWN")))
            rows.append(("Runtime", f"{metrics.get('runtime_ms', 0.0):.1f} ms"))

            self.tbl_results.setRowCount(len(rows))
            for r, (k, v) in enumerate(rows):
                item_k = QTableWidgetItem(k)
                item_v = QTableWidgetItem(v)
                if k == "Status":
                    font = QFont()
                    font.setBold(True)
                    item_v.setFont(font)
                    if v == "PASS":
                        item_v.setForeground(QColor(76, 175, 80))
                    elif v == "FAIR":
                        item_v.setForeground(QColor(255, 152, 0))
                    else:
                        item_v.setForeground(QColor(244, 67, 54))
                self.tbl_results.setItem(r, 0, item_k)
                self.tbl_results.setItem(r, 1, item_v)

        def _on_export_annotation(self) -> None:
            if not self.current_image_path:
                return
            default_fn = DEFAULT_ANNOTATION_DIR / self.annotation.dataset / f"{self.current_image_path.stem}.json"
            fn, _ = QFileDialog.getSaveFileName(
                self, "Export Annotation & Contours", str(default_fn), "JSON Files (*.json)"
            )
            if fn:
                try:
                    self.annotation.save(fn)
                    QMessageBox.information(self, "Export Successful", f"Annotation and contours saved to:\n{fn}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to export annotation:\n{e}")

        def _on_import_annotation(self) -> None:
            fn, _ = QFileDialog.getOpenFileName(
                self, "Import Annotation / Contours", str(DEFAULT_ANNOTATION_DIR), "JSON Files (*.json)"
            )
            if fn:
                try:
                    loaded = BenchmarkAnnotation.load(fn)
                    self.annotation = loaded
                    self.txt_dataset.setText(loaded.dataset)
                    self.txt_pipeline.setText(loaded.pipeline)
                    if loaded.needle_diameter_mm:
                        self.txt_needle_dia.setText(str(loaded.needle_diameter_mm))
                    if loaded.px_per_mm:
                        self.txt_px_mm.setText(str(loaded.px_per_mm))
                    gt = loaded.ground_truth
                    if "contact_angle_deg" in gt and gt["contact_angle_deg"] is not None:
                        self.txt_expected_ca.setText(f"{gt['contact_angle_deg']:.2f}")
                    if "surface_tension_mN_m" in gt and gt["surface_tension_mN_m"] is not None:
                        self.txt_expected_gamma.setText(f"{gt['surface_tension_mN_m']:.2f}")

                    self.canvas.set_geometry(loaded)
                    self._sync_inputs_from_annotation()
                    QMessageBox.information(self, "Import Successful", f"Annotation loaded from:\n{fn}")
                except Exception as e:
                    QMessageBox.critical(self, "Import Error", f"Failed to import annotation:\n{e}")


# ==============================================================================
# 5. CLI Batch Runner & Entry Point
# ==============================================================================


def run_cli_batch(args: argparse.Namespace) -> int:
    """Run headless benchmark execution from command-line arguments."""
    dataset = args.dataset
    export_dir = Path(args.export_dir) if args.export_dir else None

    # Discover images
    images: list[Path] = []
    if args.image:
        images = [Path(args.image)]
    elif dataset:
        target_dir = BENCHMARK_DIR / dataset
        if not target_dir.exists():
            print(f"Error: Dataset directory not found: {target_dir}")
            return 1
        for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
            images.extend(target_dir.glob(ext))
        images.sort()

    if not images:
        print("No images found to process.")
        return 1

    if args.limit and args.limit > 0:
        images = images[: args.limit]

    print("\n=======================================================")
    print("  MENIPY MANUAL BENCHMARK & REPRODUCIBILITY ENGINE")
    print(f"  Target: {len(images)} images | Dataset: {dataset or 'custom'}")
    print("=======================================================")
    print(f"{'Filename':<30} {'Expected':<12} {'Computed':<12} {'Error/Dev':<12} {'Status'}")
    print("-" * 72)

    pass_count = 0
    total_count = 0

    for img_path in images:
        total_count += 1

        # Check if an existing annotation exists
        ann_path = (
            Path(args.import_annotation)
            if args.import_annotation
            else (DEFAULT_ANNOTATION_DIR / (dataset or "custom") / f"{img_path.stem}.json")
        )

        if ann_path.exists():
            ann = BenchmarkAnnotation.load(ann_path)
        else:
            ann = get_known_benchmark_preset(img_path)

        # CLI overrides
        if args.pipeline:
            ann.pipeline = args.pipeline
        if args.substrate:
            parts = [float(v.strip()) for v in args.substrate.split(",")]
            if len(parts) == 4:
                ann.substrate_line = ((parts[0], parts[1]), (parts[2], parts[3]))
        if args.needle_diameter:
            ann.needle_diameter_mm = args.needle_diameter
        if args.px_per_mm:
            ann.px_per_mm = args.px_per_mm
        if args.expected_ca is not None:
            ann.ground_truth["contact_angle_deg"] = args.expected_ca
        if args.expected_gamma is not None:
            ann.ground_truth["surface_tension_mN_m"] = args.expected_gamma

        try:
            _ctx, metrics = run_pipeline_with_annotation(img_path, ann, use_auto_calibration_seed=True)

            status = metrics.get("status", "UNKNOWN")
            if status == "PASS":
                pass_count += 1

            if ann.pipeline == "sessile":
                exp_str = f"{metrics.get('expected_ca', 0.0):.2f}°" if "expected_ca" in metrics else "N/A"
                comp_str = f"{metrics.get('contact_angle_deg', 0.0):.2f}°" if "contact_angle_deg" in metrics else "N/A"
                err_str = f"{metrics.get('error_deg', 0.0):.2f}°" if "error_deg" in metrics else "N/A"
            else:
                exp_str = f"{metrics.get('expected_gamma', 0.0):.2f}" if "expected_gamma" in metrics else "N/A"
                comp_str = f"{metrics.get('surface_tension_mN_m', 0.0):.2f}" if "surface_tension_mN_m" in metrics else "N/A"
                err_str = f"{metrics.get('dev_pct', 0.0):.2f}%" if "dev_pct" in metrics else "N/A"

            print(f"{img_path.name[:28]:<30} {exp_str:<12} {comp_str:<12} {err_str:<12} [{status}]")

            # Export annotation with contours if requested
            if export_dir:
                out_p = export_dir / (dataset or "custom") / f"{img_path.stem}.json"
                ann.save(out_p)

        except Exception as e:
            print(f"{img_path.name[:28]:<30} ERROR: {e}")

    print("-" * 72)
    print(f"Summary: {pass_count}/{total_count} passed ({pass_count/max(1, total_count)*100:.1f}%)")
    if export_dir:
        print(f"Exported annotations with contours saved to: {export_dir}")

    return 0 if pass_count == total_count else (0 if not args.verify else 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Menipy Manual Benchmark & Contour Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", help="Launch interactive PySide6 graphical tool")
    parser.add_argument("--image", type=str, help="Path to single image to analyze")
    parser.add_argument("--dataset", type=str, help="Dataset folder name inside data/benchmarks/ (e.g. conan_ml, pendant, ucla_pendant)")
    parser.add_argument("--pipeline", choices=["sessile", "pendant"], help="Force pipeline type")
    parser.add_argument("--substrate", type=str, help="Manual substrate line coordinates: 'x1,y1,x2,y2'")
    parser.add_argument("--needle-diameter", type=float, help="Physical needle diameter in mm")
    parser.add_argument("--px-per-mm", type=float, help="Optical calibration scale in px/mm")
    parser.add_argument("--expected-ca", type=float, help="Expected contact angle in degrees")
    parser.add_argument("--expected-gamma", type=float, help="Expected surface tension in mN/m")
    parser.add_argument("--import-annotation", type=str, help="Path to JSON annotation to load")
    parser.add_argument("--export-dir", type=str, help="Directory to export JSON annotations with contours")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to process")
    parser.add_argument("--verify", action="store_true", help="Exit with non-zero status code if any benchmark fails")

    args = parser.parse_args()

    # Launch GUI if explicitly requested or if no CLI image/dataset arguments provided
    if args.gui or (not args.image and not args.dataset):
        if not HAS_PYSIDE6:
            print("PySide6 is not available. Running in CLI mode.")
            parser.print_help()
            return 1

        app = QApplication.instance() or QApplication(sys.argv)
        initial_img = args.image
        if not initial_img and (BENCHMARK_DIR / "pendant" / "water_in_air002.png").exists():
            initial_img = str(BENCHMARK_DIR / "pendant" / "water_in_air002.png")

        window = ManualBenchmarkWindow(initial_image=initial_img)
        window.show()
        return app.exec()

    return run_cli_batch(args)


if __name__ == "__main__":
    sys.exit(main())
