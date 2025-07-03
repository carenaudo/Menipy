"""Main window module for Menipy GUI."""

from pathlib import Path

import numpy as np
import cv2

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QAction,
    QPainter,
    QPainterPath,
    QPen,
    QColor,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QTabWidget,
)
import pandas as pd

from .controls import (
    ZoomControl,
    ParameterPanel,
    MetricsPanel,
    DropAnalysisPanel,
)
from .image_view import ImageView

from ..processing.reader import load_image
from ..processing import (
    detect_droplet,
    detect_sessile_droplet,
    detect_pendant_droplet,
    segmentation,
)
from ..processing.segmentation import find_contours
from ..utils import (
    get_calibration,
    pixels_to_mm,
    auto_calibrate,
    calibrate_from_points,
)
from ..models.properties import (
    droplet_volume,
    estimate_surface_tension,
    contact_angle_from_mask,
)
from ..models.geometry import fit_circle
from ..analysis import (
    detect_vertical_edges,
    extract_external_contour,
    compute_drop_metrics,
)
from .overlay import draw_drop_overlay


class MainWindow(QMainWindow):
    """Main application window with image view and control panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Menipy")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create widgets, menus, and layouts."""
        splitter = QSplitter()

        # Image display area
        self.graphics_view = ImageView()
        self.graphics_scene = self.graphics_view.scene()
        splitter.addWidget(self.graphics_view)

        # Control panel wrapped in tabs
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        classic_widget = QWidget()
        classic_layout = QVBoxLayout(classic_widget)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Otsu", "Adaptive"])
        classic_layout.addWidget(self.algorithm_combo)

        self.zoom_control = ZoomControl()
        self.zoom_control.zoomChanged.connect(self.set_zoom)
        classic_layout.addWidget(self.zoom_control)

        self.parameter_panel = ParameterPanel()
        classic_layout.addWidget(self.parameter_panel)

        self.metrics_panel = MetricsPanel()
        classic_layout.addWidget(self.metrics_panel)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_image)
        classic_layout.addWidget(self.process_button)

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_parameters)
        classic_layout.addWidget(self.calculate_button)

        self.draw_button = QPushButton("Draw Model")
        self.draw_button.clicked.connect(self.draw_model)
        classic_layout.addWidget(self.draw_button)

        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.clicked.connect(lambda: self.save_csv())
        classic_layout.addWidget(self.save_csv_button)

        classic_layout.addStretch()
        self.tabs.addTab(classic_widget, "Classic")

        self.analysis_panel = DropAnalysisPanel()
        self.tabs.addTab(self.analysis_panel, "Drop Analysis")

        # Drop analysis button connections
        self.analysis_panel.needle_region_button.clicked.connect(
            lambda: self.set_needle_mode(True)
        )
        self.analysis_panel.drop_region_button.clicked.connect(
            lambda: self.set_drop_mode(True)
        )
        self.analysis_panel.detect_needle_button.clicked.connect(self.detect_needle)
        self.analysis_panel.analyze_button.clicked.connect(self.analyze_drop_image)

        self.setCentralWidget(splitter)

        # Menu actions
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(lambda: self.open_image())
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)

        save_action = QAction("Save Annotated Image", self)
        save_action.triggered.connect(lambda: self.save_annotated_image())
        file_menu.addAction(save_action)

        tools_menu = self.menuBar().addMenu("Tools")

        self.use_ml_action = QAction("Use ML Segmentation", self)
        self.use_ml_action.setCheckable(True)
        tools_menu.addAction(self.use_ml_action)

        self.mask_item = None
        self.contour_items = []
        self.calibration_rect_item = None
        self.calibration_rect = None
        self.calibration_line_item = None
        self.calibration_line = None
        self.apex_item = None
        self.contact_line_item = None
        self._calib_start = None
        self.roi_rect_item = None
        self.roi_rect = None
        self._roi_start = None
        self._default_press = self.graphics_view.mousePressEvent
        self._default_move = self.graphics_view.mouseMoveEvent
        self._default_release = self.graphics_view.mouseReleaseEvent

        self.last_mask = None
        self.mask_offset = (0, 0)
        self.model_item = None

        # Drop analysis state
        self.needle_rect_item = None
        self.needle_rect = None
        self._needle_start = None
        self.drop_rect_item = None
        self.drop_rect = None
        self._drop_start = None
        self.needle_axis_item = None
        self.needle_edge_items = []
        self.drop_contour_item = None
        self.drop_axis_item = None
        self.diameter_item = None
        self.apex_dot_item = None
        self.px_per_mm_drop = 0.0

        self.parameter_panel.calibration_mode.toggled.connect(
            self.set_calibration_mode
        )
        self.parameter_panel.manual_toggle.toggled.connect(
            lambda _: self.set_calibration_mode(self.parameter_panel.is_calibration_enabled())
        )
        self.parameter_panel.calibrate_button.clicked.connect(self.open_calibration)
        self.parameter_panel.roi_mode.toggled.connect(self.set_roi_mode)

    def open_image(self) -> None:
        """Open an image file and display it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()), "Images (*.png *.jpg *.bmp)"
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        self.image = load_image(path)
        if self.image.ndim == 3:
            rgb = QImage(
                self.image.data,
                self.image.shape[1],
                self.image.shape[0],
                self.image.strides[0],
                QImage.Format_BGR888,
            )
        else:
            rgb = QImage(
                self.image.data,
                self.image.shape[1],
                self.image.shape[0],
                self.image.strides[0],
                QImage.Format_Grayscale8,
            )
        pixmap = QPixmap.fromImage(rgb)
        self.graphics_scene.clear()
        self.needle_rect_item = None
        self.drop_rect_item = None
        self.needle_rect = None
        self.drop_rect = None
        self.needle_axis_item = None
        self.needle_edge_items = []
        self.analysis_panel.set_regions(needle=None, drop=None)
        self.graphics_view.set_pixmap(pixmap)
        self.pixmap_item = self.graphics_view.pixmap_item
        self.adjustSize()
        self.set_zoom(self.zoom_control.slider.value() / 100.0)


    def process_image(self) -> None:
        """Run segmentation on the loaded image and overlay the mask."""
        if getattr(self, "image", None) is None:
            return
        image = self.image
        if self.roi_rect is not None:
            x1, y1, x2, y2 = map(int, self.roi_rect)
        else:
            x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
        roi = (x1, y1, x2 - x1, y2 - y1)

        # Clear previous markers
        if self.apex_item is not None:
            self.graphics_scene.removeItem(self.apex_item)
            self.apex_item = None
        if self.contact_line_item is not None:
            self.graphics_scene.removeItem(self.contact_line_item)
            self.contact_line_item = None

        cal = get_calibration()
        px_to_mm = 1.0 / cal.pixels_per_mm
        try:
            mode_sel = self.parameter_panel.detection_mode()
            if mode_sel == "sessile":
                droplet = detect_sessile_droplet(self.image, roi, px_to_mm)
                mode = "sessile"
            else:
                droplet = detect_pendant_droplet(self.image, roi, px_to_mm)
                mode = "pendant"
        except ValueError as exc:
            QMessageBox.warning(self, "Detection", str(exc))
            self.last_mask = None
            return

        self.last_mask = droplet.mask
        self.mask_offset = (x1, y1)
        if self.model_item is not None:
            self.graphics_scene.removeItem(self.model_item)
            self.model_item = None

        mask_img = QImage(
            droplet.mask.data,
            droplet.mask.shape[1],
            droplet.mask.shape[0],
            droplet.mask.strides[0],
            QImage.Format_Grayscale8,
        )
        mask_pix = QPixmap.fromImage(mask_img)
        if self.mask_item is not None:
            self.graphics_scene.removeItem(self.mask_item)
        self.mask_item = self.graphics_scene.addPixmap(mask_pix)
        self.mask_item.setOffset(x1, y1)
        self.mask_item.setOpacity(0.4)

        for item in self.contour_items:
            self.graphics_scene.removeItem(item)
        self.contour_items.clear()

        path = QPainterPath()
        c = droplet.contour_px
        path.moveTo(*c[0])
        for point in c[1:]:
            path.lineTo(*point)
        pen = QPen(QColor("red"))
        pen.setWidth(2)
        item = self.graphics_scene.addPath(path, pen)
        self.contour_items.append(item)

        # Apex and contact line markers from silhouette
        if self.apex_item is not None:
            self.graphics_scene.removeItem(self.apex_item)
            self.apex_item = None
        if self.contact_line_item is not None:
            self.graphics_scene.removeItem(self.contact_line_item)
            self.contact_line_item = None

        pen = QPen(QColor("yellow"))
        brush = QColor("yellow")
        self.apex_item = self.graphics_scene.addEllipse(
            droplet.apex_px[0] - 3,
            droplet.apex_px[1] - 3,
            6,
            6,
            pen,
            brush,
        )

        row_y = droplet.contact_px[1] - y1
        if 0 <= row_y < droplet.mask.shape[0]:
            cols = np.where(droplet.mask[row_y] > 0)[0]
            if cols.size >= 2:
                pen = QPen(QColor("cyan"))
                pen.setWidth(2)
                self.contact_line_item = self.graphics_scene.addLine(
                    droplet.contact_px[0],
                    droplet.contact_px[1],
                    droplet.contact_px[2],
                    droplet.contact_px[3],
                    pen,
                )
        # Basic metrics from the silhouette
        cal = get_calibration()
        px_to_mm = 1.0 / cal.pixels_per_mm
        height = droplet.height_mm
        diameter = droplet.r_max_mm * 2.0
        volume = droplet_volume(droplet.mask, px_to_mm=px_to_mm)

        self.metrics_panel.set_metrics(
            ift=0.0,
            wo=0.0,
            volume=volume if volume is not None else 0.0,
            contact_angle=0.0,
            height=height,
            diameter=diameter,
            mode=mode,
        )

    def calculate_parameters(self) -> None:
        """Calculate surface tension and contact angle from the mask."""
        if self.last_mask is None:
            self.process_image()
        if self.last_mask is None:
            return

        values = self.parameter_panel.values()
        cal = get_calibration()
        gamma = estimate_surface_tension(
            self.last_mask,
            values["air_density"],
            values["liquid_density"],
            px_to_mm=1.0 / cal.pixels_per_mm,
        )
        angle = contact_angle_from_mask(self.last_mask)

        self.metrics_panel.set_metrics(ift=gamma, contact_angle=angle)

    def draw_model(self) -> None:
        """Draw a fitted circle model overlay."""
        if self.last_mask is None:
            self.process_image()
        if self.last_mask is None:
            return

        contours = find_contours(self.last_mask)
        if not contours:
            return
        contour = max(contours, key=lambda c: c.shape[0]).astype(float)
        contour += np.array(self.mask_offset)
        center, radius = fit_circle(contour)
        theta = np.linspace(0, 2 * np.pi, 200)
        path = QPainterPath()
        path.moveTo(center[0] + radius * np.cos(theta[0]), center[1] + radius * np.sin(theta[0]))
        for t in theta[1:]:
            path.lineTo(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t))
        pen = QPen(QColor("magenta"))
        pen.setWidth(2)
        if self.model_item is not None:
            self.graphics_scene.removeItem(self.model_item)
        self.model_item = self.graphics_scene.addPath(path, pen)

    def save_annotated_image(self, path: Path | None = None) -> None:
        """Save the current scene (image + overlays) to a file."""
        if self.pixmap_item is None:
            return
        if path is None:
            p, _ = QFileDialog.getSaveFileName(
                self,
                "Save Annotated Image",
                str(Path.home() / "annotated.png"),
                "Images (*.png *.jpg *.bmp)",
            )
            if not p:
                return
            path = Path(p)
        image = QImage(
            self.graphics_view.viewport().size(), QImage.Format_ARGB32
        )
        painter = QPainter(image)
        self.graphics_view.render(painter)
        painter.end()
        image.save(str(path))

    def save_csv(self, path: Path | None = None) -> None:
        """Export parameters and metrics to a CSV file."""
        if path is None:
            p, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV",
                str(Path.home() / "results.csv"),
                "CSV Files (*.csv)",
            )
            if not p:
                return
            path = Path(p)

        data = {**self.parameter_panel.values(), **self.metrics_panel.values()}
        pd.DataFrame([data]).to_csv(path, index=False)

    # --- Drop analysis processing -------------------------------------------

    def detect_needle(self) -> None:
        """Detect the needle axis inside the selected region."""
        if getattr(self, "image", None) is None or self.needle_rect is None:
            return
        x1, y1, x2, y2 = map(int, self.needle_rect)
        roi = self.image[y1:y2, x1:x2]
        try:
            top, bottom, length_px = detect_vertical_edges(roi)
        except ValueError as exc:
            QMessageBox.warning(self, "Needle Detection", str(exc))
            return
        self.px_per_mm_drop = length_px / max(self.analysis_panel.needle_length.value(), 1e-6)
        self.analysis_panel.set_metrics(scale=self.px_per_mm_drop)
        axis_x = top[0] + x1
        y_top = top[1] + y1
        y_bottom = bottom[1] + y1
        half_width = length_px / 2.0
        left_x = int(round(axis_x - half_width))
        right_x = int(round(axis_x + half_width))

        if self.needle_axis_item is not None:
            self.graphics_scene.removeItem(self.needle_axis_item)
            self.needle_axis_item = None
        for item in self.needle_edge_items:
            self.graphics_scene.removeItem(item)
        self.needle_edge_items.clear()

        pen = QPen(QColor("yellow"))
        pen.setWidth(2)
        left_item = self.graphics_scene.addLine(left_x, y_top, left_x, y_bottom, pen)
        right_item = self.graphics_scene.addLine(right_x, y_top, right_x, y_bottom, pen)
        self.needle_edge_items = [left_item, right_item]

    def analyze_drop_image(self) -> None:
        """Analyze the drop region and draw overlays."""
        if (
            getattr(self, "image", None) is None
            or self.drop_rect is None
            or self.px_per_mm_drop <= 0
        ):
            return
        x1, y1, x2, y2 = map(int, self.drop_rect)
        roi = self.image[y1:y2, x1:x2]
        try:
            contour = extract_external_contour(roi)
        except ValueError as exc:
            QMessageBox.warning(self, "Drop Analysis", str(exc))
            return
        contour += np.array([x1, y1])
        mode = self.analysis_panel.method_combo.currentText()
        metrics = compute_drop_metrics(
            contour.astype(float), self.px_per_mm_drop, mode
        )
        self.analysis_panel.set_metrics(
            height=metrics["height_mm"],
            diameter=metrics["diameter_mm"],
            volume=metrics["volume_uL"] if metrics["volume_uL"] is not None else 0.0,
            angle=metrics["contact_angle_deg"],
            ift=metrics["ift_mN_m"] if metrics["ift_mN_m"] is not None else 0.0,
            wo=metrics["wo"],
            radius=metrics["radius_apex_mm"],
        )
        y_min = int(contour[:, 1].min())
        y_max = int(contour[:, 1].max())
        diameter_line = (
            metrics["diameter_line"][0],
            metrics["diameter_line"][1],
        )
        axis_line = (
            metrics["apex"],
            (metrics["apex"][0], y_min if mode == "pendant" else y_max),
        )
        if self.drop_contour_item is not None:
            self.graphics_scene.removeItem(self.drop_contour_item)
        if self.diameter_item is not None:
            self.graphics_scene.removeItem(self.diameter_item)
        if self.drop_axis_item is not None:
            self.graphics_scene.removeItem(self.drop_axis_item)
        if self.apex_dot_item is not None:
            self.graphics_scene.removeItem(self.apex_dot_item)

        overlay = draw_drop_overlay(
            self.image,
            contour,
            diameter_line=diameter_line,
            axis_line=axis_line,
            contact_line=metrics.get("contact_line"),
            apex=metrics["apex"],
        )
        self.drop_contour_item = self.graphics_scene.addPixmap(overlay)

    # --- Calibration drawing handling -------------------------------------------------

    def set_calibration_mode(self, enabled: bool) -> None:
        """Enable or disable interactive calibration drawing."""
        if enabled:
            self.graphics_view.setCursor(Qt.CrossCursor)
            method = self.parameter_panel.calibration_method()
            if method == "manual":
                self.graphics_view.mousePressEvent = self._line_press
                self.graphics_view.mouseMoveEvent = self._line_move
                self.graphics_view.mouseReleaseEvent = self._line_release
            else:
                self.graphics_view.mousePressEvent = self._box_press
                self.graphics_view.mouseMoveEvent = self._box_move
                self.graphics_view.mouseReleaseEvent = self._box_release
        else:
            self.graphics_view.setCursor(Qt.ArrowCursor)
            self.graphics_view.mousePressEvent = self._default_press
            self.graphics_view.mouseMoveEvent = self._default_move
            self.graphics_view.mouseReleaseEvent = self._default_release
            if self.calibration_rect_item is not None:
                self.graphics_scene.removeItem(self.calibration_rect_item)
                self.calibration_rect_item = None
            self.calibration_rect = None
            if self.calibration_line_item is not None:
                self.graphics_scene.removeItem(self.calibration_line_item)
                self.calibration_line_item = None
            self.calibration_line = None
            self._calib_start = None

    def _box_press(self, event):
        pos = self.graphics_view.mapToScene(event.pos())
        self._calib_start = pos
        if self.calibration_rect_item is not None:
            self.graphics_scene.removeItem(self.calibration_rect_item)
        pen = QPen(QColor("blue"))
        pen.setWidth(2)
        rect = QRectF(pos, pos)
        self.calibration_rect_item = self.graphics_scene.addRect(rect, pen)
        event.accept()

    def _box_move(self, event):
        if self._calib_start is None or self.calibration_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._calib_start, pos).normalized()
        self.calibration_rect_item.setRect(rect)
        event.accept()

    def _box_release(self, event):
        if self._calib_start is None or self.calibration_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._calib_start, pos).normalized()
        self.calibration_rect_item.setRect(rect)
        self.calibration_rect = (
            rect.left(),
            rect.top(),
            rect.right(),
            rect.bottom(),
        )
        self._calib_start = None
        event.accept()

    # --- Manual line drawing -------------------------------------------------

    def _line_press(self, event):
        pos = self.graphics_view.mapToScene(event.pos())
        self._calib_start = pos
        if self.calibration_line_item is not None:
            self.graphics_scene.removeItem(self.calibration_line_item)
        pen = QPen(QColor("blue"))
        pen.setWidth(2)
        self.calibration_line_item = self.graphics_scene.addLine(pos.x(), pos.y(), pos.x(), pos.y(), pen)
        event.accept()

    def _line_move(self, event):
        if self._calib_start is None or self.calibration_line_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        line = self.calibration_line_item.line()
        line.setP2(pos)
        self.calibration_line_item.setLine(line)
        event.accept()

    def _line_release(self, event):
        if self._calib_start is None or self.calibration_line_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        line = self.calibration_line_item.line()
        line.setP2(pos)
        self.calibration_line_item.setLine(line)
        self.calibration_line = (
            line.x1(),
            line.y1(),
            line.x2(),
            line.y2(),
        )
        self._calib_start = None
        event.accept()

    # --- ROI drawing ---------------------------------------------------------

    def set_roi_mode(self, enabled: bool) -> None:
        """Enable or disable ROI drawing mode."""
        if enabled:
            self.graphics_view.setCursor(Qt.CrossCursor)
            self.graphics_view.mousePressEvent = self._roi_press
            self.graphics_view.mouseMoveEvent = self._roi_move
            self.graphics_view.mouseReleaseEvent = self._roi_release
        else:
            self.graphics_view.setCursor(Qt.ArrowCursor)
            self.graphics_view.mousePressEvent = self._default_press
            self.graphics_view.mouseMoveEvent = self._default_move
            self.graphics_view.mouseReleaseEvent = self._default_release
            if self.roi_rect_item is not None:
                self.graphics_scene.removeItem(self.roi_rect_item)
                self.roi_rect_item = None
            self.roi_rect = None
            self._roi_start = None

    def _roi_press(self, event):
        pos = self.graphics_view.mapToScene(event.pos())
        self._roi_start = pos
        if self.roi_rect_item is not None:
            self.graphics_scene.removeItem(self.roi_rect_item)
        pen = QPen(QColor("green"))
        pen.setWidth(2)
        rect = QRectF(pos, pos)
        self.roi_rect_item = self.graphics_scene.addRect(rect, pen)
        event.accept()

    def _roi_move(self, event):
        if self._roi_start is None or self.roi_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._roi_start, pos).normalized()
        self.roi_rect_item.setRect(rect)
        event.accept()

    def _roi_release(self, event):
        if self._roi_start is None or self.roi_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._roi_start, pos).normalized()
        self.roi_rect_item.setRect(rect)
        self.roi_rect = (
            rect.left(),
            rect.top(),
            rect.right(),
            rect.bottom(),
        )
        self._roi_start = None
        event.accept()

    # --- Drop analysis drawing modes ---------------------------------------

    def set_needle_mode(self, enabled: bool) -> None:
        """Enable rectangle drawing for the needle ROI."""
        if enabled:
            self.graphics_view.setCursor(Qt.CrossCursor)
            self.graphics_view.mousePressEvent = self._needle_press
            self.graphics_view.mouseMoveEvent = self._needle_move
            self.graphics_view.mouseReleaseEvent = self._needle_release
        else:
            self.graphics_view.setCursor(Qt.ArrowCursor)
            self.graphics_view.mousePressEvent = self._default_press
            self.graphics_view.mouseMoveEvent = self._default_move
            self.graphics_view.mouseReleaseEvent = self._default_release
            self._needle_start = None

    def _needle_press(self, event):
        pos = self.graphics_view.mapToScene(event.pos())
        self._needle_start = pos
        if self.needle_rect_item is not None:
            self.graphics_scene.removeItem(self.needle_rect_item)
        pen = QPen(QColor("blue"))
        pen.setWidth(2)
        rect = QRectF(pos, pos)
        self.needle_rect_item = self.graphics_scene.addRect(rect, pen)
        self.needle_rect_item.setZValue(1)
        event.accept()

    def _needle_move(self, event):
        if self._needle_start is None or self.needle_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._needle_start, pos).normalized()
        self.needle_rect_item.setRect(rect)
        event.accept()

    def _needle_release(self, event):
        if self._needle_start is None or self.needle_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._needle_start, pos).normalized()
        self.needle_rect_item.setRect(rect)
        self.needle_rect = (
            rect.left(),
            rect.top(),
            rect.right(),
            rect.bottom(),
        )
        self.analysis_panel.set_regions(needle=self.needle_rect)
        self._needle_start = None
        self.set_needle_mode(False)
        event.accept()

    def set_drop_mode(self, enabled: bool) -> None:
        """Enable rectangle drawing for the drop ROI."""
        if enabled:
            self.graphics_view.setCursor(Qt.CrossCursor)
            self.graphics_view.mousePressEvent = self._drop_press
            self.graphics_view.mouseMoveEvent = self._drop_move
            self.graphics_view.mouseReleaseEvent = self._drop_release
        else:
            self.graphics_view.setCursor(Qt.ArrowCursor)
            self.graphics_view.mousePressEvent = self._default_press
            self.graphics_view.mouseMoveEvent = self._default_move
            self.graphics_view.mouseReleaseEvent = self._default_release
            self._drop_start = None

    def _drop_press(self, event):
        pos = self.graphics_view.mapToScene(event.pos())
        self._drop_start = pos
        if self.drop_rect_item is not None:
            self.graphics_scene.removeItem(self.drop_rect_item)
        pen = QPen(QColor("green"))
        pen.setWidth(2)
        rect = QRectF(pos, pos)
        self.drop_rect_item = self.graphics_scene.addRect(rect, pen)
        self.drop_rect_item.setZValue(1)
        event.accept()

    def _drop_move(self, event):
        if self._drop_start is None or self.drop_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._drop_start, pos).normalized()
        self.drop_rect_item.setRect(rect)
        event.accept()

    def _drop_release(self, event):
        if self._drop_start is None or self.drop_rect_item is None:
            return
        pos = self.graphics_view.mapToScene(event.pos())
        rect = QRectF(self._drop_start, pos).normalized()
        self.drop_rect_item.setRect(rect)
        self.drop_rect = (
            rect.left(),
            rect.top(),
            rect.right(),
            rect.bottom(),
        )
        self.analysis_panel.set_regions(drop=self.drop_rect)
        self._drop_start = None
        self.set_drop_mode(False)
        event.accept()

    def open_calibration(self) -> None:
        """Open a dialog to calibrate pixel size."""
        if getattr(self, "image", None) is None:
            QMessageBox.information(self, "Calibration", "Load an image first")
            return
        method = self.parameter_panel.calibration_method()
        if method == "manual":
            if self.calibration_line is None:
                QMessageBox.information(self, "Calibration", "Draw calibration line first")
                return
            x1, y1, x2, y2 = self.calibration_line
            length_mm = self.parameter_panel.calibration_length()
            calibrate_from_points((x1, y1), (x2, y2), length_mm)
            cal = get_calibration()
            self.parameter_panel.set_scale_display(cal.pixels_per_mm)
            QMessageBox.information(
                self,
                "Calibration",
                f"Calibration set to {cal.pixels_per_mm:.2f} px/mm",
            )
        else:
            if self.calibration_rect is None:
                QMessageBox.information(self, "Calibration", "Draw calibration box first")
                return
            length_mm = self.parameter_panel.calibration_length()
            try:
                auto_calibrate(self.image, self.calibration_rect, length_mm)
            except Exception as exc:
                QMessageBox.warning(self, "Calibration", str(exc))
                return
            cal = get_calibration()
            self.parameter_panel.set_scale_display(cal.pixels_per_mm)
            QMessageBox.information(
                self,
                "Calibration",
                f"Calibration set to {cal.pixels_per_mm:.2f} px/mm",
            )

    def set_zoom(self, factor: float) -> None:
        """Scale the graphics view by ``factor``."""
        self.graphics_view.set_zoom(factor)


def main():
    """Launch the Menipy GUI application."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
