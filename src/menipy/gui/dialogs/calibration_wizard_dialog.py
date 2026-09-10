"""
Calibration Wizard Dialog for automatic region detection.

This dialog provides a step-by-step wizard for automatic detection of
substrate, needle, drop, and ROI regions with live preview.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from menipy.common.auto_calibrator import CalibrationResult

logger = logging.getLogger(__name__)


class CalibrationWizardDialog(QDialog):
    """
    Modal wizard dialog for automatic calibration with preview.

    Displays detected regions overlaid on the image and allows the user
    to accept, reject, or manually adjust each detected region.
    """

    # Emitted when user accepts calibration results
    calibration_complete = Signal(object)  # CalibrationResult

    # Colors for overlay visualization
    SUBSTRATE_COLOR = QColor(255, 0, 255)  # Magenta
    NEEDLE_COLOR = QColor(0, 0, 255)  # Blue
    DROP_COLOR = QColor(0, 255, 0)  # Green
    ROI_COLOR = QColor(255, 255, 0)  # Yellow
    CONTACT_COLOR = QColor(255, 0, 0)  # Red

    def __init__(
        self,
        image: np.ndarray,
        pipeline_name: str = "sessile",
        parent: QWidget | None = None,
    ) -> None:
        """
        Initialize the calibration wizard dialog.

        Args:
            image: Input image (BGR format)
            pipeline_name: Pipeline type for detection strategy
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(f"Calibration Wizard - {pipeline_name.title()}")
        self.setMinimumSize(800, 600)
        self.setModal(True)

        from PySide6.QtWidgets import QApplication

        from menipy.gui.services.pipeline_runner import PipelineRunner

        self._runner = getattr(parent, "runner", None) or PipelineRunner(
            QApplication.instance()
        )
        self._runner.finished.connect(self._on_detection_finished)
        self._detection_job = None
        self._manual_revision = 0
        self._closed = False
        self._input_image = image
        self.original_image = (
            image.copy()
            if isinstance(image, np.ndarray)
            else np.zeros((1, 1, 3), dtype=np.uint8)
        )
        self.pipeline_name = pipeline_name.lower()
        self.result: CalibrationResult | None = None

        # Region enable flags
        self._region_enabled = {
            "substrate": True,
            "needle": True,
            "drop": True,
            "contact": True,  # Contact points
            "roi": True,
        }

        # Drawing state for manual regions
        self._drawing_mode = None  # None, "substrate", "needle", or "roi"
        self._draw_start_point = None
        self._draw_end_point = None

        # Legacy compatibility
        self._drawing_substrate = False
        self._substrate_start_point = None
        self._substrate_end_point = None

        self._build_ui()
        self._wire_signals()
        if not isinstance(image, np.ndarray):
            QTimer.singleShot(0, self._load_initial_preview)

    def _load_initial_preview(self):
        if self._closed:
            return
        from menipy.gui.services.calibration_service import calibration_preview_task
        from menipy.gui.services.pipeline_runner import RunRequest

        request = RunRequest.create(
            self.pipeline_name,
            {"image": self._input_image},
            operation="calibration",
            revision=str(self._manual_revision),
        )
        self._detection_job = request.job_id
        self._detect_btn.setEnabled(False)
        self._progress.show()
        try:
            self._runner.submit(request, calibration_preview_task)
        except Exception as exc:
            self._detection_job = None
            self._progress.hide()
            self._detect_btn.setEnabled(True)
            self._confidence_label.setText(f"Could not load preview: {exc}")

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QLabel(
            f"<b>Auto-Calibration for {self.pipeline_name.title()} Pipeline</b><br>"
            "Click 'Detect' to automatically find regions in the image."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Progress bar (initially hidden)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # Indeterminate
        self._progress.hide()
        layout.addWidget(self._progress)

        # Main content area
        content_layout = QHBoxLayout()

        # Left: Image preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Scroll area for large images
        scroll = QScrollArea()
        self._preview_scroll = scroll
        self._fit_mode = True
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(320, 260)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        scroll.setWidget(self._preview_label)
        preview_layout.addWidget(scroll)

        zoom_layout = QHBoxLayout()
        self._fit_btn = QPushButton("Fit to Window")
        self._actual_btn = QPushButton("100%")
        zoom_layout.addWidget(self._fit_btn)
        zoom_layout.addWidget(self._actual_btn)
        zoom_layout.addStretch()
        preview_layout.addLayout(zoom_layout)

        # Enable mouse tracking for drawing
        self._preview_label.setMouseTracking(True)
        self._preview_label.mousePressEvent = self._on_preview_mouse_press
        self._preview_label.mouseMoveEvent = self._on_preview_mouse_move
        self._preview_label.mouseReleaseEvent = self._on_preview_mouse_release

        content_layout.addWidget(preview_group, stretch=3)

        # Right: Detection results
        results_group = QGroupBox("Detected Regions")
        results_group.setMinimumWidth(255)
        results_layout = QVBoxLayout(results_group)

        # Region checkboxes with status
        self._region_widgets = {}

        regions = [
            ("substrate", "Substrate Line", self.SUBSTRATE_COLOR),
            ("needle", "Needle Region", self.NEEDLE_COLOR),
            ("drop", "Drop Contour", self.DROP_COLOR),
            ("contact", "Contact Points", self.CONTACT_COLOR),
            ("roi", "ROI Rectangle", self.ROI_COLOR),
        ]

        for region_id, label, color in regions:
            widget = self._create_region_widget(region_id, label, color)
            results_layout.addWidget(widget)
            self._region_widgets[region_id] = widget

        results_layout.addStretch()

        # Confidence display
        self._confidence_label = QLabel("Confidence: --")
        self._confidence_label.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(self._confidence_label)

        content_layout.addWidget(results_group, stretch=1)
        layout.addLayout(content_layout)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self._detect_btn = QPushButton("🔍 Detect")
        self._detect_btn.setMinimumHeight(40)
        self._detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setMinimumHeight(35)

        self._apply_btn = QPushButton("✓ Apply All")
        self._apply_btn.setMinimumHeight(40)
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #7ED321;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #6BC01A;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

        button_layout.addWidget(self._detect_btn)
        button_layout.addStretch()
        button_layout.addWidget(self._cancel_btn)
        button_layout.addWidget(self._apply_btn)
        layout.addLayout(button_layout)

        # Show original image initially
        self._show_original_image()

    def _create_region_widget(
        self, region_id: str, label: str, color: QColor
    ) -> QWidget:
        """Create a widget for a single region with checkbox and status."""
        widget = QFrame()
        widget.setFrameShape(QFrame.StyledPanel)
        outer = QVBoxLayout(widget)
        layout = QHBoxLayout()
        outer.addLayout(layout)
        layout.setContentsMargins(5, 5, 5, 5)

        # Color indicator
        color_label = QLabel()
        color_label.setFixedSize(16, 16)
        color_label.setStyleSheet(
            f"background-color: {color.name()}; border: 1px solid #333; border-radius: 3px;"
        )
        layout.addWidget(color_label)

        # Checkbox
        checkbox = QCheckBox(label)
        checkbox.setChecked(True)
        checkbox.setProperty("region_id", region_id)
        layout.addWidget(checkbox)

        # Status label
        status = QLabel("--")
        status.setAlignment(Qt.AlignRight)
        status.setMinimumWidth(60)
        status.setWordWrap(True)
        outer.addWidget(status)

        widget.checkbox = checkbox
        widget.status = status
        widget.color_label = color_label
        widget.draw_btn = None  # Will be set for drawable regions

        # Add "Draw" button for drawable regions (substrate=line, needle=line, roi=rectangle)
        drawable_regions = {
            "substrate": ("✏ Draw", "Click to manually draw substrate line"),
            "needle": ("✏ Draw", "Click to manually draw needle line"),
            "roi": ("▢ Draw", "Click to manually draw ROI rectangle"),
        }

        if region_id in drawable_regions:
            btn_text, tooltip = drawable_regions[region_id]
            draw_btn = QPushButton(btn_text)
            draw_btn.setToolTip(tooltip)
            draw_btn.setMaximumWidth(60)
            draw_btn.setStyleSheet("""
                QPushButton {
                    background-color: #E0E0E0;
                    border-radius: 3px;
                    padding: 2px 6px;
                }
                QPushButton:hover {
                    background-color: #FFD700;
                }
                QPushButton:checked {
                    background-color: #FFD700;
                    font-weight: bold;
                }
            """)
            draw_btn.setCheckable(True)
            draw_btn.setProperty("region_id", region_id)
            draw_btn.clicked.connect(self._on_draw_region_clicked)
            layout.addWidget(draw_btn)
            widget.draw_btn = draw_btn

        return widget

    def _wire_signals(self) -> None:
        """Connect UI signals."""
        self._detect_btn.clicked.connect(self.run_detection)
        self._cancel_btn.clicked.connect(self.reject)
        self._apply_btn.clicked.connect(self._on_apply)
        self._fit_btn.clicked.connect(self._fit_preview)
        self._actual_btn.clicked.connect(self._actual_preview)

        # Connect region checkboxes
        for _region_id, widget in self._region_widgets.items():
            widget.checkbox.stateChanged.connect(self._on_region_toggled)

    def showEvent(self, event):
        super().showEvent(event)
        if self._fit_mode:
            QTimer.singleShot(0, self._fit_preview)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_fit_mode", False):
            QTimer.singleShot(0, self._fit_preview)

    def _show_original_image(self) -> None:
        """Display the original image in the preview."""
        self._display_image(self.original_image)

    def _display_image(self, image: np.ndarray) -> None:
        """Convert and display an image in the preview label."""
        if len(image.shape) == 2:
            # Grayscale
            qimg = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                QImage.Format_Grayscale8,
            )
        else:
            # BGR to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            )

        pixmap = QPixmap.fromImage(qimg)
        self._current_pixmap = pixmap
        self._fit_preview()

    def _fit_preview(self) -> None:
        """Scale image to fit in preview area."""
        self._fit_mode = True
        self._preview_scroll.setWidgetResizable(True)
        if hasattr(self, "_current_pixmap"):
            scaled = self._current_pixmap.scaled(
                self._preview_scroll.viewport().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)

    def _actual_preview(self) -> None:
        """Show image at 100% scale."""
        self._fit_mode = False
        self._preview_scroll.setWidgetResizable(False)
        if hasattr(self, "_current_pixmap"):
            self._preview_label.resize(self._current_pixmap.size())
            self._preview_label.setPixmap(self._current_pixmap)

    def run_detection(self):
        from menipy.gui.services.calibration_service import calibration_task
        from menipy.gui.services.pipeline_runner import RunRequest

        if self._runner.busy:
            self._confidence_label.setText("Another operation is running.")
            return
        request = RunRequest.create(
            self.pipeline_name,
            {
                "image": self._input_image,
                "pipeline": self.pipeline_name,
                "manual_result": self.result,
            },
            operation="calibration",
            revision=str(self._manual_revision),
        )
        self._detect_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._progress.show()
        self._detection_job = request.job_id
        try:
            self._runner.submit(request, calibration_task)
        except Exception as exc:
            self._detection_job = None
            self._progress.hide()
            self._detect_btn.setEnabled(True)
            self._confidence_label.setText(f"Could not submit calibration: {exc}")

    def _on_detection_finished(self, completion):
        if completion.request.job_id != self._detection_job:
            return
        self._detection_job = None
        if self._closed:
            return
        self._progress.hide()
        self._detect_btn.setEnabled(True)
        if completion.state != "completed":
            self._confidence_label.setText(completion.error or "Calibration cancelled.")
            return
        if completion.request.revision != str(self._manual_revision):
            self._confidence_label.setText(
                "Calibration discarded; manual regions changed."
            )
            return
        self.original_image, self.result = completion.value
        self._input_image = self.original_image
        if self.result is None:
            self._show_original_image()
        else:
            self._show_results()

    def done(self, result):
        self._closed = True
        if self._detection_job is not None:
            self._runner.cancel(self._detection_job)
        super().done(result)

    def _run_best_auto_calibration(self, runner, *, allow_fallback=True):
        from menipy.gui.services.calibration_service import CalibrationComputation

        return CalibrationComputation(
            self.original_image, self.pipeline_name
        )._run_best_auto_calibration(runner, allow_fallback=allow_fallback)

    def _show_results(self) -> None:
        """Display detection results with overlays."""
        if self.result is None:
            return

        # Create overlay image
        overlay = self._draw_overlays()
        self._display_image(overlay)

        # Update status labels
        self._update_region_statuses()

        # Update confidence
        overall = self.result.confidence_scores.get("overall", 0.0)
        self._confidence_label.setText(f"Overall Confidence: {overall * 100:.0f}%")

        # Enable apply button
        self._apply_btn.setEnabled(True)

    def _draw_overlays(self) -> np.ndarray:
        """Draw detection overlays on the image."""
        overlay = self.original_image.copy()
        result = self.result

        if result is None:
            return overlay

        # Draw substrate line or curved arc
        if self._region_enabled.get("substrate", True):
            prof = getattr(result, "substrate_profile", None)
            if prof and prof.type == "circle_arc":
                pts = prof.sample_points(n_points=100)
                pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    overlay, [pts_i], isClosed=False, color=(255, 0, 255), thickness=2
                )
                r_val = float(prof.parameters.get("radius", 0.0))
                cv2.putText(
                    overlay,
                    f"Substrate Arc (R={r_val:.1f}px)",
                    (10, max(20, int(pts[0, 1]) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )
            elif result.substrate_line:
                p1, p2 = result.substrate_line
                cv2.line(overlay, p1, p2, (255, 0, 255), 2)  # Magenta (BGR)
                cv2.putText(
                    overlay,
                    "Substrate",
                    (10, max(20, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )

        # Draw needle region
        if result.needle_rect and self._region_enabled.get("needle", True):
            x, y, w, h = result.needle_rect
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue (BGR)
            cv2.putText(
                overlay,
                "Needle",
                (x + w + 5, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        # Draw drop contour
        if result.drop_contour is not None and self._region_enabled.get("drop", True):
            boundary = getattr(result, "liquid_boundary", None)
            contour = np.asarray(
                boundary if boundary is not None else result.drop_contour, dtype=np.int32
            )
            if contour.ndim == 2:
                contour = contour.reshape(-1, 1, 2)
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)  # Green (BGR)

        # Draw contact points (separate from drop contour for independent visibility)
        if result.contact_points and self._region_enabled.get("contact", True):
            left, right = result.contact_points
            boundary = getattr(result, "liquid_boundary", None)
            if boundary is not None and len(boundary) >= 2:
                left, right = [tuple(np.rint(point).astype(int)) for point in boundary[[0, -1]]]
            cv2.circle(overlay, left, 5, (0, 0, 255), -1)  # Red
            cv2.circle(overlay, right, 5, (0, 0, 255), -1)
            # Draw connecting line for visibility
            cv2.line(overlay, left, right, (0, 0, 255), 1)

        # Draw ROI rectangle
        if result.roi_rect and self._region_enabled.get("roi", True):
            x, y, w, h = result.roi_rect
            cv2.rectangle(
                overlay,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2,  # Yellow (BGR)
            )
            cv2.putText(
                overlay,
                "ROI",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        # Blend overlay
        alpha = 0.7
        blended = cv2.addWeighted(overlay, alpha, self.original_image, 1 - alpha, 0)

        return blended

    def _update_region_statuses(self) -> None:
        """Update region status labels based on detection results."""
        if self.result is None:
            return

        sub_conf = self.result.confidence_scores.get("substrate", 0.0)
        sub_warn = getattr(self.result, "substrate_warning", False) or sub_conf < 0.75
        if self.result.substrate_line and not sub_warn:
            sub_label = "✓ Found"
        elif self.result.substrate_line and sub_warn:
            sub_label = "⚠ Doubtful"
        else:
            sub_label = "✗ Not found"

        statuses = {
            "substrate": (
                sub_label,
                sub_conf,
            ),
            "needle": (
                "✓ Found" if self.result.needle_rect else "✗ Not found",
                self.result.confidence_scores.get("needle", 0.0),
            ),
            "drop": (
                "✓ Found" if self.result.drop_contour is not None else "✗ Not found",
                self.result.confidence_scores.get("drop", 0.0),
            ),
            "contact": (
                "✓ Found" if self.result.contact_points else "✗ Not found",
                # Contact points share drop confidence as they're detected together
                (
                    self.result.confidence_scores.get("drop", 0.0)
                    if self.result.contact_points
                    else 0.0
                ),
            ),
            "roi": (
                "✓ Found" if self.result.roi_rect else "✗ Not found",
                self.result.confidence_scores.get("roi", 0.0),
            ),
        }

        for region_id, (text, conf) in statuses.items():
            widget = self._region_widgets.get(region_id)
            if widget:
                conf_text = f"({conf * 100:.0f}%)" if conf > 0 else ""
                widget.status.setText(f"{text} {conf_text}")
                if "✓" in text:
                    widget.status.setStyleSheet("color: #1A7F37; font-weight: 600;")
                elif "⚠" in text:
                    widget.status.setStyleSheet("color: #9A6700; font-weight: 600;")
                else:
                    widget.status.setStyleSheet("color: #CF222E; font-weight: 600;")

    def _on_region_toggled(self, state: int) -> None:
        """Handle region checkbox toggle."""
        sender = self.sender()
        if sender:
            region_id = sender.property("region_id")
            self._region_enabled[region_id] = state == Qt.Checked

            # Redraw overlays with updated visibility
            if self.result:
                overlay = self._draw_overlays()
                self._display_image(overlay)

    def _on_apply(self) -> None:
        """Apply calibration results and close dialog."""
        if self.result is None:
            self.reject()
            return

        # Filter result based on enabled regions
        if not self._region_enabled.get("substrate", True):
            self.result.substrate_line = None
        if not self._region_enabled.get("needle", True):
            self.result.needle_rect = None
        if not self._region_enabled.get("drop", True):
            self.result.drop_contour = None
            self.result.liquid_boundary = None
            self.result.contact_points = None
        if not self._region_enabled.get("roi", True):
            self.result.roi_rect = None

        self.calibration_complete.emit(self.result)
        self.accept()

    def get_result(self) -> CalibrationResult | None:
        """Get the calibration result after dialog closes."""
        return self.result

    def _on_draw_region_clicked(self, checked: bool) -> None:
        """Handle draw button click - enter/exit drawing mode for any region."""
        sender = self.sender()
        if not sender:
            return

        region_id = sender.property("region_id")

        if checked:
            # Uncheck any other draw buttons first
            for rid, widget in self._region_widgets.items():
                if widget.draw_btn and rid != region_id:
                    widget.draw_btn.setChecked(False)

            self._drawing_mode = region_id
            self._draw_start_point = None
            self._draw_end_point = None
            self._preview_label.setCursor(Qt.CrossCursor)

            shape = "rectangle" if region_id == "roi" else "line"
            logger.info(
                f"Entering {region_id} drawing mode - click and drag to draw {shape}"
            )
        else:
            self._drawing_mode = None
            self._preview_label.setCursor(Qt.ArrowCursor)

    # Legacy compatibility
    def _on_draw_substrate_clicked(self, checked: bool) -> None:
        """Legacy handler - redirects to unified handler."""
        widget = self._region_widgets.get("substrate")
        if widget and widget.draw_btn:
            widget.draw_btn.setChecked(checked)

    def _on_preview_mouse_press(self, event) -> None:
        """Handle mouse press in preview - start drawing region."""
        if not self._drawing_mode:
            return

        pos = event.pos()
        img_point = self._widget_to_image_coords(pos.x(), pos.y())
        if img_point:
            self._draw_start_point = img_point
            self._draw_end_point = img_point

    def _on_preview_mouse_move(self, event) -> None:
        """Handle mouse move in preview - update region endpoint."""
        if not self._drawing_mode or self._draw_start_point is None:
            return

        pos = event.pos()
        img_point = self._widget_to_image_coords(pos.x(), pos.y())
        if img_point:
            self._draw_end_point = img_point
            self._draw_region_preview()

    def _on_preview_mouse_release(self, event) -> None:
        self._manual_revision += 1
        """Handle mouse release - finalize region."""
        if not self._drawing_mode or self._draw_start_point is None:
            return

        pos = event.pos()
        img_point = self._widget_to_image_coords(pos.x(), pos.y())
        if img_point:
            self._draw_end_point = img_point

        # Validate and save the region
        if self._draw_start_point and self._draw_end_point:
            p1 = self._draw_start_point
            p2 = self._draw_end_point

            from menipy.common.auto_calibrator import CalibrationResult

            if self.result is None:
                self.result = CalibrationResult(confidence_scores={"overall": 0.5})
            if self._drawing_mode not in self.result.manual_regions:
                self.result.manual_regions.append(self._drawing_mode)

            if self._drawing_mode == "substrate":
                # Ensure left-to-right order for substrate line
                if p1[0] > p2[0]:
                    p1, p2 = p2, p1
                self.result.substrate_line = (p1, p2)
                self.result.substrate_warning = False
                self.result.confidence_scores["substrate"] = 1.0
                logger.info(f"Manual substrate line set: {p1} -> {p2}")

            elif self._drawing_mode == "needle":
                # Store needle as rect from line (x, y, width, height)
                x = min(p1[0], p2[0])
                y = min(p1[1], p2[1])
                w = abs(p2[0] - p1[0])
                h = abs(p2[1] - p1[1])
                # Needle should be vertical, so ensure minimum width
                if w < 10:
                    w = 40
                    x = x - 20
                self.result.needle_rect = (x, y, w, h)
                self.result.confidence_scores["needle"] = 1.0
                logger.info(f"Manual needle rect set: ({x}, {y}, {w}, {h})")

            elif self._drawing_mode == "roi":
                # Store ROI as rect (x, y, width, height)
                x = min(p1[0], p2[0])
                y = min(p1[1], p2[1])
                w = abs(p2[0] - p1[0])
                h = abs(p2[1] - p1[1])
                self.result.roi_rect = (x, y, w, h)
                self.result.confidence_scores["roi"] = 1.0
                logger.info(f"Manual ROI rect set: ({x}, {y}, {w}, {h})")

            # Refresh only display geometry; retain measured samples for analysis.
            from menipy.common.liquid_boundary import update_calibration_boundary

            update_calibration_boundary(self.result, self.pipeline_name)
            # Update UI
            self._show_results()
            self._apply_btn.setEnabled(True)

        # Exit drawing mode
        current_mode = self._drawing_mode
        self._drawing_mode = None
        self._preview_label.setCursor(Qt.ArrowCursor)

        # Uncheck draw button
        if current_mode:
            widget = self._region_widgets.get(current_mode)
            if widget and widget.draw_btn:
                widget.draw_btn.setChecked(False)

    def _widget_to_image_coords(self, wx: int, wy: int) -> tuple | None:
        """Convert widget coordinates to image coordinates."""
        if not hasattr(self, "_current_pixmap") or self._current_pixmap is None:
            return None

        # Get the displayed pixmap (may be scaled)
        displayed_pm = self._preview_label.pixmap()
        if displayed_pm is None:
            return None

        # Calculate offset (pixmap is centered in label)
        label_size = self._preview_label.size()
        pm_size = displayed_pm.size()

        offset_x = (label_size.width() - pm_size.width()) // 2
        offset_y = (label_size.height() - pm_size.height()) // 2

        # Adjust for offset
        px = wx - offset_x
        py = wy - offset_y

        # Check bounds
        if px < 0 or px >= pm_size.width() or py < 0 or py >= pm_size.height():
            return None

        # Scale to original image size
        orig_w = self.original_image.shape[1]
        orig_h = self.original_image.shape[0]

        img_x = int(px * orig_w / pm_size.width())
        img_y = int(py * orig_h / pm_size.height())

        return (img_x, img_y)

    def _draw_region_preview(self) -> None:
        """Draw region preview during dragging."""
        if self._draw_start_point is None or self._draw_end_point is None:
            return

        # Draw on a copy of the current overlay
        overlay = self.original_image.copy()

        # Draw existing detections (except the one being drawn)
        if self.result:
            overlay = self._draw_overlays()

        p1 = self._draw_start_point
        p2 = self._draw_end_point

        if self._drawing_mode == "substrate":
            # Draw line in magenta
            cv2.line(overlay, p1, p2, (255, 0, 255), 2)
            cv2.circle(overlay, p1, 4, (255, 0, 255), -1)
            cv2.circle(overlay, p2, 4, (255, 0, 255), -1)

        elif self._drawing_mode == "needle":
            # Draw line/rect in blue
            cv2.line(overlay, p1, p2, (255, 0, 0), 2)
            cv2.circle(overlay, p1, 4, (255, 0, 0), -1)
            cv2.circle(overlay, p2, 4, (255, 0, 0), -1)

        elif self._drawing_mode == "roi":
            # Draw rectangle in yellow
            cv2.rectangle(overlay, p1, p2, (0, 255, 255), 2)

        # Display
        self._display_image(overlay)

    # Legacy compatibility
    def _draw_substrate_preview(self) -> None:
        """Legacy handler - redirects to unified handler."""
        self._draw_region_preview()


# Standalone test
if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = 200  # Light gray background

    # Draw fake drop
    cv2.ellipse(test_image, (320, 350), (80, 60), 0, 0, 360, (50, 50, 50), -1)

    # Draw fake substrate
    cv2.line(test_image, (0, 400), (640, 400), (30, 30, 30), 3)

    # Draw fake needle
    cv2.rectangle(test_image, (300, 0), (340, 300), (40, 40, 40), -1)

    dlg = CalibrationWizardDialog(test_image, "sessile")
    dlg.show()

    sys.exit(app.exec())
