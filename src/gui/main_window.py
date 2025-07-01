"""Main window module for Menipy GUI."""

from pathlib import Path

import numpy as np

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
    QGraphicsView,
    QGraphicsScene,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
)

from .controls import ZoomControl, ParameterPanel, MetricsPanel

from ..processing.reader import load_image
from ..processing import segmentation
from ..processing.segmentation import (
    morphological_cleanup,
    external_contour_mask,
    find_contours,
    ml_segment,
)
from ..utils import (
    get_calibration,
    pixels_to_mm,
    auto_calibrate,
    calibrate_from_points,
)
from ..models.properties import droplet_volume


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
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        splitter.addWidget(self.graphics_view)

        # Control panel
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Otsu", "Adaptive"])
        control_layout.addWidget(self.algorithm_combo)

        self.zoom_control = ZoomControl()
        self.zoom_control.zoomChanged.connect(self.set_zoom)
        control_layout.addWidget(self.zoom_control)

        self.parameter_panel = ParameterPanel()
        control_layout.addWidget(self.parameter_panel)

        self.metrics_panel = MetricsPanel()
        control_layout.addWidget(self.metrics_panel)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_image)
        control_layout.addWidget(self.process_button)

        control_layout.addStretch()
        splitter.addWidget(control_widget)

        self.setCentralWidget(splitter)

        # Menu actions
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)

        save_action = QAction("Save Annotated Image", self)
        save_action.triggered.connect(self.save_annotated_image)
        file_menu.addAction(save_action)

        calib_action = QAction("Calibration", self)
        calib_action.triggered.connect(self.open_calibration)
        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction(calib_action)

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
        self.pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_view.resetTransform()

        rect = self.pixmap_item.boundingRect()
        self.graphics_view.setSceneRect(rect)
        self.graphics_view.setFixedSize(int(rect.width()), int(rect.height()))
        self.adjustSize()
        self.set_zoom(self.zoom_control.slider.value() / 100.0)


    def process_image(self) -> None:
        """Run segmentation on the loaded image and overlay the mask."""
        if getattr(self, "image", None) is None:
            return
        image = self.image
        offset = (0, 0)
        if self.roi_rect is not None:
            x1, y1, x2, y2 = map(int, self.roi_rect)
            image = image[y1:y2, x1:x2]
            offset = (x1, y1)

        # Clear previous markers
        if self.apex_item is not None:
            self.graphics_scene.removeItem(self.apex_item)
            self.apex_item = None
        if self.contact_line_item is not None:
            self.graphics_scene.removeItem(self.contact_line_item)
            self.contact_line_item = None

        if getattr(self, "use_ml_action", None) and self.use_ml_action.isChecked():
            mask = ml_segment(image)
        else:
            algo = self.algorithm_combo.currentText()
            if algo == "Otsu":
                mask = segmentation.otsu_threshold(image)
            else:
                mask = segmentation.adaptive_threshold(image)

        mask = morphological_cleanup(mask, kernel_size=3, iterations=1)
        mask = external_contour_mask(mask)
        mask_img = QImage(
            mask.data,
            mask.shape[1],
            mask.shape[0],
            mask.strides[0],
            QImage.Format_Grayscale8,
        )
        mask_pix = QPixmap.fromImage(mask_img)
        if self.mask_item is not None:
            self.graphics_scene.removeItem(self.mask_item)
        self.mask_item = self.graphics_scene.addPixmap(mask_pix)
        self.mask_item.setOffset(*offset)
        self.mask_item.setOpacity(0.4)

        for item in self.contour_items:
            self.graphics_scene.removeItem(item)
        self.contour_items.clear()

        for contour in find_contours(mask):
            path = QPainterPath()
            if contour.size == 0:
                continue
            if offset != (0, 0):
                contour = contour + np.array(offset)
            path.moveTo(*contour[0])
            for point in contour[1:]:
                path.lineTo(*point)
            pen = QPen(QColor("red"))
            pen.setWidth(2)
            item = self.graphics_scene.addPath(path, pen)
            self.contour_items.append(item)

        # Apex marker from mask
        if self.apex_item is not None:
            self.graphics_scene.removeItem(self.apex_item)
            self.apex_item = None
        if self.contact_line_item is not None:
            self.graphics_scene.removeItem(self.contact_line_item)
            self.contact_line_item = None

        if mask.any():
            ys, xs = np.nonzero(mask)
            idx = int(ys.argmin())
            apex_x = xs[idx] + offset[0]
            apex_y = ys[idx] + offset[1]
            pen = QPen(QColor("yellow"))
            brush = QColor("yellow")
            self.apex_item = self.graphics_scene.addEllipse(
                apex_x - 3,
                apex_y - 3,
                6,
                6,
                pen,
                brush,
            )

            if self.calibration_rect is not None:
                contact_y = int(self.calibration_rect[1])
            else:
                contact_y = ys.max() + offset[1]
            row_y = contact_y - offset[1]
            row_y = min(max(int(row_y), 0), mask.shape[0] - 1)
            row = mask[row_y]
            cols = np.where(row > 0)[0]
            if cols.size >= 2:
                x1 = cols.min() + offset[0]
                x2 = cols.max() + offset[0]
                pen = QPen(QColor("cyan"))
                pen.setWidth(2)
                self.contact_line_item = self.graphics_scene.addLine(
                    x1,
                    contact_y,
                    x2,
                    contact_y,
                    pen,
                )

        # Basic metrics from the mask
        ys, xs = np.nonzero(mask)
        if ys.size > 0 and xs.size > 0:
            height_px = ys.max() - ys.min()
            diameter_px = xs.max() - xs.min()
            height = pixels_to_mm(float(height_px))
            diameter = pixels_to_mm(float(diameter_px))
            volume = droplet_volume(diameter / 2.0, np.deg2rad(90.0))
        else:
            height = diameter = volume = 0.0

        self.metrics_panel.set_metrics(
            ift=0.0,
            wo=0.0,
            volume=volume,
            contact_angle=0.0,
            height=height,
            diameter=diameter,
        )

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
        self.graphics_view.resetTransform()
        self.graphics_view.scale(factor, factor)


def main():
    """Launch the Menipy GUI application."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
