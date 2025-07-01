"""Main window module for Menipy GUI."""

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..processing.reader import load_image
from ..processing import segmentation


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with image view and control panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Menipy")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create widgets, menus, and layouts."""
        splitter = QtWidgets.QSplitter()

        # Image display area
        self.graphics_view = QtWidgets.QGraphicsView()
        self.graphics_scene = QtWidgets.QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)
        splitter.addWidget(self.graphics_view)

        # Control panel
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)

        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_combo.addItems(["Otsu", "Adaptive"])
        control_layout.addWidget(self.algorithm_combo)

        self.process_button = QtWidgets.QPushButton("Process")
        self.process_button.clicked.connect(self.process_image)
        control_layout.addWidget(self.process_button)

        control_layout.addStretch()
        splitter.addWidget(control_widget)

        self.setCentralWidget(splitter)

        # Menu actions
        open_action = QtWidgets.QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)

        calib_action = QtWidgets.QAction("Calibration", self)
        calib_action.triggered.connect(self.open_calibration)
        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction(calib_action)

        self.mask_item = None

    def open_image(self) -> None:
        """Open an image file and display it."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()), "Images (*.png *.jpg *.bmp)"
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        self.image = load_image(path)
        if self.image.ndim == 3:
            rgb = QtGui.QImage(
                self.image.data,
                self.image.shape[1],
                self.image.shape[0],
                self.image.strides[0],
                QtGui.QImage.Format_BGR888,
            )
        else:
            rgb = QtGui.QImage(
                self.image.data,
                self.image.shape[1],
                self.image.shape[0],
                self.image.strides[0],
                QtGui.QImage.Format_Grayscale8,
            )
        pixmap = QtGui.QPixmap.fromImage(rgb)
        self.graphics_scene.clear()
        self.pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_view.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)

    def process_image(self) -> None:
        """Run segmentation on the loaded image and overlay the mask."""
        if getattr(self, "image", None) is None:
            return
        algo = self.algorithm_combo.currentText()
        if algo == "Otsu":
            mask = segmentation.otsu_threshold(self.image)
        else:
            mask = segmentation.adaptive_threshold(self.image)
        mask_img = QtGui.QImage(
            mask.data,
            mask.shape[1],
            mask.shape[0],
            mask.strides[0],
            QtGui.QImage.Format_Grayscale8,
        )
        mask_pix = QtGui.QPixmap.fromImage(mask_img)
        if self.mask_item is not None:
            self.graphics_scene.removeItem(self.mask_item)
        self.mask_item = self.graphics_scene.addPixmap(mask_pix)
        self.mask_item.setOpacity(0.4)

    def open_calibration(self) -> None:
        """Display a placeholder calibration dialog."""
        QtWidgets.QMessageBox.information(self, "Calibration", "Not implemented")


def main():
    """Launch the Menipy GUI application."""
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
