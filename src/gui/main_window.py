"""Main Window module for Menipy GUI."""

from PySide6 import QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with image view and control panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Menipy")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create widgets and layouts."""
        splitter = QtWidgets.QSplitter()

        # Image display area
        self.graphics_view = QtWidgets.QGraphicsView()
        splitter.addWidget(self.graphics_view)

        # Control panel
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)
        self.process_button = QtWidgets.QPushButton("Process")
        control_layout.addWidget(self.process_button)
        control_layout.addStretch()
        splitter.addWidget(control_widget)

        self.setCentralWidget(splitter)


def main():
    """Launch the Menipy GUI application."""
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
