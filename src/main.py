"""Entry point for the Menipy application."""

from PySide6 import QtWidgets

from .gui.main_window import MainWindow


def main() -> None:
    """Launch the Menipy GUI application."""
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
