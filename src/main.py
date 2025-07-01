"""Entry point for the Menipy application."""

from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def main() -> None:
    """Launch the Menipy GUI application."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
