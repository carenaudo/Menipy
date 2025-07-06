"""Launch the Menipy GUI application."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def main() -> None:
    """Run the graphical interface."""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.showMaximized()
    app.exec()


if __name__ == "__main__":
    main()
