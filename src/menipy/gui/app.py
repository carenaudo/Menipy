# src/menipy/gui/app.py
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt, QResource
from PySide6.QtWidgets import QApplication, QMessageBox

# Try to register Qt resources (either compiled .py from pyside6-rcc or a bundled .rcc)
def _register_qrc():
    try:
        # If you ran: pyside6-rcc src/menipy/gui/resources/app.qrc -o src/menipy/gui/resources/app_rc.py
        from .resources import app_rc  # noqa: F401  # registers on import
        return
    except Exception:
        pass
    # Fallback: look for an .rcc side-by-side with this file
    rcc = Path(__file__).resolve().parent / "resources" / "app.rcc"
    if rcc.exists():
        QResource.registerResource(str(rcc))

def _install_exception_hook(app: QApplication):
    def _hook(exctype, value, tb):
        # Print to stderr
        import traceback
        traceback.print_exception(exctype, value, tb)
        # Show a critical dialog (avoid recursion if app is shutting down)
        try:
            QMessageBox.critical(
                None,
                "Unexpected Error",
                f"{exctype.__name__}: {value}",
                QMessageBox.Ok,
            )
        finally:
            # Let Qt keep running (or change to sys.exit(1) if you prefer)
            pass
    sys.excepthook = _hook

def _configure_qt(app: QApplication):
    # Optional: Fusion style for consistency across platforms
    app.setStyle("Fusion")
    # High-DPI friendly defaults
    QCoreApplication.setOrganizationName("Menipy")
    QCoreApplication.setApplicationName("Menipy GUI")
    QCoreApplication.setApplicationVersion("0.1")

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv

    # Enable high-DPI before QApplication is created
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    _register_qrc()

    app = QApplication(argv)
    _install_exception_hook(app)
    _configure_qt(app)

    # Import here so resources (:/views/...) are registered first
    from .mainwindow import MainWindow

    w = MainWindow()
    w.resize(1200, 800)
    w.show()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
