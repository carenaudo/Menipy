"""
Main GUI application setup and initialization.
"""

# src/menipy/gui/app.py
# type: ignore
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt, QResource
from PySide6.QtWidgets import QApplication, QMessageBox

# Suppress noisy Qt SVG resource missing warnings for :/icons/* when resources aren't registered
try:
    from PySide6.QtCore import qInstallMessageHandler
    import sys

    def _qt_msg_handler(msg_type, context, message):
        try:
            if (
                isinstance(message, str)
                and ":/icons/" in message
                and "Cannot open file" in message
            ):
                return
        except Exception:
            pass
        # fallback: print to stderr so other messages still appear
        try:
            sys.__stderr__.write(message + "\n")
        except Exception:
            pass

    qInstallMessageHandler(_qt_msg_handler)
except Exception:
    pass
# Register :/icons/... resources once at startup


# Try to register Qt resources (either compiled .py from pyside6-rcc or a bundled .rcc)
def _register_qrc():
    # compiled Python resource modules
    try:
        from .resources import app_rc  # registers on import
    except Exception:
        pass
    try:
        from .resources import icons_rc  # registers on import
    except Exception:
        pass
    # some projects generate named resource modules differently; try common alternatives
    try:
        from .resources import menipy_icons_rc  # registers on import
    except Exception:
        pass
    try:
        from .resources import menipy_icons  # registers on import
    except Exception:
        pass

    # optional fallbacks to .rcc files
    base = Path(__file__).resolve().parent / "resources"
    for fname in ("app.rcc", "icons.rcc"):
        rcc = base / fname
        if rcc.exists():
            from PySide6.QtCore import QResource

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

    # Ensure controllers can clean up background threads before Qt shuts down
    def _on_quit():
        try:
            if hasattr(w, "main_controller") and w.main_controller:
                try:
                    w.main_controller.shutdown()
                except Exception:
                    pass
        except Exception:
            pass

    try:
        app.aboutToQuit.connect(_on_quit)
    except Exception:
        pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
