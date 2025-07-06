"""Top-level package for the refactored goniometry library."""

from .sharpen_plugin import sharpen_filter

try:
    from .gui import main as gui_main
except Exception:  # pragma: no cover - optional dependency
    gui_main = None

__all__ = ["sharpen_filter", "gui_main"]
