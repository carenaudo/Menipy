"""GUI entry point for running Menipy GUI directly."""

# python -m menipy.gui  --> GUI entry
from .app import main as gui_main  # adjust if your GUI entry is elsewhere

if __name__ == "__main__":
    gui_main()
