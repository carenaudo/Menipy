"""Run menipy as a module: `python -m menipy` -> GUI entrypoint.
"""
from .gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
