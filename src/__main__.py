"""Entry point for running the Menipy GUI."""

# When executed as ``python -m src`` the package root is ``src``. Use a
# relative import so the contained ``menipy`` package is discoverable
# without installing the project.
from .menipy.gui import main


if __name__ == "__main__":
    main()
