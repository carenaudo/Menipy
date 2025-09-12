"""Top-level menipy package.

Expose a package version and re-export commonly-used subpackages so
`import menipy; menipy.gui` and `menipy.__version__` work.
"""

__version__ = "0.1.0"

# Re-export commonly used subpackages to make them available as attributes
from . import gui  # noqa: F401

__all__ = ["gui", "pipelines"]