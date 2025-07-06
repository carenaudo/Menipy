"""Ensure :mod:`menipy` can be imported.

This test simply attempts to import each top-level module to verify that
the scaffolded package layout is discoverable by Python.
"""

from __future__ import annotations

import importlib
import pytest
import sys
from pathlib import Path

# Add the src_alt directory itself to ``sys.path`` so that the package can be
# imported without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = [
    "menipy",
    "menipy.io.loaders",
    "menipy.preprocessing.preprocess",
    "menipy.calibration.calibrator",
    "menipy.detection.needle",
    "menipy.detection.droplet",
    "menipy.detection.substrate",
    "menipy.analysis.pendant",
    "menipy.analysis.sessile",
    "menipy.analysis.commons",
    "menipy.metrics.metrics",
    "menipy.ui.main_window",
    "menipy.cli",
    "menipy.gui",
    "menipy.utils",
    "menipy.plugins",
]


def test_imports() -> None:
    """Packages should import without error."""
    for name in MODULES:
        try:
            assert importlib.import_module(name)
        except ImportError as exc:
            if name == "menipy.ui.main_window":
                pytest.skip(f"PySide6 not available: {exc}")
            else:
                raise
