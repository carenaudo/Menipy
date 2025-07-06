"""Ensure :mod:`optical_goniometry` can be imported.

This test simply attempts to import each top-level module to verify that
the scaffolded package layout is discoverable by Python.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Add the src_alt directory itself to ``sys.path`` so that the package can be
# imported without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = [
    "optical_goniometry",
    "optical_goniometry.io.loaders",
    "optical_goniometry.preprocessing.preprocess",
    "optical_goniometry.calibration.calibrator",
    "optical_goniometry.detection.needle",
    "optical_goniometry.detection.droplet",
    "optical_goniometry.detection.substrate",
    "optical_goniometry.analysis.pendant",
    "optical_goniometry.analysis.sessile",
    "optical_goniometry.analysis.commons",
    "optical_goniometry.metrics.metrics",
    "optical_goniometry.ui.main_window",
    "optical_goniometry.cli",
    "optical_goniometry.utils",
    "optical_goniometry.plugins",
]


def test_imports() -> None:
    """Packages should import without error."""
    for name in MODULES:
        assert importlib.import_module(name)
