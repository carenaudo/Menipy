"""Integration coverage for discovered sub-pixel edge detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from menipy.common.edge_detection import get_contour_detector
from menipy.common.plugin_db import PluginDB
from menipy.common.plugin_settings import get_detector_settings_model
from menipy.common.plugins import discover_into_db, load_active_plugins
from menipy.common.registry import EDGE_DETECTORS
from menipy.models.config import EdgeDetectionSettings


def test_subpixel_plugin_is_discovered_and_configurable_without_manual_import(tmp_path):
    """The normal discovery path exposes both detector and settings UI data."""
    plugins_dir = Path(__file__).parents[1] / "plugins"
    db = PluginDB(tmp_path / "subpixel-discovery.sqlite")

    discover_into_db(db, [plugins_dir])
    db.set_active("auto_subpixel_edge", "edge", True)
    load_active_plugins(db)

    assert "subpixel" in EDGE_DETECTORS
    assert get_detector_settings_model("subpixel") is not None
    assert get_contour_detector("subpixel") is EDGE_DETECTORS["subpixel"]

    image = np.full((120, 120), 220, dtype=np.uint8)
    cv2.circle(image, (60, 60), 25, 40, -1)
    settings = EdgeDetectionSettings(
        method="subpixel",
        plugin_settings={"subpixel": {"base_method": "otsu", "sample_points": 9}},
        min_contour_length=10,
    )
    contour = EDGE_DETECTORS["subpixel"](image, settings)

    assert contour.shape[1] == 2
    assert contour.dtype.kind == "f"
