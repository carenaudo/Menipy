"""Tests for the Sub-pixel Edge Detection Plugin."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

# Ensure plugins directory is on path
plugins_dir = Path(__file__).parent.parent / "plugins"
if str(plugins_dir) not in sys.path:
    sys.path.insert(0, str(plugins_dir))

# Import plugins to register them
import_module("edge_detectors")
import_module("auto_subpixel_edge")

from auto_subpixel_edge import (  # noqa: E402
    refine_contour_subpixel,
    subpixel_edge_detect,
)

from menipy.common.registry import EDGE_DETECTORS  # noqa: E402
from menipy.models.config import EdgeDetectionSettings  # noqa: E402


def test_subpixel_plugin_is_registered():
    """Verify subpixel detector is registered in EDGE_DETECTORS registry."""
    assert "subpixel" in EDGE_DETECTORS
    detector = EDGE_DETECTORS["subpixel"]
    assert callable(detector)


def test_subpixel_refines_fractional_circle_radius():
    """Verify subpixel refinement recovers true subpixel radius accurately (<0.08 px)."""
    h, w = 200, 200
    center = (100.0, 100.0)
    true_radius = 45.35  # non-integer radius

    # Generate high-resolution anti-aliased circle
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    # Sigmoid smooth edge
    edge_width = 1.5
    img_float = 200.0 / (1.0 + np.exp((dist_from_center - true_radius) / (edge_width * 0.5)))
    img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)

    # Initial discrete contour snapped to integer radius (e.g. 45.0)
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    int_radius = 45.0
    initial_cnt = np.column_stack([
        center[0] + int_radius * np.cos(theta),
        center[1] + int_radius * np.sin(theta),
    ])

    refined_cnt = refine_contour_subpixel(
        img_uint8,
        initial_cnt,
        search_radius_px=3.0,
        sample_points=15,
        blur_ksize=1,
    )

    refined_radii = np.sqrt(
        (refined_cnt[:, 0] - center[0]) ** 2 + (refined_cnt[:, 1] - center[1]) ** 2
    )
    mean_refined_radius = float(np.mean(refined_radii))

    # The subpixel detector should shift from 45.0 to near 45.35
    assert abs(mean_refined_radius - true_radius) < 0.08
    # Improvement over integer contour
    assert abs(mean_refined_radius - true_radius) < abs(int_radius - true_radius)


def test_subpixel_edge_detect_callable_with_settings():
    """Verify subpixel_edge_detect works with EdgeDetectionSettings."""
    img = np.full((160, 160), 200, dtype=np.uint8)
    cv2.circle(img, (80, 80), 30, 50, -1)

    settings = EdgeDetectionSettings(
        method="subpixel",
        plugin_settings={
            "base_method": "otsu",
            "search_radius_px": 2.5,
            "sample_points": 9,
        },
    )

    result = subpixel_edge_detect(img, settings)
    assert result is not None
    assert len(result) > 10
    assert result.shape[1] == 2
    # Verify coordinates are float
    assert issubclass(result.dtype.type, np.floating)


def test_subpixel_handles_small_or_empty_contour():
    """Verify small contours are safely handled without errors."""
    img = np.zeros((50, 50), dtype=np.uint8)
    tiny_cnt = np.array([[10.0, 10.0], [12.0, 12.0]])
    refined = refine_contour_subpixel(img, tiny_cnt)
    assert np.array_equal(refined, tiny_cnt)
