"""
Detection stage helpers for pipeline integration.

This module provides pipeline-agnostic helper functions that use the
detection plugins to automatically detect ROI, needle, drop, and other
features from images.
"""

from __future__ import annotations

import logging
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import numpy as np

from menipy.common.detection_result import normalize_detection_result
from menipy.common.image_utils import edges_to_xy, ensure_gray

logger = logging.getLogger(__name__)

# Import detection plugins (registers them with the registry)
try:
    import sys
    from pathlib import Path

    plugins_dir = Path(__file__).parent.parent.parent.parent / "plugins"
    if plugins_dir.exists() and str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))

    for plugin_name in (
        "detect_apex",
        "detect_drop",
        "detect_needle",
        "detect_roi",
        "detect_substrate",
    ):
        import_module(plugin_name)

    _PLUGINS_LOADED = True
except ImportError as e:
    logger.warning(f"Detection plugins not available: {e}")
    _PLUGINS_LOADED = False

from menipy.common.registry import (
    APEX_DETECTORS,
    DROP_DETECTORS,
    NEEDLE_DETECTORS,
    ROI_DETECTORS,
    SUBSTRATE_DETECTORS,
)


def plugins_available() -> bool:
    """Check if detection plugins are loaded."""
    return _PLUGINS_LOADED and len(NEEDLE_DETECTORS) > 0


def auto_detect_features(
    image: np.ndarray,
    pipeline: str = "sessile",
    *,
    detect_needle: bool = True,
    detect_substrate: bool = True,
    detect_drop: bool = True,
    detect_apex: bool = True,
    detect_roi: bool = True,
    experimental_geometry_mode: str = "off",
    needle_geometry_method: str = "legacy",
    onnx_proposal_mode: str = "off",
    segmentation_provider: str = "mobilesam",
) -> dict[str, Any]:
    """
    Run automatic feature detection for a pipeline.

    Args:
        image: Input image (BGR or grayscale)
        pipeline: Pipeline name ("sessile" or "pendant")
        detect_*: Flags to enable/disable specific detections

    Returns:
        Dictionary with detected features:
            - needle_rect: (x, y, w, h) or None
        - substrate_line: ((x1, y1), (x2, y2)) or None
        - drop_contour: Nx2 array or None
        - contact_points: ((left), (right)) or None
        - apex_point: (x, y) or None
        - roi_rect: (x, y, w, h) or None
    """
    if not plugins_available():
        logger.warning("Detection plugins not loaded")
        return {}

    results: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    pipeline = pipeline.lower()

    # 1. Detect substrate (sessile only)
    substrate_y = None
    if detect_substrate and pipeline == "sessile":
        if "gradient" in SUBSTRATE_DETECTORS:
            substrate_line = SUBSTRATE_DETECTORS["gradient"](image)
            outcome = normalize_detection_result(
                substrate_line, feature="substrate", parameters={"detector": "gradient"}
            )
            diagnostics["substrate"] = outcome.to_diagnostics()
            if outcome.accepted:
                substrate_line = outcome.value
                results["substrate_line"] = substrate_line
                # Extract Y for other detections
                substrate_y = int((substrate_line[0][1] + substrate_line[1][1]) / 2)

    # 2. Detect needle
    if detect_needle:
        detector_name = pipeline if pipeline in NEEDLE_DETECTORS else "sessile"
        experimental_name = f"{pipeline}_bilateral"
        run_experimental = needle_geometry_method == "bilateral_robust" or experimental_geometry_mode == "shadow"
        if detector_name in NEEDLE_DETECTORS:
            needle_result = NEEDLE_DETECTORS[detector_name](image)
            outcome = normalize_detection_result(
                needle_result,
                feature="needle",
                parameters={"detector": detector_name},
            )
            diagnostics["needle"] = outcome.to_diagnostics()
            if outcome.accepted:
                needle_result = outcome.value
                if (
                    pipeline == "pendant"
                    and isinstance(needle_result, tuple)
                    and len(needle_result) == 2
                ):
                    # Pendant returns (needle_rect, contact_points)
                    results["needle_rect"] = needle_result[0]
                    results["contact_points"] = needle_result[1]
                else:
                    results["needle_rect"] = needle_result
        if run_experimental and experimental_name in NEEDLE_DETECTORS:
            experimental_raw = NEEDLE_DETECTORS[experimental_name](image)
            experimental = normalize_detection_result(experimental_raw, feature="needle", parameters={"detector": experimental_name})
            diagnostics["needle_bilateral"] = experimental.to_diagnostics()
            if needle_geometry_method == "bilateral_robust":
                if experimental.accepted and isinstance(experimental.value, dict):
                    results["needle_rect"] = experimental.value.get("needle_rect")
                    if pipeline == "pendant":
                        results["contact_points"] = experimental.value.get("contact_points")
                else:
                    results.pop("needle_rect", None)
                    results.pop("contact_points", None)

    # 3. Detect drop contour
    drop_contour = None
    if detect_drop:
        detector_name = pipeline if pipeline in DROP_DETECTORS else "sessile"
        if detector_name in DROP_DETECTORS:
            if pipeline == "sessile":
                drop_result = DROP_DETECTORS[detector_name](
                    image, substrate_y=substrate_y
                )
                outcome = normalize_detection_result(
                    drop_result,
                    feature="drop",
                    parameters={"detector": detector_name},
                )
                diagnostics["drop"] = outcome.to_diagnostics()
                if outcome.accepted:
                    drop_result = outcome.value
                    if isinstance(drop_result, tuple) and len(drop_result) == 2:
                        drop_contour, contact_pts = drop_result
                        results["drop_contour"] = drop_contour
                        if contact_pts:
                            results["contact_points"] = contact_pts
                    else:
                        drop_contour = drop_result
                        results["drop_contour"] = drop_contour
            else:
                raw_drop = DROP_DETECTORS[detector_name](image)
                outcome = normalize_detection_result(
                    raw_drop,
                    feature="drop",
                    parameters={"detector": detector_name},
                )
                diagnostics["drop"] = outcome.to_diagnostics()
                drop_contour = outcome.value
                if outcome.accepted:
                    results["drop_contour"] = drop_contour

    # 4. Detect apex
    apex_point = None
    if detect_apex and drop_contour is not None:
        detector_name = pipeline if pipeline in APEX_DETECTORS else "auto"
        if detector_name in APEX_DETECTORS:
            if pipeline == "sessile":
                apex_point = APEX_DETECTORS[detector_name](
                    drop_contour, substrate_y=substrate_y
                )
            else:
                apex_point = APEX_DETECTORS[detector_name](drop_contour)
            if apex_point:
                results["apex_point"] = apex_point
            outcome = normalize_detection_result(apex_point, feature="apex")
            diagnostics["apex"] = outcome.to_diagnostics()

    # 5. Detect ROI
    if detect_roi:
        detector_name = pipeline if pipeline in ROI_DETECTORS else "auto"
        if detector_name in ROI_DETECTORS:
            # Build pipeline-specific kwargs
            if pipeline == "sessile":
                roi_kwargs = {
                    "drop_contour": results.get("drop_contour"),
                    "substrate_y": substrate_y,
                    "needle_rect": results.get("needle_rect"),
                }
            else:  # pendant
                roi_kwargs = {
                    "drop_contour": results.get("drop_contour"),
                    "apex_point": results.get("apex_point"),
                }

            roi_rect = ROI_DETECTORS[detector_name](image, **roi_kwargs)
            if roi_rect:
                results["roi_rect"] = roi_rect
            outcome = normalize_detection_result(roi_rect, feature="roi")
            diagnostics["roi"] = outcome.to_diagnostics()

    results["detector_diagnostics"] = diagnostics
    if onnx_proposal_mode == "shadow":
        from menipy.common.onnx_shadow import run_shadow_segmentation

        proxy = SimpleNamespace(
            image=image,
            frames=None,
            drop_contour=results.get("drop_contour"),
            detected_contour=results.get("drop_contour"),
            contour=None,
            needle_rect=results.get("needle_rect"),
            onnx_proposal_mode="shadow",
            segmentation_provider=segmentation_provider,
            onnx_proposal_classes=["droplet", "needle"],
            onnx_proposals={},
        )
        run_shadow_segmentation(proxy, pipeline)
        diagnostics["onnx_proposals"] = proxy.onnx_proposals
    logger.info(f"Auto-detected features for {pipeline}: {list(results.keys())}")
    return results


def apply_roi_crop(
    image: np.ndarray,
    roi_rect: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Crop image to ROI region.

    Args:
        image: Input image
        roi_rect: (x, y, width, height)

    Returns:
        Cropped image.
    """
    x, y, w, h = roi_rect
    return image[y : y + h, x : x + w].copy()


def adjust_coordinates_for_roi(
    points: np.ndarray,
    roi_rect: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Adjust point coordinates for ROI offset.

    Args:
        points: Nx2 array of (x, y) coordinates
        roi_rect: (x, y, width, height)

    Returns:
        Adjusted points with ROI offset subtracted.
    """
    x, y, w, h = roi_rect
    adjusted = points.copy()
    adjusted[:, 1] -= y
    return adjusted


# -----------------------------------------------------------------------------
# Shared Edge Detection Helpers
# -----------------------------------------------------------------------------
# Re-exporting from image_utils for backward compatibility if needed
# (though ideally consumers should import from image_utils directly)
__all__ = [
    "ensure_gray",
    "edges_to_xy",
    "auto_detect_features",
    "apply_roi_crop",
    "adjust_coordinates_for_roi",
    "plugins_available",
]
