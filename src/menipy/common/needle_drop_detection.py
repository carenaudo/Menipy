"""Needle-in-sessile-drop segmentation and contact line detection.

Reuses detection primitives from :mod:`menipy.common.sessile_detection` while
tailoring contour extraction for drops with an immersed dispensing needle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from menipy.common.geometry import find_contact_points_from_contour
from menipy.common.sessile_detection import (
    detect_sessile_needle_shaft,
    detect_sessile_substrate_robust,
    ensure_gray_image,
    segment_sessile_binary,
)

logger = logging.getLogger(__name__)


@dataclass
class NeedleDropDetection:
    """Detection results for a single needle-in-sessile-drop frame."""

    drop_contour: np.ndarray | None
    """Outer droplet contour (excluding the vertical needle shaft)."""
    contact_points: tuple[tuple[float, float], tuple[float, float]] | None
    """Left and right contact coordinates along the baseline (p_left, p_right)."""
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None
    """Detected or calibrated substrate baseline endpoints."""
    needle_rect: tuple[int, int, int, int] | None
    """Needle shaft bounding rectangle (x, y, w, h)."""
    needle_expansion_y: int | None
    """Y-coordinate where the needle shaft expands into the droplet surface."""
    base_diameter_px: float | None
    """Euclidean distance between contact points in pixels."""
    confidence: float
    """Confidence score [0.0, 1.0]."""
    diagnostics: dict[str, Any]


def detect_needle_drop_features(
    image: np.ndarray,
    *,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    needle_rect: tuple[int, int, int, int] | None = None,
    shaft_margin_px: int = 4,
    contact_tolerance_px: float = 20.0,
) -> NeedleDropDetection:
    """Detect drop contour, contact line, and needle shaft for a needle-in-drop frame.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    substrate_line : ((x1, y1), (x2, y2)) | None
        Optional prior baseline coordinates.  If None, detected automatically.
    needle_rect : (x, y, w, h) | None
        Optional prior needle shaft rectangle.
    shaft_margin_px : int
        Horizontal padding around needle shaft to exclude.
    contact_tolerance_px : float
        Vertical tolerance when searching for contact points along baseline.

    Returns
    -------
    NeedleDropDetection
        Detected features and geometry.
    """
    gray = ensure_gray_image(image)
    height, width = gray.shape[:2]

    # 1. Baseline detection if not provided
    if substrate_line is None:
        sub_line, sub_conf, sub_diag, _ = detect_sessile_substrate_robust(image)
        substrate_line = sub_line
    else:
        sub_conf = 1.0
        sub_diag = {"source": "calibrated_or_prior"}

    sub_y = (
        (substrate_line[0][1] + substrate_line[1][1]) / 2.0
        if substrate_line is not None
        else int(height * 0.8)
    )

    # 2. Needle shaft detection
    expansion_y = None
    if needle_rect is None:
        n_rect, n_conf, exp_y = detect_sessile_needle_shaft(image, substrate_y=int(sub_y))
        needle_rect = n_rect
        expansion_y = exp_y
    else:
        expansion_y = needle_rect[1] + needle_rect[3]

    # 3. Adaptive thresholding and segmentation
    binary = segment_sessile_binary(image, substrate_y=int(sub_y) + 3)

    # 4. Needle shaft exclusion mask
    # Blank out the needle column from top boundary down to the expansion boundary
    if needle_rect is not None:
        nx, ny, nw, nh = needle_rect
        mask_x0 = max(0, nx - shaft_margin_px)
        mask_x1 = min(width, nx + nw + shaft_margin_px)
        mask_y1 = min(height, (expansion_y if expansion_y is not None else ny + nh) + 2)
        binary[0:mask_y1, mask_x0:mask_x1] = 0

    # Clean morphological noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Extract droplet candidate contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return NeedleDropDetection(
            drop_contour=None,
            contact_points=None,
            substrate_line=substrate_line,
            needle_rect=needle_rect,
            needle_expansion_y=expansion_y,
            base_diameter_px=None,
            confidence=0.0,
            diagnostics={"error": "no_contours_found", "substrate_diag": sub_diag},
        )

    # Filter for the main droplet contour attached to the substrate
    center_x = width // 2
    candidates: list[tuple[np.ndarray, float, float]] = []

    for cnt in contours:
        cnt_2d = cnt.reshape(-1, 2)
        area = float(cv2.contourArea(cnt))
        if area < 50:
            continue
        max_y = float(np.max(cnt_2d[:, 1]))
        mean_x = float(np.mean(cnt_2d[:, 0]))

        # Must reach close to substrate line
        dist_to_sub = abs(max_y - sub_y)
        dist_to_center = abs(mean_x - center_x)

        if dist_to_sub <= contact_tolerance_px + 10.0:
            candidates.append((cnt_2d, area, dist_to_center))

    if not candidates:
        # Fallback to largest contour
        largest = max(contours, key=cv2.contourArea).reshape(-1, 2)
        candidates.append((largest, float(cv2.contourArea(largest)), 0.0))

    # Sort by largest area and proximity to center
    candidates.sort(key=lambda c: (-c[1], c[2]))
    drop_contour = candidates[0][0]

    # 6. Contact points detection
    contact_points = None
    diameter_px = None
    if substrate_line is not None:
        p1, p2 = find_contact_points_from_contour(
            drop_contour, substrate_line, tolerance=contact_tolerance_px
        )
        if p1 is not None and p2 is not None:
            contact_points = (
                (float(p1[0]), float(p1[1])),
                (float(p2[0]), float(p2[1])),
            )
            # Ensure p_left is left of p_right
            if contact_points[0][0] > contact_points[1][0]:
                contact_points = (contact_points[1], contact_points[0])

            diameter_px = float(
                np.hypot(
                    contact_points[1][0] - contact_points[0][0],
                    contact_points[1][1] - contact_points[0][1],
                )
            )

    conf = float(np.clip(sub_conf * (0.9 if contact_points else 0.4), 0.0, 1.0))

    return NeedleDropDetection(
        drop_contour=drop_contour,
        contact_points=contact_points,
        substrate_line=substrate_line,
        needle_rect=needle_rect,
        needle_expansion_y=expansion_y,
        base_diameter_px=diameter_px,
        confidence=conf,
        diagnostics={"substrate_diag": sub_diag, "candidates_count": len(candidates)},
    )
