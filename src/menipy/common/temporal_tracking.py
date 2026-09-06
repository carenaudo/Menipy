"""Temporal droplet tracking engine for dynamic video sequences.

This module provides high-speed temporal tracking for sessile and pendant droplet
video sequences. It exploits physical invariants (fixed substrate line in sessile,
fixed dispensing needle in pendant), predicts droplet bounding regions, and warm-starts
Menipy's mathematical active contour solver from prior frame states.

Key Literature and Theoretical Foundations:
    1. Active Contours in Temporal Tracking:
       Kass, M., Witkin, A., & Terzopoulos, D. (1988).
       "Snakes: Active contour models."
       International Journal of Computer Vision, 1(4), 321–331.
       DOI: 10.1007/BF00133570

       Terzopoulos, D., & Szeliski, R. (1992).
       "Tracking with dynamic deformable models."
       In Active Vision, MIT Press, Cambridge, MA, pp. 3–20.

    2. Optical Flow Feature Tracking:
       Lucas, B. D., & Kanade, T. (1981).
       "An iterative image registration technique with an application to stereo vision."
       Proceedings of Imaging Understanding Workshop, pp. 121–130.

       Shi, J., & Tomasi, C. (1994).
       "Good features to track."
       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 593–600.

    3. Dynamic Contact Angle & Video Tensiometry:
       Dorywalski, K. (2026).
       "Drop-O-Matic: Open-source tool for dynamic contact angle determination."
       Zenodo. DOI: 10.5281/zenodo.19470985

       Berry, J. D., Neeson, M. J., Dagastine, R. R., Chan, D. Y., & Tabor, R. F. (2015).
       "Measurement of surface and interfacial tension using pendant drop tensiometry."
       Journal of Colloid and Interface Science, 454, 226–237.
       DOI: 10.1016/j.jcis.2015.05.012

License & Clean-Room Compliance:
    Authored for Menipy using standard OpenCV (Apache-2.0) and SciPy/NumPy (BSD-3-Clause)
    primitives under Menipy's MIT open-source license.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from menipy.common.geometry import find_contact_points_from_contour
from menipy.common.image_utils import ensure_gray
from menipy.math.active_contour import (
    ActiveContourConfig,
    SnakeBoundaryCondition,
    evolve_active_contour,
    resample_contour_arclength,
)
from menipy.pipelines.sessile.geometry import clip_contour_to_substrate

logger = logging.getLogger(__name__)


def predict_droplet_roi(
    contour: np.ndarray,
    contacts: tuple[tuple[float, float], tuple[float, float]] | None,
    velocity: np.ndarray,
    shape: tuple[int, ...],
    margin_ratio: float = 0.15,
    min_margin: int = 15,
) -> tuple[int, int, int, int]:
    """Calculate predicted bounding ROI from previous contour and contact velocity.

    Args:
        contour: (N, 2) array of contour coordinates.
        contacts: Optional ((x_left, y_left), (x_right, y_right)) contact points.
        velocity: (2,) displacement vector [vx, vy].
        shape: Image shape (height, width, ...).

    Returns:
        (x0, y0, w, h) bounding box.
    """
    predicted = contour + velocity.reshape(1, 2)
    x_min, y_min = np.floor(np.min(predicted, axis=0)).astype(int)
    x_max, y_max = np.ceil(np.max(predicted, axis=0)).astype(int)

    if contacts is not None:
        base_width = max(1.0, abs(contacts[1][0] - contacts[0][0]))
    else:
        base_width = max(1.0, float(x_max - x_min))

    margin = max(min_margin, int(round(base_width * margin_ratio)))
    height, width = shape[:2]

    x0 = max(0, x_min - margin)
    y0 = max(0, y_min - margin)
    x1 = min(width, x_max + margin)
    y1 = min(height, y_max + margin)

    return int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))


def estimate_contact_velocity_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    contacts: tuple[tuple[float, float], tuple[float, float]],
    search_window: tuple[int, int] = (15, 15),
) -> tuple[np.ndarray, bool]:
    """Estimate contact line displacement using Lucas-Kanade pyramidal optical flow.

    Args:
        prev_gray: Previous frame grayscale image.
        curr_gray: Current frame grayscale image.
        contacts: ((x_left, y_left), (x_right, y_right)) contact coordinates.
        search_window: Window size for optical flow search.

    Returns:
        (displacement_vector, success_flag) where displacement_vector is (2,) float array.
    """
    prev_pts = np.array(
        [[contacts[0][0], contacts[0][1]], [contacts[1][0], contacts[1][1]]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=search_window,
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    if status is not None and status[0, 0] and status[1, 0]:
        disp_left = new_pts[0, 0] - prev_pts[0, 0]
        disp_right = new_pts[1, 0] - prev_pts[1, 0]
        mean_disp = (disp_left + disp_right) / 2.0
        return np.asarray(mean_disp, dtype=float), True

    return np.zeros(2, dtype=float), False


class TemporalDropletTracker:
    """Stateful temporal tracker for sessile and pendant droplet video sequences.

    Maintains physical calibration invariants (substrate baseline for sessile,
    needle cannula for pendant) established on the first analyzed frame, and tracks
    the droplet boundary across time via localized ROI active contour warm-starts.
    """

    def __init__(
        self,
        pipeline: str = "sessile",
        use_optical_flow: bool = True,
        snake_iterations: int = 15,
        alpha: float = 0.015,
        beta: float = 10.0,
        gamma: float = 0.01,
        gaussian_sigma: float = 1.5,
        num_nodes: int = 100,
    ) -> None:
        self.pipeline = pipeline.lower()
        self.use_optical_flow = use_optical_flow
        self.snake_iterations = snake_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gaussian_sigma = gaussian_sigma
        self.num_nodes = num_nodes

        # Physical invariants locked from frame 1
        self.reference_baseline: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.reference_needle: tuple[int, int, int, int] | None = None
        self.reference_scale: float | None = None

        # Tracking state
        self.previous_contour: np.ndarray | None = None
        self.previous_contacts: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.previous_gray: np.ndarray | None = None
        self.previous_area: float | None = None
        self.contact_velocity: np.ndarray = np.zeros(2, dtype=float)
        self.is_tracking: bool = False
        self.frames_tracked: int = 0

    def reset(self) -> None:
        """Reset temporal state while retaining locked physical invariants."""
        self.previous_contour = None
        self.previous_contacts = None
        self.previous_gray = None
        self.previous_area = None
        self.contact_velocity = np.zeros(2, dtype=float)
        self.is_tracking = False
        self.frames_tracked = 0

    def reset_all(self) -> None:
        """Reset all state including locked physical invariants."""
        self.reset()
        self.reference_baseline = None
        self.reference_needle = None
        self.reference_scale = None

    def initialize(
        self,
        image: np.ndarray,
        detection: dict[str, Any],
        scale: float | None = None,
    ) -> bool:
        """Lock reference invariants and initialize tracking state from a valid detection.

        Args:
            image: Frame image.
            detection: Detection dictionary containing drop_contour, substrate_line, etc.
            scale: Optional calibrated px_per_mm scale.

        Returns:
            True if tracker initialized successfully, False otherwise.
        """
        contour_val = detection.get("drop_contour")
        if contour_val is None or len(contour_val) < 4:
            return False

        contour = np.asarray(contour_val, dtype=float).reshape(-1, 2)

        # Lock physical invariants if not yet locked
        if self.reference_baseline is None and "substrate_line" in detection:
            line = detection["substrate_line"]
            if line is not None:
                self.reference_baseline = (
                    (float(line[0][0]), float(line[0][1])),
                    (float(line[1][0]), float(line[1][1])),
                )

        if self.reference_needle is None and "needle_rect" in detection:
            needle = detection["needle_rect"]
            if needle is not None and len(needle) == 4:
                self.reference_needle = (
                    int(needle[0]), int(needle[1]), int(needle[2]), int(needle[3])
                )

        if self.reference_scale is None and scale is not None and scale > 0:
            self.reference_scale = float(scale)

        contacts_val = detection.get("contact_points")
        if contacts_val is not None and len(contacts_val) == 2:
            self.previous_contacts = (
                (float(contacts_val[0][0]), float(contacts_val[0][1])),
                (float(contacts_val[1][0]), float(contacts_val[1][1])),
            )
        else:
            self.previous_contacts = None

        # Resample contour to target nodes
        n_target = min(max(self.num_nodes, 20), 300)
        self.previous_contour = resample_contour_arclength(
            contour, n_points=n_target, closed=True
        )

        self.previous_gray = ensure_gray(image)
        self.previous_area = float(abs(cv2.contourArea(np.asarray(self.previous_contour, dtype=np.float32))))
        self.contact_velocity = np.zeros(2, dtype=float)
        self.is_tracking = True
        self.frames_tracked = 0
        return True

    def track_frame(
        self,
        image: np.ndarray,
        dt: float = 0.033,
    ) -> dict[str, Any] | None:
        """Track droplet in the current frame using localized ROI active contour warm-starting.

        Args:
            image: Current frame image.
            dt: Time elapsed since previous frame in seconds.

        Returns:
            Detection dictionary matching auto_detect_features schema on success,
            or None if tracking fails quality gating (triggering cold-start fallback).
        """
        if not self.is_tracking or self.previous_contour is None or self.previous_gray is None:
            return None

        curr_gray = ensure_gray(image)
        h_img, w_img = curr_gray.shape[:2]

        # 1. Optical flow displacement estimation
        disp = np.zeros(2, dtype=float)
        if self.use_optical_flow and self.previous_contacts is not None:
            of_disp, of_ok = estimate_contact_velocity_optical_flow(
                self.previous_gray, curr_gray, self.previous_contacts
            )
            if of_ok:
                disp = of_disp
            else:
                disp = self.contact_velocity * dt
        else:
            disp = self.contact_velocity * dt

        # 2. Compute predicted ROI
        x0, y0, w_roi, h_roi = predict_droplet_roi(
            self.previous_contour, self.previous_contacts, disp, (h_img, w_img)
        )

        if w_roi < 10 or h_roi < 10:
            self.reset()
            return None

        roi_img = curr_gray[y0 : y0 + h_roi, x0 : x0 + w_roi]

        # 3. Local contour warm-start
        local_contour = (self.previous_contour + disp.reshape(1, 2)).copy()
        local_contour[:, 0] -= x0
        local_contour[:, 1] -= y0

        # Substrate masking in ROI
        snake_roi = roi_img.copy()
        sub_y_roi = None
        if self.reference_baseline is not None:
            line_y = (self.reference_baseline[0][1] + self.reference_baseline[1][1]) / 2.0
            sub_y_roi = int(round(line_y - y0))
            if 0 <= sub_y_roi < snake_roi.shape[0]:
                snake_roi[sub_y_roi:, :] = 255
                local_contour[:, 1] = np.minimum(local_contour[:, 1], sub_y_roi - 1.0)

        # 4. Fast active contour evolution (5-15 iterations)
        cfg = ActiveContourConfig(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            w_edge=1.0,
            w_line=0.0,
            w_balloon=0.0,
            gaussian_sigma=self.gaussian_sigma,
            max_iterations=self.snake_iterations,
            convergence=0.01,
        )

        try:
            res = evolve_active_contour(
                snake_roi,
                local_contour,
                config=cfg,
                boundary_condition=SnakeBoundaryCondition.PERIODIC,
            )
            e_xy = res.xy
        except Exception as exc:
            logger.debug("Temporal active contour evolution failed: %s", exc)
            self.reset()
            return None

        # Transform back to global image coordinates
        global_contour = e_xy.copy()
        global_contour[:, 0] += x0
        global_contour[:, 1] += y0

        if sub_y_roi is not None:
            sub_y_global = sub_y_roi + y0
            global_contour[:, 1] = np.minimum(global_contour[:, 1], sub_y_global - 1.0)

        # 5. Extract contact points and apex
        if self.pipeline == "pendant":
            apex_idx = int(np.argmax(global_contour[:, 1]))
        else:
            apex_idx = int(np.argmin(global_contour[:, 1]))
        apex_xy = (float(global_contour[apex_idx, 0]), float(global_contour[apex_idx, 1]))

        new_contacts = None
        if self.reference_baseline is not None:
            clipped, contacts = clip_contour_to_substrate(
                global_contour, self.reference_baseline, apex_xy
            )
            if contacts is not None and len(contacts) == 2:
                c_left = (float(contacts[0][0]), float(contacts[0][1]))
                c_right = (float(contacts[1][0]), float(contacts[1][1]))
                if c_left[0] < c_right[0]:
                    new_contacts = (c_left, c_right)
                elif c_right[0] < c_left[0]:
                    new_contacts = (c_right, c_left)
            else:
                p1, p2 = find_contact_points_from_contour(
                    global_contour, self.reference_baseline, tolerance=25.0
                )
                if p1 is not None and p2 is not None:
                    c_left = (float(p1[0]), float(p1[1]))
                    c_right = (float(p2[0]), float(p2[1]))
                    if c_left[0] < c_right[0]:
                        new_contacts = (c_left, c_right)
                    elif c_right[0] < c_left[0]:
                        new_contacts = (c_right, c_left)

        if self.pipeline == "sessile" and new_contacts is None:
            self.reset()
            return None

        # 6. Physical quality gates
        current_area = float(abs(cv2.contourArea(np.asarray(global_contour, dtype=np.float32))))

        if self.previous_area is not None and self.previous_area > 0:
            area_change = abs(current_area - self.previous_area) / self.previous_area
            if area_change > 0.25:
                # Sudden excessive area jump (> 25%) signals tracking anomaly
                self.reset()
                return None

        if self.previous_contacts is not None and new_contacts is not None:
            prior_width = max(1.0, self.previous_contacts[1][0] - self.previous_contacts[0][0])
            disp_contacts = max(
                np.linalg.norm(np.asarray(new_contacts[i]) - np.asarray(self.previous_contacts[i]))
                for i in (0, 1)
            )
            if disp_contacts > 0.10 * prior_width:
                # Contact displacement jump (> 10% of width) signals tracking anomaly
                self.reset()
                return None

            # Update contact velocity
            old_center = np.mean(np.asarray(self.previous_contacts), axis=0)
            new_center = np.mean(np.asarray(new_contacts), axis=0)
            if dt > 0:
                self.contact_velocity = (new_center - old_center) / dt

        # 7. Update tracker state
        self.previous_contour = resample_contour_arclength(
            global_contour, n_points=self.num_nodes, closed=True
        )
        self.previous_contacts = new_contacts
        self.previous_gray = curr_gray
        self.previous_area = current_area
        self.frames_tracked += 1

        result_dict: dict[str, Any] = {
            "drop_contour": global_contour,
            "apex_point": apex_xy,
            "roi_rect": (x0, y0, w_roi, h_roi),
            "detector_diagnostics": {
                "tracking": {
                    "tracked": True,
                    "frames_tracked": self.frames_tracked,
                    "roi": [x0, y0, w_roi, h_roi],
                }
            },
        }

        if self.reference_baseline is not None:
            result_dict["substrate_line"] = self.reference_baseline
        if self.reference_needle is not None:
            result_dict["needle_rect"] = self.reference_needle
        if new_contacts is not None:
            result_dict["contact_points"] = new_contacts

        return result_dict


__all__ = [
    "TemporalDropletTracker",
    "predict_droplet_roi",
    "estimate_contact_velocity_optical_flow",
]
