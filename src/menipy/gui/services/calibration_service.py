"""Non-Qt calibration computation shared by workers and the calibration wizard."""

import logging
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from menipy.common.cancellation import check_cancelled

logger = logging.getLogger(__name__)


class CalibrationComputation:
    def __init__(self, image, pipeline_name, result=None):
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not load the calibration image.")
        self.original_image = image
        self.pipeline_name = pipeline_name
        self.result = result

    def run(self):
        check_cancelled()
        # Preserve manually set values before auto-detection
        manual_substrate = None
        manual_roi = None
        manual_needle = None

        if self.result:
            # Check if these were manually set (confidence = 1.0 indicates manual)
            if (
                self.result.substrate_line
                and self.result.confidence_scores.get("substrate", 0) >= 1.0
            ):
                manual_substrate = self.result.substrate_line
                logger.info("Preserving manual substrate line")
            if (
                self.result.roi_rect
                and self.result.confidence_scores.get("roi", 0) >= 1.0
            ):
                manual_roi = self.result.roi_rect
                logger.info("Preserving manual ROI")
            if (
                self.result.needle_rect
                and self.result.confidence_scores.get("needle", 0) >= 1.0
            ):
                manual_needle = self.result.needle_rect
                logger.info("Preserving manual needle")

        # Import here to avoid circular imports
        from menipy.common.auto_calibrator import AutoCalibrator, run_auto_calibration

        self.result = self._run_best_auto_calibration(
            run_auto_calibration,
            allow_fallback=not (manual_substrate or manual_roi or manual_needle),
        )

        # Restore manually set values
        need_redetect_drop = False

        if manual_substrate:
            self.result.substrate_line = manual_substrate
            self.result.confidence_scores["substrate"] = 1.0
            need_redetect_drop = True  # Need to re-detect drop with correct substrate
        if manual_roi:
            self.result.roi_rect = manual_roi
            self.result.confidence_scores["roi"] = 1.0
        if manual_needle:
            self.result.needle_rect = manual_needle
            self.result.confidence_scores["needle"] = 1.0
            need_redetect_drop = (
                True  # Need to re-detect drop with correct needle filter
            )

        # Re-run drop detection if manual substrate/needle was set
        # This ensures drop is detected relative to correct substrate line
        if need_redetect_drop and manual_substrate:
            logger.info("Re-running drop detection with manual substrate line...")
            calibrator = AutoCalibrator(self.original_image, self.pipeline_name)
            # Set the correct substrate_y from manual line
            p1, p2 = manual_substrate
            calibrator._substrate_y = (p1[1] + p2[1]) // 2
            # Set needle rect if available
            if manual_needle:
                calibrator._needle_rect = manual_needle
            elif self.result.needle_rect:
                calibrator._needle_rect = self.result.needle_rect
            # Segment and detect drop
            calibrator._segment_image_adaptive()
            drop_contour, contact_pts, drop_conf = calibrator._detect_drop_sessile()
            if drop_contour is not None and len(drop_contour) > 0:
                self.result.drop_contour = drop_contour
                self.result.contact_points = contact_pts
                self.result.confidence_scores["drop"] = drop_conf
                logger.info(f"Drop re-detected with {len(drop_contour)} points")

        check_cancelled()
        return self.original_image, self.result

    def _run_best_auto_calibration(self, runner, *, allow_fallback: bool = True):
        """Run requested calibration, then try supported detector branches if needed."""
        check_cancelled()
        primary = runner(self.original_image, self.pipeline_name)
        if not allow_fallback:
            return primary

        supported_detectors = {"pendant", "sessile"}
        candidates = (
            [(self.pipeline_name, primary)]
            if self.pipeline_name in supported_detectors
            else []
        )
        for detector_name in ("pendant", "sessile"):
            check_cancelled()
            if detector_name == self.pipeline_name:
                continue
            try:
                candidates.append(
                    (detector_name, runner(self.original_image, detector_name))
                )
            except Exception:
                logger.debug(
                    "Fallback auto-calibration failed for %s",
                    detector_name,
                    exc_info=True,
                )

        preferred_fallback = {"captive_bubble": "pendant"}.get(self.pipeline_name)
        if preferred_fallback is not None:
            for detector_name, candidate in candidates:
                if (
                    detector_name == preferred_fallback
                    and self._calibration_score(candidate) > 0
                ):
                    candidate.confidence_scores["detector_pipeline"] = detector_name
                    return candidate

        best_name, best = max(
            candidates, key=lambda item: self._calibration_score(item[1])
        )
        primary_score = self._calibration_score(primary)
        best_score = self._calibration_score(best)
        if self.pipeline_name not in supported_detectors or (
            best is not primary and best_score > primary_score + 0.15
        ):
            best.confidence_scores["detector_pipeline"] = best_name
            logger.info(
                "Auto-calibration used %s detector instead of %s (score %.2f > %.2f)",
                best_name,
                self.pipeline_name,
                best_score,
                primary_score,
            )
            return best
        primary.confidence_scores.setdefault("detector_pipeline", self.pipeline_name)
        return primary

    def _calibration_score(self, result) -> float:
        """Score a calibration result by useful detected geometry."""
        if result is None:
            return 0.0
        score = float(result.confidence_scores.get("overall", 0.0) or 0.0)
        if result.drop_contour is not None:
            try:
                if len(result.drop_contour) > 0:
                    score += 0.35
            except Exception:
                score += 0.2
        if result.needle_rect:
            score += 0.12
        if result.contact_points:
            score += 0.1
        if result.roi_rect:
            score += 0.08
        if result.substrate_line:
            score += 0.06
        return score


def calibration_task(parameters, token):
    return CalibrationComputation(
        parameters["image"], parameters["pipeline"], parameters.get("manual_result")
    ).run()


def prepare_stage_calibration(pipeline, parameters):
    from menipy.common.auto_calibrator import AutoCalibrator

    fallback_warnings = (
        []
        if parameters.get("needle_rect")
        else ["Needle width was unavailable; using fallback scale."]
    )

    image = parameters.get("image")
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if image is None:
        return (
            parameters,
            ["No selected image was available for silent auto-calibration."]
            + fallback_warnings,
        )
    check_cancelled()
    try:
        result = AutoCalibrator(image, pipeline).detect_all()
    except Exception as exc:
        return parameters, [f"Auto-calibration failed: {exc}"] + fallback_warnings
    check_cancelled()
    for attribute, keys in (
        ("roi_rect", ("roi", "roi_rect")),
        ("needle_rect", ("needle_rect",)),
        ("substrate_line", ("substrate_line", "contact_line")),
        ("drop_contour", ("drop_contour", "detected_contour")),
        ("contact_points", ("contact_points",)),
        ("apex_point", ("apex_point",)),
    ):
        value = getattr(result, attribute, None)
        if value is not None:
            for key in keys:
                parameters[key] = value
    diameter = float(
        (parameters.get("calibration_params") or {}).get("needle_diameter_mm", 0.54)
    )
    if result.needle_rect and diameter > 0:
        parameters["scale"] = {"px_per_mm": result.needle_rect[2] / diameter}
    warnings = [
        f"Auto-calibration did not detect {label}."
        for label, value in (
            ("ROI", result.roi_rect),
            ("needle region", result.needle_rect),
            ("drop contour", result.drop_contour),
        )
        if value is None
    ]
    if not parameters.get("needle_rect"):
        warnings.extend(fallback_warnings)
    return parameters, warnings
