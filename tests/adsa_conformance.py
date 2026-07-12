"""Deterministic synthetic ADSA detector-conformance fixtures and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from menipy.common.auto_calibrator import AutoCalibrator, CalibrationResult


@dataclass
class SyntheticADSA:
    image: np.ndarray
    target_mask: np.ndarray
    needle_rect: tuple[int, int, int, int] | None
    baseline: tuple[tuple[float, float], tuple[float, float]] | None
    apex: tuple[float, float]
    contacts: tuple[tuple[float, float], tuple[float, float]] | None


def _transform_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    vector = matrix @ np.asarray([point[0], point[1], 1.0])
    return float(vector[0]), float(vector[1])


def _mask_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    return None if points is None else cv2.boundingRect(points)


def generate_case(spec: dict[str, Any], seed: int) -> SyntheticADSA:
    """Generate one manifest case without writing binary fixtures to disk."""
    rng = np.random.default_rng(seed + int(spec["variant"]) * 101)
    height, width = 480, 640
    background, foreground = 200, 50
    image = np.full((height, width, 3), background, dtype=np.uint8)
    target = np.zeros((height, width), dtype=np.uint8)
    needle_mask = np.zeros_like(target)
    pipeline = str(spec["pipeline"])

    if pipeline == "sessile":
        substrate_y = 400
        rx, ry = (80, 70) if not spec["variant"] else (72, 64)
        center = (320, 370)
        cv2.rectangle(image, (0, substrate_y - 2), (width, height), (foreground,) * 3, -1)
        cv2.ellipse(image, center, (rx, ry), 0, 0, 360, (foreground,) * 3, -1)
        cv2.ellipse(target, center, (rx, ry), 0, 0, 360, 255, -1)
        needle_bottom = center[1] - ry + (10 if spec["variant"] else -60)
        cv2.rectangle(image, (300, 0), (340, needle_bottom), (60,) * 3, -1)
        cv2.rectangle(needle_mask, (300, 0), (340, needle_bottom), 255, -1)
        baseline = ((0.0, float(substrate_y)), (float(width - 1), float(substrate_y)))
        apex = (float(center[0]), float(center[1] - ry))
        contact_dx = rx * np.sqrt(max(0.0, 1.0 - ((substrate_y - center[1]) / ry) ** 2))
        contacts = (
            (float(center[0] - contact_dx), float(substrate_y)),
            (float(center[0] + contact_dx), float(substrate_y)),
        )
        if spec["variant"]:
            visible_lobe_y = center[1] - ry + 6
            target[:visible_lobe_y, :] = 0
            apex = (float(center[0]), float(visible_lobe_y))
        target[substrate_y - 4 :, :] = 0
        target_y, target_x = np.where(target > 0)
        visible_contact_y = int(np.max(target_y))
        visible_contact_x = target_x[target_y == visible_contact_y]
        contacts = (
            (float(np.min(visible_contact_x)), float(substrate_y)),
            (float(np.max(visible_contact_x)), float(substrate_y)),
        )
    else:
        center = (320, 205)
        rx, ry = (82, 120) if not spec["variant"] else (72, 108)
        needle_bottom = center[1] - ry + 3
        cv2.rectangle(image, (302, 0), (338, needle_bottom), (45,) * 3, -1)
        cv2.rectangle(needle_mask, (302, 0), (338, needle_bottom), 255, -1)
        cv2.ellipse(image, center, (rx, ry), 0, 0, 360, (foreground,) * 3, -1)
        cv2.ellipse(target, center, (rx, ry), 0, 0, 360, 255, -1)
        cv2.bitwise_or(target, needle_mask, dst=target)
        baseline = None
        apex = (float(center[0]), float(center[1] + ry))
        contacts = ((302.0, float(needle_bottom)), (338.0, float(needle_bottom)))

    perturbation = str(spec["perturbation"])
    matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    if perturbation == "reflection":
        matrix = np.asarray([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0]])
    elif perturbation == "tilt":
        if pipeline == "sessile":
            angle = 0.8 if not spec["variant"] else -1.2
        else:
            angle = 0.1 if not spec["variant"] else -0.1
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    elif perturbation == "safe_crop":
        shift = 3 if not spec["variant"] else -4
        matrix = np.asarray([[1.0, 0.0, shift], [0.0, 1.0, 0.0]])

    if perturbation in {"reflection", "tilt", "safe_crop"}:
        image = cv2.warpAffine(image, matrix, (width, height), borderValue=(background,) * 3)
        target = cv2.warpAffine(target, matrix, (width, height), flags=cv2.INTER_NEAREST)
        needle_mask = cv2.warpAffine(
            needle_mask, matrix, (width, height), flags=cv2.INTER_NEAREST
        )
        apex = _transform_point(matrix, apex)
        contacts = (
            (_transform_point(matrix, contacts[0]), _transform_point(matrix, contacts[1]))
            if contacts is not None
            else None
        )
        if baseline is not None:
            baseline = (
                _transform_point(matrix, baseline[0]),
                _transform_point(matrix, baseline[1]),
            )
    elif perturbation == "blur":
        image = cv2.GaussianBlur(image, (0, 0), 1.0 + 0.5 * spec["variant"])
    elif perturbation == "noise":
        sigma = 3.0 + 2.0 * spec["variant"]
        image = np.clip(image.astype(float) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)
    elif perturbation == "jpeg":
        quality = 85 if not spec["variant"] else 70
        encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif perturbation == "low_contrast":
        image = np.where(image < 128, 105, 165).astype(np.uint8)
    elif perturbation == "destructive_crop":
        image[:] = background
    elif perturbation == "contact_occluded":
        if pipeline == "sessile":
            cv2.rectangle(image, (200, 250), (440, 430), (background,) * 3, -1)
        else:
            cv2.rectangle(image, (225, 70), (415, 360), (background,) * 3, -1)
    elif perturbation == "missing_needle":
        image[needle_mask > 0] = background
    elif perturbation == "severe_degradation":
        image[:] = 128

    return SyntheticADSA(
        image=image,
        target_mask=target,
        needle_rect=_mask_rect(needle_mask),
        baseline=baseline,
        apex=apex,
        contacts=contacts,
    )


def evaluate_detection(case: SyntheticADSA, pipeline: str) -> tuple[CalibrationResult, dict[str, float], list[str]]:
    """Run calibration and calculate conformance metrics and rejection codes."""
    result = AutoCalibrator(case.image, pipeline).detect_all()
    reasons: list[str] = []
    if result.drop_contour is None:
        reasons.append("drop_not_detected")
    if result.needle_rect is None:
        reasons.append("needle_not_detected")
    if result.apex_point is None:
        reasons.append("apex_not_detected")
    if result.contact_points is None:
        reasons.append("contact_points_not_detected")
    if pipeline == "sessile" and result.substrate_line is None:
        reasons.append("substrate_not_detected")

    metrics: dict[str, float] = {}
    if result.drop_contour is not None:
        predicted = np.zeros_like(case.target_mask)
        contour = np.asarray(result.drop_contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(predicted, [contour], -1, 255, -1)
        intersection = np.logical_and(predicted > 0, case.target_mask > 0).sum()
        union = np.logical_or(predicted > 0, case.target_mask > 0).sum()
        metrics["iou"] = float(intersection / union) if union else 0.0
        true_contours, _ = cv2.findContours(
            case.target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if true_contours:
            observed_contours, _ = cv2.findContours(
                predicted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            observed = max(observed_contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            expected = max(true_contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            metrics["hausdorff_px"] = float(
                max(
                    cKDTree(expected).query(observed)[0].max(),
                    cKDTree(observed).query(expected)[0].max(),
                )
            )
    if result.needle_rect is not None and case.needle_rect is not None:
        x, _, width, _ = result.needle_rect
        tx, _, twidth, _ = case.needle_rect
        metrics["needle_width_error_px"] = float(abs(width - twidth))
        metrics["needle_center_error_px"] = float(abs((x + width / 2) - (tx + twidth / 2)))
    if result.apex_point is not None:
        metrics["apex_error_px"] = float(np.linalg.norm(np.asarray(result.apex_point) - case.apex))
    if result.contact_points is not None and case.contacts is not None:
        observed = sorted(result.contact_points)
        expected = sorted(case.contacts)
        if pipeline == "sessile":
            # Baseline displacement is measured independently below.
            metrics["contact_error_px"] = float(
                max(abs(float(a[0]) - float(b[0])) for a, b in zip(observed, expected))
            )
        else:
            metrics["contact_error_px"] = float(
                max(
                    np.linalg.norm(np.asarray(a) - np.asarray(b))
                    for a, b in zip(observed, expected)
                )
            )
    if result.substrate_line is not None and case.baseline is not None:
        observed = result.substrate_line
        oy = (observed[0][1] + observed[1][1]) / 2
        ty = (case.baseline[0][1] + case.baseline[1][1]) / 2
        metrics["baseline_y_error_px"] = float(abs(oy - ty))
        observed_angle = np.degrees(np.arctan2(observed[1][1] - observed[0][1], observed[1][0] - observed[0][0]))
        true_angle = np.degrees(np.arctan2(case.baseline[1][1] - case.baseline[0][1], case.baseline[1][0] - case.baseline[0][0]))
        delta = abs(observed_angle - true_angle) % 180.0
        metrics["baseline_angle_error_deg"] = float(min(delta, 180.0 - delta))
    return result, metrics, reasons


__all__ = ["SyntheticADSA", "evaluate_detection", "generate_case"]
