"""Deterministic, image-free Phase-B geometry benchmark fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from menipy.common.geometry_prototypes import (
    detect_bilateral_needle,
    robust_pendant_initializer,
)
from menipy.pipelines.sessile.metrics import compute_sessile_metrics


@dataclass
class GeometryCase:
    image: np.ndarray | None = None
    contour: np.ndarray | None = None
    expected: dict[str, Any] | None = None


def generate_geometry_case(spec: dict[str, Any], seed: int = 20260710) -> GeometryCase:
    rng = np.random.default_rng(seed + sum(ord(c) for c in str(spec["id"])))
    kind = spec["kind"]
    if kind == "needle":
        image = np.full((240, 320), 210, dtype=np.uint8)
        if spec["perturbation"] == "severe":
            image.fill(128)
            return GeometryCase(image=image, expected={"valid": False})
        angle = 0.0 if spec["perturbation"] == "clean" else (0.8 if spec["pipeline"] == "sessile" else -0.6)
        mask = np.zeros_like(image)
        cv2.rectangle(mask, (145, 10), (175, 150), 255, -1)
        matrix = cv2.getRotationMatrix2D((160, 120), angle, 1.0)
        mask = cv2.warpAffine(mask, matrix, (320, 240), flags=cv2.INTER_NEAREST)
        image[mask > 0] = 45
        if spec["perturbation"] == "noise":
            image = np.clip(image.astype(float) + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
        if spec["perturbation"] == "occlusion":
            image[130:155, :] = 210
        return GeometryCase(image=image, expected={"valid": True, "center_x": 160.0, "width": 30.0})

    if kind == "initializer":
        t = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        contour = np.column_stack([160 + 55 * np.cos(t), 120 + 90 * np.sin(t)])
        if spec["perturbation"] == "tilt":
            matrix = cv2.getRotationMatrix2D((160, 120), 4.0, 1.0)
            contour = cv2.transform(contour.astype(np.float32)[None, :, :], matrix)[0]
        if spec["perturbation"] == "noise":
            contour += rng.normal(0, 1.0, contour.shape)
        return GeometryCase(contour=contour, expected={"valid": True})

    # Contact-angle cases use a circular cap as deterministic geometric input.
    angle = float(spec.get("angle_deg", 90.0))
    theta = np.radians(np.linspace(180.0 - angle, angle, 160))
    radius = 80.0
    center = np.array([160.0, 180.0])
    contour = np.column_stack([center[0] + radius * np.cos(theta), center[1] - radius * np.sin(theta)])
    contour = np.vstack([contour, [[80.0, 180.0], [240.0, 180.0]]])
    return GeometryCase(contour=contour, expected={"valid": True, "angle_deg": angle})


def evaluate_geometry_case(case: GeometryCase, spec: dict[str, Any]) -> dict[str, Any]:
    if spec["kind"] == "needle":
        outcome = detect_bilateral_needle(case.image)
        return {"accepted": outcome.accepted, "diagnostics": outcome.to_diagnostics()}
    if spec["kind"] == "initializer":
        outcome = robust_pendant_initializer(case.contour)
        return {"accepted": outcome.accepted, "diagnostics": outcome.to_diagnostics()}
    metrics = compute_sessile_metrics(
        case.contour, 1.0,
        substrate_line=((0.0, 180.0), (320.0, 180.0)),
        apex=(160.0, 100.0),
        contact_points=((80, 180), (240, 180)),
        contact_angle_method="auto_residual",
    )
    return {"accepted": bool(metrics.get("theta_left_deg", 0) > 0 and metrics.get("theta_right_deg", 0) > 0), "metrics": metrics}


__all__ = ["GeometryCase", "evaluate_geometry_case", "generate_geometry_case"]
