"""Synthetic, binary-free Phase-D sequence generator and ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class SyntheticTemporalSequence:
    images: list[np.ndarray]
    timestamps_s: list[float]
    contacts: list[tuple[tuple[float, float], tuple[float, float]] | None]
    angles_deg: list[tuple[float, float] | None]
    states: list[str]
    baselines: list[tuple[tuple[float, float], tuple[float, float]] | None]


def generate_temporal_case(spec: dict[str, Any], seed: int) -> SyntheticTemporalSequence:
    """Render one deterministic injection/hold/withdrawal sequence in memory."""
    rng = np.random.default_rng(seed + sum(ord(char) for char in spec["id"]))
    count = int(spec.get("frames", 60))
    fps = 30.0
    images: list[np.ndarray] = []
    contacts = []
    angles = []
    states = []
    baselines = []
    perturbation = str(spec["perturbation"])
    for index in range(count):
        phase = index / max(1, count - 1)
        if phase < 0.4:
            radius = 42.0 + 18.0 * phase / 0.4
            state, angle = "advancing", 118.0
        elif phase < 0.6:
            radius = 60.0
            state, angle = "pinned", 100.0
        else:
            radius = 60.0 - 16.0 * (phase - 0.6) / 0.4
            state, angle = "receding", 82.0
        baseline_y = 190.0
        baseline_angle = 0.0
        if perturbation in {"drift", "mixed", "camera_shift"}:
            baseline_y += min(4.0, index * 0.05)
        if perturbation in {"tilt", "mixed"}:
            baseline_angle = 1.0
        slope = np.tan(np.radians(baseline_angle))
        baseline = ((0.0, baseline_y - slope * 160), (319.0, baseline_y + slope * 159))
        left, right = (160.0 - radius, baseline_y), (160.0 + radius, baseline_y)
        image = np.full((240, 320, 3), 205, dtype=np.uint8)
        cv2.rectangle(image, (0, int(round(baseline_y))), (319, 239), (50, 50, 50), -1)
        cv2.ellipse(image, (160, int(round(baseline_y - 35))), (int(round(radius)), 48), baseline_angle, 0, 360, (45, 45, 45), -1)
        if perturbation in {"noise", "mixed"}:
            image = np.clip(image.astype(float) + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
        if perturbation == "blur":
            image = cv2.GaussianBlur(image, (5, 5), 1.2)
        if perturbation in {"occlusion", "loss_prolonged"}:
            width = 3 if perturbation == "occlusion" else 10
            if count // 2 <= index < count // 2 + width:
                cv2.rectangle(image, (80, 120), (240, 205), (205, 205, 205), -1)
        invalid = (not spec["expected_valid"]) and perturbation not in {"occlusion"}
        if invalid and perturbation in {"contacts_hidden", "destructive_crop", "ambiguous"}:
            contacts.append(None)
            angles.append(None)
        else:
            contacts.append((left, right))
            angles.append((angle, angle))
        states.append("invalid" if contacts[-1] is None else state)
        baselines.append(None if invalid and perturbation == "baseline_unstable" else baseline)
        images.append(image)
    timestamps = [index / fps for index in range(count)]
    if perturbation == "timestamps_invalid" and len(timestamps) > 2:
        timestamps[2] = timestamps[1]
    return SyntheticTemporalSequence(images, timestamps, contacts, angles, states, baselines)


__all__ = ["SyntheticTemporalSequence", "generate_temporal_case"]
