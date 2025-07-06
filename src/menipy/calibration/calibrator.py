"""Calibration utilities ported from the legacy code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Calibration:
    """Simple calibration data container."""

    pixels_per_mm: float


_default_calibration = Calibration(pixels_per_mm=1.0)


def set_calibration(pixels_per_mm: float) -> None:
    """Set the global calibration value."""
    _default_calibration.pixels_per_mm = pixels_per_mm


def get_calibration() -> Calibration:
    """Return the current calibration data."""
    return _default_calibration


def mm_to_pixels(mm: float) -> float:
    """Convert a length in millimeters to pixels using current calibration."""
    return mm * _default_calibration.pixels_per_mm


def pixels_to_mm(pixels: float) -> float:
    """Convert a pixel distance to millimeters using current calibration."""
    return pixels / _default_calibration.pixels_per_mm


def calibrate_from_points(p1: tuple[float, float], p2: tuple[float, float], length_mm: float) -> float:
    """Set calibration based on two points and a known physical length."""
    from math import hypot

    pixel_length = hypot(p1[0] - p2[0], p1[1] - p2[1])
    if length_mm <= 0:
        raise ValueError("length_mm must be positive")
    set_calibration(pixel_length / length_mm)
    return pixel_length


def auto_calibrate(image, box: tuple[float, float, float, float], length_mm: float = 1.0) -> float:
    """Automatically calibrate using vertical edge detection within a box."""
    import cv2
    import numpy as np

    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Calibration ROI is empty")

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=gray.shape[0] // 2,
        maxLineGap=5,
    )
    if lines is None or len(lines) < 2:
        raise ValueError("Could not detect calibration lines")

    vertical_positions = []
    for x_start, y_start, x_end, y_end in lines[:, 0]:
        if abs(x_end - x_start) <= 3:
            vertical_positions.append((x_start + x_end) / 2)

    if len(vertical_positions) < 2:
        raise ValueError("Could not detect calibration lines")

    vertical_positions.sort()
    separation = float(vertical_positions[-1] - vertical_positions[0])
    if length_mm <= 0:
        raise ValueError("length_mm must be positive")
    set_calibration(separation / length_mm)
    return float(separation)
