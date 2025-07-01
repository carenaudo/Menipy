"""Calibration utilities for pixel to physical units."""

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
    """Set calibration based on two points and a known physical length.

    Parameters
    ----------
    p1, p2:
        Endpoints of the reference line in pixels.
    length_mm:
        The physical distance in millimeters corresponding to the line.

    Returns
    -------
    float
        The measured pixel length between the two points.
    """
    from math import hypot

    pixel_length = hypot(p1[0] - p2[0], p1[1] - p2[1])
    if length_mm <= 0:
        raise ValueError("length_mm must be positive")
    set_calibration(pixel_length / length_mm)
    return pixel_length
