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
