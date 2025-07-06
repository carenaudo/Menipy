"""Surface tension and related pendant-drop calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Jennings–Pallas cubic correlation -----------------------------------------

def jennings_pallas_beta(s1: float) -> float:
    """Return the dimensionless form factor ``beta``.

    Parameters
    ----------
    s1:
        Ratio ``De / (2 * r0)`` where ``De`` is the maximum diameter and
        ``r0`` is the apex radius of curvature. Valid for ``0.5 <= s1 <= 2.0``.
    """
    a3, a2, a1, a0 = 0.41727, -1.0908, 1.3906, 0.005306
    return ((a3 * s1 + a2) * s1 + a1) * s1 + a0


# Surface tension from Young–Laplace ----------------------------------------

def surface_tension(delta_rho: float, g: float, r0_mm: float, beta: float) -> float:
    """Return surface tension in Newton per metre."""
    r0 = r0_mm / 1000.0
    return delta_rho * g * r0**2 / beta


# Bond number ---------------------------------------------------------------

def bond_number(delta_rho: float, g: float, r0_mm: float, gamma: float) -> float:
    """Return dimensionless Bond number."""
    r0 = r0_mm / 1000.0
    return delta_rho * g * r0**2 / gamma


# Volume by revolution of profile ------------------------------------------

def volume_from_contour(contour_mm: NDArray[np.float_, np.float_]) -> float:
    """Return droplet volume in microlitres from a 2-D contour.

    Parameters
    ----------
    contour_mm:
        ``Nx2`` array of ``(r, y)`` coordinates in millimetres, where ``r`` is
        the radial distance from the symmetry axis and ``y`` is the vertical
        coordinate.
    """
    r = contour_mm[:, 0] / 1000.0
    y = contour_mm[:, 1] / 1000.0
    idx = np.argsort(y)
    y_sorted = y[idx]
    r_sorted = r[idx]
    vol = np.pi * np.trapz(r_sorted**2, y_sorted)  # m^3
    return vol * 1e9
