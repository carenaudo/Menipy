"""Exact synthetic pendant drops for zone-spline tests, profiling and benchmarks.

Profiles come from an adaptive ``solve_ivp`` integration of the dimensionless
Young-Laplace equation, independent of the RK4 table of
:mod:`menipy.math.pendant_box`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

from tests.synthetic_sessile import degrade

EQUATOR_RADIUS_PX = 150.0


@lru_cache(maxsize=64)
def pendant_half_profile(bond: float, needle_fraction: float = 0.55) -> dict:
    """Dense right half-profile, apex at the origin, ``z`` upward, apex radius 1.

    The profile ends where, past the equator, the radius falls to
    ``needle_fraction`` of the equatorial radius (never below the neck).

    Returns
    -------
    dict
        ``x, z, phi`` arrays (spacing about ``2e-4``), ``x_eq`` and ``i_eq``.
    """

    def rhs(_s, y):
        x, z, phi = y
        return [np.cos(phi), np.sin(phi), 2.0 - bond * z - np.sin(phi) / x]

    def closing(s, y):
        return y[0] - 0.05 if s > 1.0 else 1.0

    closing.terminal = True
    s0 = 1e-5
    sol = solve_ivp(rhs, (s0, 12.0), [s0, s0**2 / 2, s0], rtol=1e-11, atol=1e-13,
                    max_step=1e-2, dense_output=True, events=closing)
    s = np.arange(s0, sol.t[-1], 2e-4)
    x, z, phi = sol.sol(s)
    i_eq = int(np.argmax(phi >= np.pi / 2))
    turned = np.flatnonzero(phi[i_eq:] > np.pi)
    end = i_eq + int(turned[0]) if turned.size else len(x)
    neck = float(np.min(x[i_eq:end]))
    fraction = max(needle_fraction, neck / x[i_eq] + 0.03)
    i_n = int(np.flatnonzero((np.arange(len(x)) > i_eq) & (x <= fraction * x[i_eq]))[0])
    return {"x": x[: i_n + 1], "z": z[: i_n + 1], "phi": phi[: i_n + 1],
            "x_eq": float(x[i_eq]), "i_eq": i_eq}


def pendant_truth(bond: float, needle_fraction: float = 0.55,
                  equator_radius_px: float = EQUATOR_RADIUS_PX) -> dict:
    """Apex radius, needle angle and equator of the scaled exact profile."""
    p = pendant_half_profile(bond, needle_fraction)
    k = equator_radius_px / p["x_eq"]
    return {"bond": bond, "apex_radius_px": k, "needle_angle_deg": float(np.degrees(p["phi"][-1])),
            "height_px": float(p["z"][-1] * k), "needle_radius_px": float(p["x"][-1] * k)}


def pendant_interface(bond: float, needle_fraction: float = 0.55, spacing_px: float = 1.0,
                      equator_radius_px: float = EQUATOR_RADIUS_PX) -> np.ndarray:
    """Symmetric P1 -> apex -> P2 profile, local ``(x, z)`` with the apex at the origin."""
    p = pendant_half_profile(bond, needle_fraction)
    k = equator_radius_px / p["x_eq"]
    xy = np.column_stack([p["x"], p["z"]]) * k
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    grid = np.append(np.arange(0.0, s[-1], spacing_px), s[-1])
    right = np.column_stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])])
    return np.vstack([right[::-1] * [-1.0, 1.0], right[1:]])


def to_image(local: np.ndarray, *, apex=(320.0, 420.0), tilt_deg: float = 0.0) -> np.ndarray:
    """Map local ``(x, z)`` (z up) to image pixels, apex at ``apex``, rotated by ``tilt_deg``."""
    a = np.radians(tilt_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    img = np.column_stack([local[:, 0], -local[:, 1]]) @ rot.T
    return img + np.asarray(apex, float)


def pendant_case(bond: float, noise: str = "clean", *, seed: int = 0, tilt_deg: float = 0.0,
                 needle_fraction: float = 0.55, equator_radius_px: float = EQUATOR_RADIUS_PX,
                 apex=(320.0, 420.0)):
    """Return ``(interface_points, p1, p2)`` in image coordinates (needle up)."""
    exact = pendant_interface(bond, needle_fraction, equator_radius_px=equator_radius_px)
    pts = to_image(degrade(exact, noise, np.random.default_rng(seed)), apex=apex, tilt_deg=tilt_deg)
    p1, p2 = to_image(exact[[0, -1]], apex=apex, tilt_deg=tilt_deg)
    return pts, p1, p2


def render_pendant(
    bond: float,
    *,
    needle_fraction: float = 0.55,
    equator_radius_px: float = EQUATOR_RADIUS_PX,
    shape: tuple[int, int] = (480, 640),
    apex=(320.0, 420.0),
    tilt_deg: float = 0.0,
    drop_gray: float = 30.0,
    background_gray: float = 220.0,
    blur_sigma: float = 1.2,
    noise_gray: float = 2.0,
    seed: int = 0,
    supersample: int = 4,
):
    """Render a dark pendant drop hanging from a dark needle.

    Returns
    -------
    image : np.ndarray
        ``uint8`` grayscale image.
    p1, p2 : np.ndarray
        Exact contact points (where the drop meets the needle wall).
    """
    import cv2

    exact = pendant_interface(bond, needle_fraction, spacing_px=0.25, equator_radius_px=equator_radius_px)
    drop = to_image(exact, apex=apex, tilt_deg=tilt_deg)
    p1, p2 = drop[0], drop[-1]
    a = np.radians(tilt_deg)
    up = np.array([np.sin(a), -np.cos(a)])  # image direction from apex to needle
    far = 2.0 * max(shape)
    needle = np.array([p2, p2 + far * up, p1 + far * up, p1])
    h, w = shape
    ss = supersample
    mask = np.zeros((h * ss, w * ss), np.uint8)
    for poly in (drop, needle):
        fixed = np.round((poly + 0.5) * ss * 16 - 0.5 * 16).astype(np.int32)
        cv2.fillPoly(mask, [fixed], 255, lineType=cv2.LINE_8, shift=4)
    big = np.where(mask > 0, drop_gray, background_gray).astype(np.float32)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        k = int(2 * np.ceil(3 * blur_sigma) + 1)
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    img = img + np.random.default_rng(seed).normal(0.0, noise_gray, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), p1, p2
