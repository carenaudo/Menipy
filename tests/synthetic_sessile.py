"""Exact synthetic sessile profiles for arc-spline tests, profiling and benchmarks.

Profiles come from an independent adaptive ``solve_ivp`` integration of the
dimensionless Young-Laplace equation, not from the RK4 table used by
:mod:`menipy.math.sessile_box`, so the physics prior is never checked against
itself.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

CONTACT_RADIUS_PX = 300.0


@lru_cache(maxsize=64)
def half_profile(bond: float, theta_deg: float, contact_radius_px: float = CONTACT_RADIUS_PX) -> np.ndarray:
    """Return a dense right half-profile, apex at the origin, y towards the substrate.

    Parameters
    ----------
    bond : float
        Bond number based on the apex radius.
    theta_deg : float
        Contact angle; integration stops where the tangent reaches it.
    contact_radius_px : float, optional
        Scale: the contact point lands at this x.

    Returns
    -------
    np.ndarray
        Points from apex to contact, spacing about 0.05 px, shape ``(M, 2)``.
    """
    theta = np.radians(theta_deg)

    def rhs(_s, y):
        x, z, phi = y
        return [np.cos(phi), np.sin(phi), 2.0 + bond * z - np.sin(phi) / x]

    def reach(_s, y):
        return y[2] - theta

    reach.terminal = True
    s0 = 1e-4
    sol = solve_ivp(rhs, (s0, 50.0), [s0, s0**2 / 2, s0], events=reach,
                    max_step=1e-3, rtol=1e-10, atol=1e-12)
    xy = np.column_stack([sol.y[0], sol.y[1]]) * (contact_radius_px / sol.y[0][-1])
    steps = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    grid = np.append(np.arange(0.0, s[-1], 0.05), s[-1])
    return np.column_stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])])


def full_profile(
    bond: float, theta_deg: float, spacing_px: float = 1.0, contact_radius_px: float = CONTACT_RADIUS_PX
) -> np.ndarray:
    """Return the symmetric P1 -> P2 interface at about ``spacing_px`` spacing."""
    half = half_profile(bond, theta_deg, contact_radius_px)
    steps = np.hypot(*np.diff(half, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    grid = np.append(np.arange(0.0, s[-1], spacing_px), s[-1])
    right = np.column_stack([np.interp(grid, s, half[:, 0]), np.interp(grid, s, half[:, 1])])
    return np.vstack([right[::-1] * [-1.0, 1.0], right[1:]])


def degrade(points: np.ndarray, noise: str, rng: np.random.Generator) -> np.ndarray:
    """Apply an edge-noise model: ``clean``, ``quantized`` or ``gauss<sigma>``."""
    if noise == "clean":
        return points.copy()
    if noise == "quantized":
        q = np.round(points + rng.uniform(-0.5, 0.5, size=2))
        keep = np.concatenate([[True], np.any(np.diff(q, axis=0) != 0, axis=1)])
        return q[keep]
    sigma = float(noise.removeprefix("gauss"))
    d = np.gradient(points, axis=0)
    normal = np.column_stack([-d[:, 1], d[:, 0]]) / np.hypot(*d.T)[:, None]
    return points + rng.normal(0.0, sigma, len(points))[:, None] * normal


def place(points: np.ndarray, base_y: float, tilt_deg: float = 0.0, origin=(320.0, 400.0)) -> np.ndarray:
    """Put the baseline ``y = base_y`` at ``origin``, rotated by ``tilt_deg``."""
    a = np.radians(tilt_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    centered = points - np.array([0.0, base_y])
    return centered @ rot.T + np.asarray(origin, float)


def sessile_case(bond: float, theta_deg: float, noise: str = "clean", *, seed: int = 0,
                 tilt_deg: float = 0.0, contact_radius_px: float = CONTACT_RADIUS_PX):
    """Return ``(interface_points, p1, p2)`` in image coordinates.

    The profile has its apex at ``y = 0`` and contacts at ``y = H``, so with image
    y growing downward the drop already sits above its baseline; ``tilt_deg``
    rotates the whole scene.
    """
    exact = full_profile(bond, theta_deg, contact_radius_px=contact_radius_px)
    base_y = float(exact[0, 1])
    pts = place(degrade(exact, noise, np.random.default_rng(seed)), base_y, tilt_deg)
    p1, p2 = place(exact[[0, -1]], base_y, tilt_deg)
    return pts, p1, p2


def render_image(
    interface: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    shape: tuple[int, int] = (480, 640),
    drop_gray: float = 30.0,
    background_gray: float = 220.0,
    substrate_gray: float = 190.0,
    blur_sigma: float = 1.2,
    noise_gray: float = 2.0,
    seed: int = 0,
    supersample: int = 4,
) -> np.ndarray:
    """Render an anti-aliased grayscale image of a dark drop on a substrate.

    The drop polygon is the exact interface closed along the contact chord;
    everything beyond the chord (away from the apex) is substrate. The edge is
    anti-aliased by supersampling, then blurred and given Gaussian noise.

    Returns
    -------
    np.ndarray
        ``uint8`` image of ``shape``.
    """
    import cv2

    h, w = shape
    ss = supersample
    big = np.full((h * ss, w * ss), background_gray, np.float32)
    chord = p2 - p1
    normal = np.array([-chord[1], chord[0]]) / np.hypot(*chord)
    apex_side = np.sign((interface[len(interface) // 2] - p1) @ normal)
    yy, xx = np.mgrid[0 : h * ss, 0 : w * ss]
    centers = np.stack([(xx + 0.5) / ss - 0.5, (yy + 0.5) / ss - 0.5], -1)
    below = ((centers - p1) @ normal) * apex_side < 0
    big[below] = substrate_gray
    poly = np.vstack([interface, p2, p1])
    fixed = np.round((poly + 0.5) * ss * 16 - 0.5 * 16).astype(np.int32)
    mask = np.zeros_like(big, np.uint8)
    cv2.fillPoly(mask, [fixed], 255, lineType=cv2.LINE_8, shift=4)
    big[mask > 0] = drop_gray
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        k = int(2 * np.ceil(3 * blur_sigma) + 1)
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    img = img + np.random.default_rng(seed).normal(0.0, noise_gray, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def mask_contour(image: np.ndarray) -> np.ndarray:
    """Largest external contour of an Otsu-thresholded dark object (binary mask)."""
    import cv2

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def closed_silhouette(points: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Close an interface with a straight run along the contact chord."""
    n = max(int(np.hypot(*(p2 - p1))), 2)
    t = np.linspace(1.0, 0.0, n)[1:-1, None]
    closure = p1 + t * (p2 - p1)
    return np.vstack([points, closure])
