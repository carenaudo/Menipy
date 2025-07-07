"""Shared algorithms and constants for drop analysis."""

from __future__ import annotations



import cv2
import numpy as np


PI = 3.141592653589793


def extract_external_contour(img_roi: np.ndarray) -> np.ndarray:
    """Return the largest external contour of ``img_roi``."""
    if img_roi.ndim == 3:
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_roi

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No external contour found")

    contour = max(contours, key=cv2.contourArea).squeeze(1).astype(float)
    return contour


def _max_horizontal_diameter(contour: np.ndarray) -> tuple[int, float, int, int]:
    """Return y-position, width and edge x-coordinates of the widest slice."""
    x_min = int(np.floor(contour[:, 0].min()))
    y_min = int(np.floor(contour[:, 1].min()))
    x_max = int(np.ceil(contour[:, 0].max()))
    y_max = int(np.ceil(contour[:, 1].max()))

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = np.round(contour - [x_min, y_min]).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    left_edges = np.argmax(mask > 0, axis=1)
    right_edges = mask.shape[1] - 1 - np.argmax(mask[:, ::-1] > 0, axis=1)
    widths = right_edges - left_edges + 1
    widths[mask.sum(axis=1) == 0] = 0
    i = int(widths.argmax())
    y = y_min + i
    width = float(widths[i])
    x_left = x_min + int(left_edges[i])
    x_right = x_min + int(right_edges[i])
    return y, width, x_left, x_right


def find_apex_index(contour: np.ndarray, mode: str) -> int:
    """Return the index of the apex point."""
    if mode == "pendant":
        y_ext = contour[:, 1].max()
        candidates = np.where(contour[:, 1] == y_ext)[0]
    else:
        y_ext = contour[:, 1].min()
        candidates = np.where(contour[:, 1] == y_ext)[0]
    mid = len(candidates) // 2
    return int(candidates[mid])


def compute_drop_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    mode: str,
    needle_diam_mm: float | None = None,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> dict:
    """Return basic geometric metrics for a droplet contour."""
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    if mode not in {"pendant", "contact-angle"}:
        raise ValueError("mode must be 'pendant' or 'contact-angle'")

    x_min = float(contour[:, 0].min())
    x_max = float(contour[:, 0].max())
    y_min = float(contour[:, 1].min())
    y_max = float(contour[:, 1].max())

    y_diam, diam_px, x_left, x_right = _max_horizontal_diameter(contour)

    height_mm = (y_max - y_min) / px_per_mm
    diameter_mm = diam_px / px_per_mm

    apex_idx = find_apex_index(contour, mode)
    apex = (int(round(contour[apex_idx, 0])), int(round(contour[apex_idx, 1])))

    xi = int(np.floor(x_min))
    yi = int(np.floor(y_min))
    w = int(np.ceil(x_max) - xi + 1)
    h = int(np.ceil(y_max) - yi + 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = np.round(contour - [xi, yi]).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    from ..models import (
        contact_angle_from_mask,
        estimate_surface_tension,
        jennings_pallas_beta,
        surface_tension,
        bond_number,
        volume_from_contour,
        vmax_uL,
        worthington_number,
        apex_curvature_m_inv,
        surface_area_mm2,
        apparent_weight_mN,
    )

    px_to_mm = 1.0 / px_per_mm

    angle = contact_angle_from_mask(mask)
    ift = estimate_surface_tension(mask, 1.0, 1000.0, px_to_mm=px_to_mm)

    y_apex = float(contour[:, 1].max())
    radius_apex_mm = (y_apex - y_diam) / px_per_mm

    r0_mm = radius_apex_mm
    s1 = diameter_mm / (2.0 * r0_mm) if r0_mm > 0 else 0.0
    beta = jennings_pallas_beta(s1)
    delta_rho = 999.0
    g = 9.80665
    gamma = surface_tension(delta_rho, g, r0_mm, beta)
    bo = bond_number(delta_rho, g, r0_mm, gamma)

    contour_mm = np.column_stack(
        [np.abs(contour[:, 0] - apex[0]) / px_per_mm, (contour[:, 1] - y_min) / px_per_mm]
    )
    volume_uL = volume_from_contour(contour_mm)

    vmax = None
    wo = None
    if needle_diam_mm is not None and needle_diam_mm > 0:
        vmax = vmax_uL(gamma, needle_diam_mm, delta_rho)
        if volume_uL is not None:
            wo = worthington_number(volume_uL, vmax)

    kappa0 = apex_curvature_m_inv(r0_mm) if r0_mm > 0 else 0.0
    aproj = np.count_nonzero(mask) * (px_to_mm ** 2)

    contour_sorted = contour[np.argsort(contour[:, 1])]
    contour_local = np.column_stack([
        np.abs(contour_sorted[:, 0] - apex[0]),
        contour_sorted[:, 1] - y_min,
    ])
    asurf = surface_area_mm2(contour_local, px_per_mm)
    wapp = apparent_weight_mN(volume_uL, delta_rho) if volume_uL is not None else None

    diameter_center = (int(round((x_left + x_right) / 2)), int(round(y_diam)))
    contact_line: tuple[tuple[int, int], tuple[int, int]] | None = None
    if substrate_line is not None:
        from ..physics.contact_geom import line_params, contour_line_intersections

        a, b, c = line_params(substrate_line[0], substrate_line[1])
        try:
            left_pt, right_pt = contour_line_intersections(contour, a, b, c)
            contact_line = (
                (int(round(left_pt[0])), int(round(left_pt[1]))),
                (int(round(right_pt[0])), int(round(right_pt[1]))),
            )
        except ValueError:
            contact_line = None
    else:
        from ..models.geometry import horizontal_intersections

        xs_contact = horizontal_intersections(contour, y_min)
        if xs_contact.size >= 2:
            x_c_left = int(round(xs_contact.min()))
            x_c_right = int(round(xs_contact.max()))
            contact_line = ((x_c_left, int(round(y_min))), (x_c_right, int(round(y_min))))

    apex_to_diam_mm = abs(apex[1] - y_diam) / px_per_mm
    contact_to_diam_mm = None
    apex_to_contact_mm = None
    contact_diam_mm = None
    if contact_line is not None:
        y_contact = (contact_line[0][1] + contact_line[1][1]) / 2
        contact_to_diam_mm = abs(y_contact - y_diam) / px_per_mm
        apex_to_contact_mm = abs(apex[1] - y_contact) / px_per_mm
        contact_diam_mm = abs(contact_line[1][0] - contact_line[0][0]) / px_per_mm
        needle_area_mm2 = PI * (contact_diam_mm / 2.0) ** 2
        asurf -= needle_area_mm2
    else:
        needle_area_mm2 = None

    return {
        "height_mm": float(height_mm),
        "diameter_mm": float(diameter_mm),
        "apex": apex,
        "volume_uL": float(volume_uL) if volume_uL is not None else None,
        "contact_angle_deg": float(angle),
        "ift_mN_m": float(ift) if ift is not None else None,
        "gamma_mN_m": float(gamma * 1e3),
        "beta": float(beta),
        "s1": float(s1),
        "Bo": float(bo),
        "vmax_uL": float(vmax) if vmax is not None else None,
        "wo": float(wo) if wo is not None else None,
        "kappa0_inv_m": float(kappa0),
        "A_proj_mm2": float(aproj),
        "needle_area_mm2": float(needle_area_mm2) if needle_area_mm2 is not None else None,
        "A_surf_mm2": float(asurf),
        "W_app_mN": float(wapp) if wapp is not None else None,
        "diameter_px": float(diam_px),
        "diameter_line": ((x_left, y_diam), (x_right, y_diam)),
        "diameter_center": diameter_center,
        "radius_apex_mm": float(radius_apex_mm),
        "apex_to_diam_mm": float(apex_to_diam_mm),
        "contact_to_diam_mm": float(contact_to_diam_mm) if contact_to_diam_mm is not None else None,
        "apex_to_contact_mm": float(apex_to_contact_mm) if apex_to_contact_mm is not None else None,
        "contact_line": contact_line,
    }


__all__ = [
    "PI",
    "extract_external_contour",
    "compute_drop_metrics",
    "find_apex_index",
]

