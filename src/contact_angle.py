import math
import numpy as np


def height_width_ca(h_mm: float, base_width_mm: float) -> float:
    """Return contact angle from drop height and base width."""
    theta_rad = 2 * math.atan(h_mm / (base_width_mm / 2))
    return math.degrees(theta_rad)


def circle_fit_ca(base_radius_mm: float, circle_radius_mm: float) -> float:
    """Return contact angle from fitted circle geometry."""
    theta_rad = math.asin(base_radius_mm / circle_radius_mm)
    return math.degrees(theta_rad)


def tangent_line_ca(slope: float) -> float:
    """Return contact angle from the slope of a fitted line at the contact point."""
    theta_rad = math.atan(abs(slope))
    return math.degrees(theta_rad)


def spline_derivative_ca(dr_dz_at_base: float) -> float:
    """Return contact angle from the derivative of a spline at the contact point."""
    theta_rad = math.atan(abs(dr_dz_at_base))
    return math.degrees(theta_rad)


def adsa_ca(contour_mm: np.ndarray,
            rho_l: float, rho_g: float,
            gamma_guess: float,
            g: float = 9.80665) -> tuple[float, float]:
    """Return (theta_deg, gamma_Nm) via Young–Laplace shooting."""
    delta_rho = rho_l - rho_g
    # Placeholder for Young-Laplace shooting method
    raise NotImplementedError


# -------- Derived Silhouette Parameters -------------------------------------

def footprint_area_mm2(r_b_mm: float) -> float:
    return math.pi * r_b_mm ** 2


def drop_volume_uL(contour_mm: np.ndarray) -> float:
    r = contour_mm[:, 0] / 1000
    z = contour_mm[:, 1] / 1000
    idx = np.argsort(z)
    r, z = r[idx], z[idx]
    V_m3 = math.pi * np.trapz(r ** 2, z)
    return V_m3 * 1e9


def surface_area_mm2(contour_mm: np.ndarray) -> float:
    r = contour_mm[:, 0] / 1000
    z = contour_mm[:, 1] / 1000
    idx = np.argsort(z)
    r, z = r[idx], z[idx]
    dr_dz = np.gradient(r, z)
    integrand = r * np.sqrt(1 + dr_dz ** 2)
    A_m2 = 2 * math.pi * np.trapz(integrand, z)
    return A_m2 * 1e6


def apex_curvature_m_inv(r0_mm: float) -> float:
    return 2 / (r0_mm / 1000)


def bond_number(delta_rho: float, g: float, r_b_mm: float,
                gamma_N_m: float) -> float:
    return delta_rho * g * (r_b_mm / 1000) ** 2 / gamma_N_m


def apparent_weight_mN(volume_uL: float, delta_rho: float,
                       g: float = 9.80665) -> float:
    return delta_rho * g * (volume_uL * 1e-9) * 1e3
