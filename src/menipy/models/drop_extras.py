"""
Additional droplet property calculations (Worthington number, curvature, surface area, etc.).
"""

import numpy as np
import math


# ---------- 1. Worthington number ---------------------------------
def vmax_uL(
    gamma_N_m: float, needle_diam_mm: float, delta_rho: float, g: float = 9.80665
) -> float:
    """Max detachment volume (µL) from Berry et al. (2015)."""
    D = needle_diam_mm / 1000.0
    vmax_m3 = math.pi * gamma_N_m * D / (delta_rho * g)
    return vmax_m3 * 1e9


def worthington_number(vol_uL: float, vmax_uL: float) -> float:
    """Worthington number Wo = V / Vmax."""
    return vol_uL / vmax_uL


# ---------- 2. Local curvature at apex ----------------------------
def apex_curvature_m_inv(r0_mm: float) -> float:
    """Return mean curvature κ₀ at the apex (m⁻¹)."""
    r0_m = r0_mm / 1000.0
    return 2.0 / r0_m


# ---------- 3. Projected cross-sectional area ---------------------
def projected_area_mm2(De_mm: float) -> float:
    """Projected drop area in mm²."""
    return math.pi * (De_mm / 2.0) ** 2


# ---------- 4. True surface area by revolution --------------------
def surface_area_mm2(contour_px: np.ndarray, px_per_mm: float) -> float:
    """Surface area of a revolved contour (mm²)."""
    r_mm = contour_px[:, 0] / px_per_mm
    z_mm = contour_px[:, 1] / px_per_mm
    order = np.argsort(z_mm)
    r_mm, z_mm = r_mm[order], z_mm[order]
    z_mm, unique_idx = np.unique(z_mm, return_index=True)
    r_mm = r_mm[unique_idx]
    dr_dz = np.gradient(r_mm, z_mm, edge_order=2)
    integrand = r_mm * np.sqrt(1.0 + dr_dz**2)
    A_mm2 = 2.0 * math.pi * np.trapz(integrand, z_mm)
    return A_mm2


# ---------- 5. Apparent weight ------------------------------------
def apparent_weight_mN(vol_uL: float, delta_rho: float, g: float = 9.80665) -> float:
    """Apparent drop weight in milli-Newton."""
    V_m3 = vol_uL * 1e-9
    return delta_rho * g * V_m3 * 1e3
