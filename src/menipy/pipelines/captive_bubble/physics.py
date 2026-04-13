"""Physics functions for captive bubble."""
from __future__ import annotations

from typing import Optional

def compute_physics(
    physics_config: dict,
    r0_mm: float,
    beta: float
) -> tuple[Optional[float], Optional[float]]:
    """
    Computes gamma and capillary length from a captive bubble fit.
    Captive bubble fits are inverted sessile drops mathematically.
    
    Returns:
        tuple containing (surface_tension_mN_m, capillary_length_mm)
        If beta is invalid or negative, returns (None, None).
    """
    d_rho = abs(physics_config.get("rho1", 1000.0) - physics_config.get("rho2", 1.2))
    g = physics_config.get("g", 9.80665)
    
    gamma_mN_m = None
    capillary_length_mm = None
    
    if r0_mm is not None and beta is not None and beta > 1e-6:
        r0_m = r0_mm / 1000.0
        # beta = (d_rho * g * r0_m**2) / gamma
        gamma_N_m = (d_rho * g * r0_m**2) / beta
        gamma_mN_m = gamma_N_m * 1000.0
        if gamma_N_m > 0:
            import math
            cl_m = math.sqrt(gamma_N_m / (d_rho * g))
            capillary_length_mm = cl_m * 1000.0
            
    return gamma_mN_m, capillary_length_mm
