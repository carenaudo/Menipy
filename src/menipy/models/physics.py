"""Physical modeling utilities."""

import numpy as np
from scipy.integrate import solve_ivp


def young_laplace_ode(r, y, gamma, delta_rho, g):
    """Example ODE for the Young-Laplace equation (simplified)."""
    z, phi = y
    dzdr = np.tan(phi)
    dphidr = (delta_rho * g * r / gamma) - (np.sin(phi) / r)
    return [dzdr, dphidr]


def solve_young_laplace(r_span, y0, gamma, delta_rho, g=9.81):
    """Integrate the Young-Laplace ODE."""
    sol = solve_ivp(
        young_laplace_ode, r_span, y0, args=(gamma, delta_rho, g), dense_output=True
    )
    return sol
