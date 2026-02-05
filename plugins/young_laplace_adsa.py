# plugins/young_laplace_adsa.py
"""
ADSA (Axisymmetric Drop Shape Analysis) solver for pendant drops.

Integrates the Young-Laplace ODE to generate theoretical drop profiles,
enabling surface tension calculation via least-squares fitting.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from enum import Enum
import numpy as np
# NOTE: Heavy imports (scipy) moved inside for lazy load

from pydantic import BaseModel, Field, ConfigDict
from menipy.common.plugin_settings import register_detector_settings, resolve_plugin_settings


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class SolverMethod(str, Enum):
    """Available ODE solver methods."""
    RK45 = "RK45"
    RK23 = "RK23"
    DOP853 = "DOP853"
    Radau = "Radau"
    BDF = "BDF"
    LSODA = "LSODA"

class YoungLaplaceSettings(BaseModel):
    """Configuration for Young-Laplace drop profile generation."""
    model_config = ConfigDict(extra='ignore')

    s_max: float = Field(15.0, ge=1.0, description="Max normalized arc length")
    n_points: int = Field(400, ge=50, description="Number of profile points")
    solver_method: SolverMethod = Field(SolverMethod.RK45, description="ODE Solver method")
    rtol: float = Field(1e-8, description="Relative tolerance for ODE solver")
    atol: float = Field(1e-10, description="Absolute tolerance for ODE solver")


# -----------------------------------------------------------------------------
# Core Implementation
# -----------------------------------------------------------------------------

def _young_laplace_ode(s, y, beta):
    """
    Young-Laplace ODE for axisymmetric pendant drop.

    State vector y = [x, z, phi] where:
    - x: radial coordinate (normalized by R0)
    - z: vertical coordinate (normalized by R0, positive downward from apex)
    - phi: tangent angle
    """
    x, z, phi = y

    # Avoid division by zero at apex (x=0)
    if abs(x) < 1e-10:
        # At apex: limit of sin(phi)/x as x->0 is d(phi)/ds at s=0
        # Use L'Hopital: sin(phi)/x -> cos(phi) * dphi/dx -> 1 (at apex phi=0)
        dphi_ds = 2 + beta * z - 1.0  # approximation at apex
    else:
        dphi_ds = 2 + beta * z - np.sin(phi) / x

    dx_ds = np.cos(phi)
    dz_ds = np.sin(phi)

    return [dx_ds, dz_ds, dphi_ds]


def integrate_profile(
    R0_mm: float, 
    beta: float, 
    settings: YoungLaplaceSettings
) -> np.ndarray:
    """
    Integrate the Young-Laplace ODE to generate a pendant drop profile.
    """
    from scipy.integrate import solve_ivp

    # Initial conditions at apex: x=0, z=0, phi=0
    y0 = [1e-8, 0.0, 0.0]  # small x to avoid singularity

    s_span = (0.0, settings.s_max)
    s_eval = np.linspace(0.0, settings.s_max, settings.n_points)

    # Stopping condition: drop detaches when x starts decreasing significantly
    # or when the profile goes too far
    def event_detach(s, y, beta):
        # Stop if x becomes negative or angle > 170 degrees
        return y[2] - np.pi * 0.95  # phi approaching pi

    event_detach.terminal = True
    event_detach.direction = 1

    sol = solve_ivp(
        _young_laplace_ode,
        s_span,
        y0,
        args=(beta,),
        method=settings.solver_method,
        t_eval=s_eval,
        events=event_detach,
        dense_output=True,
        rtol=settings.rtol,
        atol=settings.atol,
    )

    if not sol.success and sol.t.size < 10:
        # Integration failed early, return minimal profile
        phi = np.linspace(0, np.pi, settings.n_points)
        x = R0_mm * np.sin(phi)
        z = R0_mm * (1 - np.cos(phi))
        return np.column_stack([x, z])

    # Extract profile (normalized coordinates)
    x_norm = sol.y[0]
    z_norm = sol.y[1]

    # Scale to mm
    x_mm = x_norm * R0_mm
    z_mm = z_norm * R0_mm

    # Mirror for full profile (left side)
    # Only return right half; fitting will handle symmetry
    return np.column_stack([x_mm, z_mm])


def young_laplace_adsa(
    params, 
    physics, 
    geometry,
    **kwargs
) -> np.ndarray:
    """
    Integrator function compatible with menipy's common solver.

    Args:
        params: [R0_mm, beta] - apex radius and Bond number
        physics: dict with 'rho1' (liquid), 'rho2' (ambient), 'g' (optional, default 9.80665)
        geometry: dict with apex/axis info (optional, used for alignment)
        kwargs: plugin options (will be resolved against YoungLaplaceSettings)

    Returns:
        xy: (N, 2) model profile coordinates in mm
    """
    R0_mm = float(params[0])
    beta = float(params[1])

    # Resolve settings
    # We look for "plugin_settings" in kwargs (passed by solver usually) or use kwargs directly
    plugin_settings_dict = kwargs.get("plugin_settings", {})
    # also support direct kwargs override
    
    # We must match the key used in registration ("young_laplace_adsa")
    # But since this function IS the plugin entry point, the solver might pass a specific dict key
    # For now, let's assume 'young_laplace_adsa' key in plugin_settings
    
    raw_cfg = resolve_plugin_settings("young_laplace_adsa", plugin_settings_dict, **kwargs)
    settings = YoungLaplaceSettings(**raw_cfg)

    # Generate profile
    xy = integrate_profile(R0_mm, beta, settings)

    # Shift origin to align with observed contour if geometry provides apex
    if geometry and hasattr(geometry, "apex_xy") and geometry.apex_xy:
        apex_x, apex_y = geometry.apex_xy
        # Model apex is at (0, 0), shift to match observed apex
        xy[:, 0] += apex_x  # x offset
        xy[:, 1] = apex_y - xy[:, 1]  # flip z and offset (pendant hangs down)

    return xy


def calculate_surface_tension(
    R0_mm: float, beta: float, delta_rho_kg_m3: float, g: float = 9.80665
) -> float:
    """
    Calculate surface tension from fitted parameters.

    gamma = (delta_rho * g * R0^2) / beta

    Args:
        R0_mm: Apex radius in mm
        beta: Bond number
        delta_rho_kg_m3: Density difference in kg/m³
        g: Gravitational acceleration in m/s²

    Returns:
        Surface tension in mN/m
    """
    if abs(beta) < 1e-10:
        return float("nan")

    R0_m = R0_mm / 1000.0  # convert to meters
    gamma_N_per_m = (delta_rho_kg_m3 * g * R0_m**2) / beta
    gamma_mN_per_m = gamma_N_per_m * 1000.0  # convert to mN/m

    return gamma_mN_per_m


# Register as plugin
SOLVERS = {
    "young_laplace_adsa": young_laplace_adsa,
}

# Register configuration
register_detector_settings("young_laplace_adsa", YoungLaplaceSettings)
