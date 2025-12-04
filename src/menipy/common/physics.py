# src/menipy/common/physics.py
"""
Physics constants and parameter management.
"""
from __future__ import annotations

def run(ctx, **overrides):
    """Collect known physical constants; pipelines can override/extend."""
    p = {
        "delta_rho": 1000.0,   # kg/m^3
        "g": 9.80665,          # m/s^2
        "surface_tension_guess": 50.0,  # mN/m
        "needle_radius_mm": None,
        "tube_radius_mm": None,
    }
    p.update(overrides)
    ctx.physics = p
    return ctx
