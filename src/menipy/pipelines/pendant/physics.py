"""
Minimal physics stage for the pendant pipeline.
Populates default densities and gravity if missing.
"""
from __future__ import annotations

from typing import Optional
from menipy.models.context import Context


def run(ctx: Context) -> Optional[Context]:
    """Run.

    Parameters
    ----------
    ctx : type
        Description.

    Returns
    -------
    type
        Description.
    """
    ctx.physics = ctx.physics or {}
    ctx.physics.setdefault("rho1", 1000.0)
    ctx.physics.setdefault("rho2", 1.2)
    ctx.physics.setdefault("g", 9.80665)
    return ctx
