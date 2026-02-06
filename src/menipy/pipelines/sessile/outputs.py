"""
Minimal outputs stage for the sessile pipeline.
Collects fit parameters into `ctx.results` and ensures a predictable dict shape.
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
    # Consolidate fit params if present
    if getattr(ctx, "fit", None):
        names = ctx.fit.get("param_names", [])
        params = ctx.fit.get("params", [])
        fit_map = {n: p for n, p in zip(names, params)}
        ctx.results = ctx.results or {}
        ctx.results.update(fit_map)
    # Ensure basic keys exist
    ctx.results = ctx.results or {}
    ctx.results.setdefault("surface_tension_mN_m", None)
    ctx.results.setdefault("volume_uL", None)
    return ctx
