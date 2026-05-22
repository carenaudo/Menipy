"""
Minimal outputs stage for the pendant pipeline.
Collects fit parameters into `ctx.results` and ensures predictable keys.
"""

from __future__ import annotations

from menipy.models.context import Context


def run(ctx: Context) -> Context | None:
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
    if getattr(ctx, "fit", None):
        names = (ctx.fit or {}).get("param_names", [])
        params = (ctx.fit or {}).get("params", [])
        fit_map = dict(zip(names, params))
        ctx.results = ctx.results or {}
        ctx.results.update(fit_map)
    ctx.results = ctx.results or {}
    ctx.results.setdefault("surface_tension_mN_m", None)
    ctx.results.setdefault("volume_uL", None)
    return ctx
