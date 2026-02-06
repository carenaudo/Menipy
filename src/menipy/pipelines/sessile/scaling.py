"""
Minimal scaling stage for the sessile pipeline.
Sets `ctx.scale['px_per_mm']` if absent using a best-effort heuristic.
"""
from __future__ import annotations

from menipy.models.context import Context
from typing import Optional


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
    # If a scale is already defined, keep it.
    ctx.scale = ctx.scale or {}
    if "px_per_mm" not in ctx.scale:
        # Default conservative value (1 px == 1 mm) so code path remains safe.
        ctx.scale["px_per_mm"] = 1.0
    return ctx
