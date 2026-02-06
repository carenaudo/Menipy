"""
Minimal scaling stage for the pendant pipeline.
Sets a conservative default `px_per_mm` if none is present.
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
    ctx.scale = ctx.scale or {}
    ctx.scale.setdefault("px_per_mm", 1.0)
    return ctx
