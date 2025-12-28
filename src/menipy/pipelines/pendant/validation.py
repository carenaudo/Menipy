"""
Minimal validation stage for the pendant pipeline.
Basic QA: true if solver succeeded or geometry exists.
"""
from __future__ import annotations

from typing import Optional
from menipy.models.context import Context


def run(ctx: Context) -> Optional[Context]:
    ok = False
    if getattr(ctx, "fit", None) and ctx.fit.get("solver", {}).get("success", False):
        ok = True
    elif getattr(ctx, "geometry", None) is not None:
        ok = True
    ctx.qa = {"ok": ok}
    return ctx
