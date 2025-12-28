"""
Minimal validation stage for the sessile pipeline.
Performs basic sanity checks (fit success or presence of geometry) and marks QA.
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
