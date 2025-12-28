"""
Minimal physics stage for the sessile pipeline.
Sets default physics parameters (densities and gravity) if absent.
"""
from __future__ import annotations

from typing import Optional
from menipy.models.context import Context


def run(ctx: Context) -> Optional[Context]:
    ctx.physics = ctx.physics or {}
    ctx.physics.setdefault("rho1", 1000.0)  # liquid density kg/m3
    ctx.physics.setdefault("rho2", 1.2)  # ambient gas density kg/m3
    ctx.physics.setdefault("g", 9.80665)
    return ctx
