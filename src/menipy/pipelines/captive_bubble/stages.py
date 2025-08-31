from __future__ import annotations

from typing import Optional
import numpy as np

from ..base import PipelineBase
from ...models.datatypes import Context, FitConfig
from ...common import edge_detection as edged
from ...common import overlay as ovl
from ...common import solver as common_solver
from plugins.toy_young_laplace import toy_young_laplace as young_laplace_sphere


def _ensure_contour(ctx: Context) -> np.ndarray:
    if getattr(ctx, "contour", None) is not None and hasattr(ctx.contour, "xy"):
        return np.asarray(ctx.contour.xy, dtype=float)
    edged.run(ctx, method="canny")
    return np.asarray(ctx.contour.xy, dtype=float)


class CaptiveBubblePipeline(PipelineBase):
    """
    Captive bubble: gas bubble pinned under a horizontal ceiling.
    Geometry (simplified):
      - ceiling_y ≈ min(y)
      - axis_x ≈ median(x)
      - 'apex' for overlay = lowest bubble point (max y)
      - cap_depth_px = max(y) - ceiling_y
    Solver: toy spherical radius just to prove wiring.
    """

    name = "captive_bubble"

    def do_acquisition(self, ctx: Context) -> Optional[Context]: return ctx
    def do_preprocessing(self, ctx: Context) -> Optional[Context]: return ctx

    def do_edge_detection(self, ctx: Context) -> Optional[Context]:
        edged.run(ctx, method="canny")
        return ctx

    def do_geometry(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]

        ceiling_y = float(np.min(y))          # ceiling at image top
        axis_x = float(np.median(x))
        bottom_i = int(np.argmax(y))          # lowest bubble point (visual apex)
        apex_xy = (float(x[bottom_i]), float(y[bottom_i]))  # for overlay cross
        cap_depth_px = float(np.max(y) - ceiling_y)

        ctx.geometry = {
            "ceiling_y": ceiling_y,
            "axis_x": axis_x,
            "apex_xy": apex_xy,
            "cap_depth_px": cap_depth_px,
        }
        return ctx

    def do_scaling(self, ctx: Context) -> Optional[Context]:
        ctx.scale = ctx.scale or {"px_per_mm": 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        # densities/gravity placeholders
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_solver(self, ctx: Context) -> Optional[Context]:
        # Toy: fit a single curvature radius
        cfg = FitConfig(
            x0=[18.0],
            bounds=([1.0], [2000.0]),
            loss="soft_l1",
            distance="pointwise",
            param_names=["R0_mm"],
        )
        common_solver.run(ctx, integrator=young_laplace_sphere, config=cfg)
        return ctx

    def do_optimization(self, ctx: Context) -> Optional[Context]: return ctx

    def do_outputs(self, ctx: Context) -> Optional[Context]:
        fit = ctx.fit or {}
        names = fit.get("param_names") or []
        params = fit.get("params", [])
        res = {n: p for n, p in zip(names, params)}
        res["cap_depth_px"] = (ctx.geometry or {}).get("cap_depth_px")
        res["residuals"] = fit.get("residuals", {})
        ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        axis_x = int(round(ctx.geometry["axis_x"]))
        ceiling_y = int(round(ctx.geometry["ceiling_y"]))
        apex_x, apex_y = ctx.geometry["apex_xy"]
        text = f"R0≈{ctx.results.get('R0_mm','?')} mm | depth≈{ctx.geometry.get('cap_depth_px',0):.0f}px"
        cmds = [
            {"type": "polyline", "points": xy.tolist(), "closed": True, "color": "yellow", "thickness": 2},
            {"type": "line", "p1": (0, ceiling_y), "p2": (int(np.max(xy[:, 0]) + 10), ceiling_y), "color": "green", "thickness": 2},
            {"type": "line", "p1": (axis_x, 0), "p2": (axis_x, int(np.max(xy[:, 1]) + 10)), "color": "cyan", "thickness": 1},
            {"type": "cross", "p": (int(apex_x), int(apex_y)), "color": "red", "size": 6, "thickness": 2},
            {"type": "text", "p": (10, 20), "text": text, "color": "white", "scale": 0.55},
        ]
        return ovl.run(ctx, commands=cmds, alpha=0.6)

    def do_validation(self, ctx: Context) -> Optional[Context]:
        ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
        ctx.qa = {"ok": ok}
        return ctx
