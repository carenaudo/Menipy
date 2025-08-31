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


class CapillaryRisePipeline(PipelineBase):
    """Capillary rise: contour → baseline & meniscus apex → height gauge → toy radius fit."""

    name = "capillary_rise"

    def do_acquisition(self, ctx: Context) -> Optional[Context]: return ctx
    def do_preprocessing(self, ctx: Context) -> Optional[Context]: return ctx

    def do_edge_detection(self, ctx: Context) -> Optional[Context]:
        edged.run(ctx, method="canny")
        return ctx

    def do_geometry(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]
        baseline_y = float(np.max(y))          # assume tube base at bottom of image
        apex_i = int(np.argmin(y))             # meniscus apex (highest point)
        apex_xy = (float(x[apex_i]), float(y[apex_i]))
        h_px = float(baseline_y - apex_xy[1])  # rise height in pixels

        # axis by median x (tube centerline)
        axis_x = float(np.median(x))

        ctx.geometry = {
            "baseline_y": baseline_y,
            "apex_xy": apex_xy,
            "axis_x": axis_x,
            "h_px": h_px,
        }
        return ctx

    def do_scaling(self, ctx: Context) -> Optional[Context]:
        ctx.scale = ctx.scale or {"px_per_mm": 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_solver(self, ctx: Context) -> Optional[Context]:
        # Wiring: toy spherical radius (real model would relate h to curvature & wetting)
        cfg = FitConfig(
            x0=[15.0],
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
        res["h_px"] = (ctx.geometry or {}).get("h_px")
        res["residuals"] = fit.get("residuals", {})
        ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        baseline_y = int(round(ctx.geometry["baseline_y"]))
        axis_x = int(round(ctx.geometry["axis_x"]))
        apex_x, apex_y = ctx.geometry["apex_xy"]
        h_px = float(ctx.geometry["h_px"])
        text = f"R0≈{ctx.results.get('R0_mm','?')} mm | h≈{h_px:.0f}px"
        x0 = int(axis_x)
        cmds = [
            {"type": "polyline", "points": xy.tolist(), "closed": True, "color": "yellow", "thickness": 2},
            {"type": "line", "p1": (0, baseline_y), "p2": (int(np.max(xy[:, 0]) + 10), baseline_y), "color": "green", "thickness": 2},
            {"type": "line", "p1": (x0, baseline_y), "p2": (x0, int(apex_y)), "color": "cyan", "thickness": 2},
            {"type": "cross", "p": (int(apex_x), int(apex_y)), "color": "red", "size": 6, "thickness": 2},
            {"type": "text", "p": (10, 20), "text": text, "color": "white", "scale": 0.55},
        ]
        return ovl.run(ctx, commands=cmds, alpha=0.6)

    def do_validation(self, ctx: Context) -> Optional[Context]:
        ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
        ctx.qa = {"ok": ok}
        return ctx
