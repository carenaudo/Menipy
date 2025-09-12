from __future__ import annotations

from typing import Optional
import numpy as np

from menipy.pipelines.base import PipelineBase
from menipy.models.datatypes import Context
from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl
from menipy.common import solver as common_solver
# Load the optional plugin implementation dynamically via the project's
# plugin utilities. This avoids hard import-time dependence on a top-level
# `plugins` package being on sys.path.
from pathlib import Path
from menipy.common.plugins import _load_module_from_path

# plugins/ is at the repository root relative to this file. parents[4] resolves
# to the project root (D:/programacion/Menipy) for the typical layout.
_repo_root = Path(__file__).resolve().parents[4]
_toy_path = _repo_root / "plugins" / "toy_young_laplace.py"
_toy_mod = _load_module_from_path(_toy_path, "adsa_plugins.toy_young_laplace")
young_laplace_sphere = getattr(_toy_mod, "toy_young_laplace")
from menipy.models.datatypes import FitConfig


def _ensure_contour(ctx: Context) -> np.ndarray:
    if getattr(ctx, "contour", None) is not None and hasattr(ctx.contour, "xy"):
        return np.asarray(ctx.contour.xy, dtype=float)
    edged.run(ctx, method="canny")
    return np.asarray(ctx.contour.xy, dtype=float)


class SessilePipeline(PipelineBase):
    """Sessile drop pipeline (simplified): contour → baseline/axis/angles → toy Y–L radius fit."""

    name = "sessile"

    def do_acquisition(self, ctx: Context) -> Optional[Context]: return ctx
    def do_preprocessing(self, ctx: Context) -> Optional[Context]: return ctx

    def do_edge_detection(self, ctx: Context) -> Optional[Context]:
        edged.run(ctx, method="canny")
        return ctx

    def do_geometry(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]

        baseline_y = float(np.max(y))
        axis_x = float(np.median(x))
        apex_i = int(np.argmin(y))
        apex_xy = (float(x[apex_i]), float(y[apex_i]))

        # crude contact angles (placeholder)
        def slope_at(i: int) -> float:
            j0 = max(0, i - 1); j1 = min(len(x) - 1, i + 1)
            dx = x[j1] - x[j0]; dy = y[j1] - y[j0]
            return float(dy / (dx + 1e-9))

        left_i = int(np.argmin(x + 10 * (baseline_y - y)))
        right_i = int(np.argmin(-x + 10 * (baseline_y - y)))
        mL, mR = slope_at(left_i), slope_at(right_i)
        thetaL = float(np.degrees(np.arctan2(-mL, 1.0)))
        thetaR = float(np.degrees(np.arctan2(mR, 1.0)))

        ctx.geometry = {
            "axis_x": axis_x,
            "baseline_y": baseline_y,
            "apex_xy": apex_xy,
            "theta_left_deg": thetaL,
            "theta_right_deg": thetaR,
        }
        return ctx

    def do_scaling(self, ctx: Context) -> Optional[Context]:
        ctx.scale = ctx.scale or {"px_per_mm": 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_solver(self, ctx: Context) -> Optional[Context]:
        cfg = FitConfig(
            x0=[20.0],
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
        res.update({
            "theta_left_deg": ctx.geometry.get("theta_left_deg"),
            "theta_right_deg": ctx.geometry.get("theta_right_deg"),
            "residuals": fit.get("residuals", {}),
        })
        ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        axis_x = int(round(ctx.geometry["axis_x"]))
        baseline_y = int(round(ctx.geometry["baseline_y"]))
        apex_x, apex_y = ctx.geometry["apex_xy"]
        text = (
            f"R0≈{ctx.results.get('R0_mm','?')} mm | "
            f"θL≈{ctx.results.get('theta_left_deg','?'):.0f}° "
            f"θR≈{ctx.results.get('theta_right_deg','?'):.0f}°"
        )
        cmds = [
            {"type": "polyline", "points": xy.tolist(), "closed": True, "color": "yellow", "thickness": 2},
            {"type": "line", "p1": (axis_x, 0), "p2": (axis_x, int(np.max(xy[:, 1]) + 10)), "color": "cyan", "thickness": 1},
            {"type": "line", "p1": (0, baseline_y), "p2": (int(np.max(xy[:, 0]) + 10), baseline_y), "color": "green", "thickness": 2},
            {"type": "cross", "p": (int(apex_x), int(apex_y)), "color": "red", "size": 6, "thickness": 2},
            {"type": "text", "p": (10, 20), "text": text, "color": "white", "scale": 0.55},
        ]
        return ovl.run(ctx, commands=cmds, alpha=0.6)

    def do_validation(self, ctx: Context) -> Optional[Context]:
        ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
        ctx.qa = {"ok": ok}
        return ctx
