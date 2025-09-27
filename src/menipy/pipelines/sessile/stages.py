from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

from menipy.pipelines.base import PipelineBase
from menipy.models.context import Context
from menipy.models.config import EdgeDetectionSettings
from menipy.models.fit import FitConfig
from menipy.models.geometry import Contour
from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl
from menipy.common import solver as common_solver
from menipy.common.plugins import _load_module_from_path
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
_toy_path = _repo_root / "plugins" / "toy_young_laplace.py"
_toy_mod = _load_module_from_path(_toy_path, "adsa_plugins.toy_young_laplace")
young_laplace_sphere = getattr(_toy_mod, "toy_young_laplace")


def _ensure_contour(ctx: Context) -> np.ndarray:
    if getattr(ctx, "contour", None) is not None and hasattr(ctx.contour, "xy"):
        return np.asarray(ctx.contour.xy, dtype=float)
    edged.run(ctx, settings=ctx.edge_detection_settings or EdgeDetectionSettings(method="canny"))
    return np.asarray(ctx.contour.xy, dtype=float)


class SessilePipeline(PipelineBase):
    """Sessile drop pipeline (simplified): contour → baseline/axis/angles → toy Y–L radius fit."""

    name = "sessile"

    def do_acquisition(self, ctx: Context) -> Optional[Context]:
        if ctx.image is not None:
            return ctx
        if ctx.image_path:
            try:
                img = cv2.imread(ctx.image_path, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning("Could not load image from path: %s", ctx.image_path)
                    return ctx
                ctx.image = img
            except Exception as exc:
                logger.error("Error loading image %s: %s", ctx.image_path, exc)
        else:
            logger.warning("No image or image path provided in context for acquisition stage.")
        return ctx
    def do_preprocessing(self, ctx: Context) -> Optional[Context]: return ctx

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

        from menipy.models.geometry import Geometry
        ctx.geometry = Geometry(
            axis_x=axis_x,
            baseline_y=baseline_y,
            apex_xy=apex_xy,
            tilt_deg=0.0  # these angles are in the results, not geometry
        )
        # Store contact angles in results
        if not hasattr(ctx, 'results'):
            ctx.results = {}
        ctx.results.update({
            'theta_left_deg': thetaL,
            'theta_right_deg': thetaR
        })
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
            "theta_left_deg": getattr(ctx.geometry, "theta_left_deg", None),
            "theta_right_deg": getattr(ctx.geometry, "theta_right_deg", None),
            "residuals": fit.get("residuals", {}),
        })
        ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        if not ctx.geometry:
            return ctx
        axis_x = int(round(ctx.geometry.axis_x)) if ctx.geometry.axis_x is not None else 0
        baseline_y = int(round(ctx.geometry.baseline_y)) if ctx.geometry.baseline_y is not None else 0
        apex_x, apex_y = ctx.geometry.apex_xy if ctx.geometry.apex_xy is not None else (0, 0)
        theta_l = ctx.results.get('theta_left_deg') 
        theta_r = ctx.results.get('theta_right_deg')
        text = (
            f"R0≈{ctx.results.get('R0_mm','?')} mm | "
            f"θL≈{theta_l if theta_l is not None else '?'}° "
            f"θR≈{theta_r if theta_r is not None else '?'}°"
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
