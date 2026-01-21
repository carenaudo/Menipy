from __future__ import annotations
from typing import Optional
import numpy as np

from menipy.pipelines.base import PipelineBase, Context, FitConfig, ovl, common_solver
from menipy.models.geometry import CaptiveBubbleGeometry
from menipy.common.plugin_loader import get_solver
from menipy.pipelines.utils import ensure_contour

# Get solver from registry (loaded at startup)
young_laplace_sphere = get_solver("toy_young_laplace")


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

    # UI metadata for plugin-centric configuration
    ui_metadata = {
        "display_name": "Captive Bubble",
        "icon": "captive_bubble.svg",
        "color": "#50E3C2",
        "stages": ["acquisition", "edge_detection", "geometry", "physics"],
        "calibration_params": [
            "needle_diameter_mm",
            "drop_density_kg_m3",
            "fluid_density_kg_m3",
        ],
        "primary_metrics": [
            "surface_tension_mN_m",
            "bubble_volume_uL",
            "pressure_difference",
        ],
    }

    def do_acquisition(self, ctx: Context) -> Optional[Context]:
        return ctx

    def do_preprocessing(self, ctx: Context) -> Optional[Context]:
        return ctx

    def do_geometry(self, ctx: Context) -> Optional[Context]:
        xy = ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]

        ceiling_y = float(np.min(y))  # ceiling at image top
        axis_x = float(np.median(x))
        bottom_i = int(np.argmax(y))  # lowest bubble point (visual apex)
        apex_xy = (float(x[bottom_i]), float(y[bottom_i]))  # for overlay cross
        cap_depth_px = float(np.max(y) - ceiling_y)

        ctx.geometry = CaptiveBubbleGeometry(
            ceiling_y=ceiling_y,
            axis_x=axis_x,
            apex_xy=apex_xy,
            cap_depth_px=cap_depth_px,
        )
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

    def do_optimization(self, ctx: Context) -> Optional[Context]:
        return ctx

    def do_outputs(self, ctx: Context) -> Optional[Context]:
        fit = ctx.fit or {}
        names = fit.get("param_names") or []
        params = fit.get("params", [])
        res = {n: p for n, p in zip(names, params)}
        # Type hint to access CaptiveBubbleGeometry specific fields
        geometry = ctx.geometry
        if not isinstance(geometry, CaptiveBubbleGeometry):
            return ctx
        res["cap_depth_px"] = (
            geometry.cap_depth_px if geometry.cap_depth_px is not None else None
        )
        res["residuals"] = fit.get("residuals", {})
        ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        # Draw measurement number if available
        if hasattr(ctx, "image") and ctx.image is not None:
            if (
                hasattr(ctx, "measurement_sequence")
                and ctx.measurement_sequence is not None
            ):
                import cv2

                img = (
                    ctx.image.copy()
                    if not hasattr(ctx, "preview") or ctx.preview is None
                    else ctx.preview.copy()
                )
                measurement_text = f"Measurement #{ctx.measurement_sequence}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    measurement_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    img, (5, 5), (15 + text_width, 15 + text_height), (0, 0, 0), -1
                )
                cv2.putText(
                    img,
                    measurement_text,
                    (10, 10 + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                ctx.preview = img

        xy = ensure_contour(ctx)
        # Type hint to access CaptiveBubbleGeometry specific fields
        geometry = ctx.geometry
        if not isinstance(geometry, CaptiveBubbleGeometry):
            return ctx
        axis_x = int(round(geometry.axis_x)) if geometry.axis_x is not None else 0
        ceiling_y = (
            int(round(geometry.ceiling_y)) if geometry.ceiling_y is not None else 0
        )
        apex_xy = geometry.apex_xy if geometry.apex_xy is not None else (0, 0)
        apex_x, apex_y = apex_xy
        cap_depth = geometry.cap_depth_px if geometry.cap_depth_px is not None else 0
        text = f"R0≈{ctx.results.get('R0_mm','?')} mm | depth≈{cap_depth:.0f}px"
        cmds = [
            {
                "type": "polyline",
                "points": xy.tolist(),
                "closed": True,
                "color": "yellow",
                "thickness": 2,
            },
            {
                "type": "line",
                "p1": (0, ceiling_y),
                "p2": (int(np.max(xy[:, 0]) + 10), ceiling_y),
                "color": "green",
                "thickness": 2,
            },
            {
                "type": "line",
                "p1": (axis_x, 0),
                "p2": (axis_x, int(np.max(xy[:, 1]) + 10)),
                "color": "cyan",
                "thickness": 1,
            },
            {
                "type": "cross",
                "p": (int(apex_x), int(apex_y)),
                "color": "red",
                "size": 6,
                "thickness": 2,
            },
            {
                "type": "text",
                "p": (10, 20),
                "text": text,
                "color": "white",
                "scale": 0.55,
            },
        ]
        return ovl.run(ctx, commands=cmds, alpha=0.6)

    def do_validation(self, ctx: Context) -> Optional[Context]:
        ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
        ctx.qa = {"ok": ok}
        return ctx
