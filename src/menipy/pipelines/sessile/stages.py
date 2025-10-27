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

    # Get image from context
    img = getattr(ctx, "frame", None)
    if img is None and hasattr(ctx, "frames") and ctx.frames:
        img = ctx.frames[0]
    if img is None:
        img = getattr(ctx, "image", None)

    # Handle Frame object
    if hasattr(img, "image"):
        img = img.image

    if img is None:
        raise RuntimeError("No image available for edge detection")

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

        # Use substrate line if provided, otherwise auto-detect
        substrate_line = getattr(ctx, 'substrate_line', None)
        auto_detect_baseline = substrate_line is None
        auto_detect_apex = True  # Always refine apex

        if substrate_line:
            # substrate_line is ((x1,y1), (x2,y2))
            p1, p2 = substrate_line
            baseline_y = float((p1[1] + p2[1]) / 2)  # approximate baseline y
            # Calculate tilt
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
        else:
            baseline_y = float(np.max(y))
            tilt_deg = 0.0

        axis_x = float(np.median(x))
        apex_i = int(np.argmin(y))
        apex_xy = (float(x[apex_i]), float(y[apex_i]))

        # Compute diameter, height, contact points using metrics logic with auto-detection
        px_per_mm = ctx.scale.get('px_per_mm', 1.0) if ctx.scale else 1.0
        # Get contact angle method from context or default to tangent
        contact_angle_method = getattr(ctx, 'contact_angle_method', 'tangent')
        if contact_angle_method not in ['tangent', 'circle_fit', 'spherical_cap']:
            contact_angle_method = 'tangent'  # Default fallback
        from .metrics import compute_sessile_metrics
        metrics = compute_sessile_metrics(
            xy,
            px_per_mm=px_per_mm,
            substrate_line=substrate_line,
            apex=apex_xy,
            contact_point_tolerance_px=20.0,  # default
            auto_detect_baseline=auto_detect_baseline,
            auto_detect_apex=auto_detect_apex,
            contact_angle_method=contact_angle_method,
        )

        from menipy.models.geometry import Geometry
        ctx.geometry = Geometry(
            axis_x=axis_x,
            baseline_y=baseline_y,
            apex_xy=apex_xy,
            tilt_deg=tilt_deg
        )

        # Store computed metrics in results
        if not hasattr(ctx, 'results'):
            ctx.results = {}
        ctx.results.update({
            'diameter_mm': metrics.get('diameter_mm', 0.0),
            'height_mm': metrics.get('height_mm', 0.0),
            'volume_uL': metrics.get('volume_uL', 0.0),
            'contact_angle_deg': metrics.get('contact_angle_deg', 0.0),  # legacy compatibility
            'theta_left_deg': metrics.get('theta_left_deg', 0.0),
            'theta_right_deg': metrics.get('theta_right_deg', 0.0),
            'contact_surface_mm2': metrics.get('contact_surface_mm2', 0.0),
            'drop_surface_mm2': metrics.get('drop_surface_mm2', 0.0),
            'baseline_tilt_deg': tilt_deg,
            'method': metrics.get('method', 'spherical_cap'),
            'uncertainty_deg': metrics.get('uncertainty_deg', {'left': 0.0, 'right': 0.0}),
            'baseline_confidence': metrics.get('baseline_confidence', 1.0),
            'apex_confidence': metrics.get('apex_confidence', 1.0),
            'baseline_method': metrics.get('baseline_method', 'manual'),
            'apex_method': metrics.get('apex_method', 'manual'),
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
            "residuals": fit.get("residuals", {}),
        })
        # Merge with existing results from geometry
        if hasattr(ctx, 'results') and ctx.results:
            ctx.results.update(res)
        else:
            ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        if ctx.image is not None and ctx.contour is not None:
            # Create a simple overlay with contour and geometry info
            overlay_img = ctx.image.copy()
            contour_xy = np.asarray(ctx.contour.xy, dtype=int)
            cv2.drawContours(overlay_img, [contour_xy], -1, (0, 255, 0), 2)

            # Draw geometry info if available
            if ctx.geometry:
                geom = ctx.geometry
                # Draw apex
                if geom.apex_xy:
                    cv2.circle(overlay_img, (int(geom.apex_xy[0]), int(geom.apex_xy[1])), 5, (255, 0, 0), -1)
                # Draw axis line
                if geom.axis_x is not None:
                    cv2.line(overlay_img, (int(geom.axis_x), 0), (int(geom.axis_x), overlay_img.shape[0]), (255, 255, 0), 1)
                # Draw baseline
                if geom.baseline_y is not None:
                    cv2.line(overlay_img, (0, int(geom.baseline_y)), (overlay_img.shape[1], int(geom.baseline_y)), (0, 255, 255), 1)

            # Add text with results
            if ctx.results:
                y_offset = 30
                for key, value in ctx.results.items():
                    if isinstance(value, (int, float)):
                        text = f"{key}: {value:.2f}"
                        cv2.putText(overlay_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 25
                        if y_offset > overlay_img.shape[0] - 50:
                            break

            ctx.preview = overlay_img
        return ctx

    def do_validation(self, ctx: Context) -> Optional[Context]:
        ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
        ctx.qa = {"ok": ok}
        return ctx
