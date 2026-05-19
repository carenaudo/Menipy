"""Stages.

Module implementation."""


from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

from menipy.pipelines.base import PipelineBase
from menipy.models.context import Context
from menipy.models.fit import FitConfig
from menipy.common import solver as common_solver
from menipy.common import overlay as ovl
from menipy.common import registry
from menipy.common.plugin_loader import get_solver
from menipy.pipelines.utils import ensure_contour

from menipy.math.young_laplace import young_laplace_ode

# Get solver from registry (loaded at startup)
young_laplace_default = get_solver("young_laplace_ode", fallback=young_laplace_ode)


def _contour_to_xy(contour: object) -> np.ndarray:
    """Normalize OpenCV and plain contours to an ``(N, 2)`` float array."""
    xy = np.asarray(contour, dtype=float)
    if xy.ndim == 3 and xy.shape[-1] == 2:
        xy = xy.reshape(-1, 2)
    elif xy.ndim == 2 and xy.shape[1] >= 2:
        xy = xy[:, :2]
    else:
        xy = xy.reshape(-1, 2)
    return np.asarray(xy, dtype=float)


class SessilePipeline(PipelineBase):
    """Sessile drop pipeline (simplified): contour → baseline/axis/angles → toy Y–L radius fit."""

    name = "sessile"
    solver_name: str = "young_laplace_ode"
    preprocessor_name: str | None = None
    edge_detector_name: str | None = None

    # UI metadata for plugin-centric configuration
    ui_metadata = {
        "display_name": "Sessile Drop",
        "icon": "sessile.svg",
        "color": "#4A90E2",
        "stages": ["acquisition", "contour_extraction", "contour_refinement", "geometric_features", "overlay", "physics"],
        "calibration_params": [
            "needle_diameter_mm",
            "drop_density_kg_m3",
            "fluid_density_kg_m3",
            "substrate_contact_angle_deg",
        ],
        "primary_metrics": ["contact_angle_deg", "surface_tension_mN_m", "volume_uL"],
    }

    def do_acquisition(self, ctx: Context) -> Optional[Context]:
        """Load frames from disk or wrap existing image in frames."""
        from menipy.common.acquisition_stage import do_acquisition
        return do_acquisition(ctx, self.logger)

    def do_preprocessing(self, ctx: Context) -> Optional[Context]:
        """Run preprocessing with automatic feature detection."""
        if self.preprocessor_name and self.preprocessor_name in registry.PREPROCESSORS:
            fn = registry.PREPROCESSORS[self.preprocessor_name]
            return fn(ctx) or ctx
        from menipy.pipelines.sessile.preprocessing import do_preprocessing
        return do_preprocessing(ctx)

    def do_contour_extraction(self, ctx: Context) -> Optional[Context]:
        """Extract droplet contour using edge detection."""
        # Check if drop_contour is already provided (e.g. from calibration wizard)
        drop_contour = getattr(ctx, "drop_contour", None)
        if drop_contour is not None:
            from menipy.models.geometry import Contour
            if isinstance(drop_contour, Contour):
                ctx.contour = Contour(
                    xy=_contour_to_xy(drop_contour.xy),
                    closed=drop_contour.closed,
                    units=drop_contour.units,
                    smoothing=drop_contour.smoothing,
                    origin_hint=drop_contour.origin_hint,
                )
            else:
                ctx.contour = Contour(xy=_contour_to_xy(drop_contour))
            
            logger.info("Using pre-detected drop contour from calibration")
            return ctx

        from menipy.common import edge_detection
        from menipy.models.config import EdgeDetectionSettings
        
        # Use custom edge detector if specified
        if self.edge_detector_name:
            settings = self.edge_detection_settings or EdgeDetectionSettings()
            # Override the method with the specified detector name
            settings = EdgeDetectionSettings(
                **{**settings.__dict__, "method": self.edge_detector_name}
            )
            return edge_detection.run(ctx, settings)
        return super().do_contour_extraction(ctx)

    def do_contour_refinement(self, ctx: Context) -> Optional[Context]:
        """Clip contour at substrate line, refine contact points, and optionally smooth."""
        # If using pre-detected contour, skip clipping to preserve the closed polygon
        if getattr(ctx, "drop_contour", None) is not None:
            # Still apply smoothing if enabled
            smoothing_settings = getattr(ctx, 'contour_smoothing_settings', None)
            if smoothing_settings and smoothing_settings.enabled:
                from menipy.common import contour_smoothing
                ctx = contour_smoothing.run(ctx, smoothing_settings)
            return ctx
            
        xy = ensure_contour(ctx)
        
        substrate_line = getattr(ctx, "substrate_line", None)
        if not substrate_line:
            return ctx
            
        # Get apex for reference
        x, y = xy[:, 0], xy[:, 1]
        if getattr(ctx, "apex_point", None) is not None:
            apex_xy = (
                float(ctx.apex_point[0]),
                float(ctx.apex_point[1]),
            )
        else:
            apex_i = int(np.argmin(y))
            apex_xy = (float(x[apex_i]), float(y[apex_i]))
        
        from .geometry import clip_contour_to_substrate
        
        xy, refined_contact_points = clip_contour_to_substrate(xy, substrate_line, apex_xy)
        
        # Update contour in context
        from menipy.models.geometry import Contour
        ctx.contour = Contour(xy=xy)
        
        # Store refined contact points for use in geometric_features
        if refined_contact_points:
            ctx.contact_points = refined_contact_points
        
        # Apply optional contour smoothing
        smoothing_settings = getattr(ctx, 'contour_smoothing_settings', None)
        if smoothing_settings and smoothing_settings.enabled:
            from menipy.common import contour_smoothing
            ctx = contour_smoothing.run(ctx, smoothing_settings)
            
        return ctx

    def do_geometric_features(self, ctx: Context) -> Optional[Context]:
        """Extract geometric features: axis, apex, baseline, tilt, and angles."""
        xy = ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]

        # Use substrate line if provided, otherwise auto-detect
        substrate_line = getattr(ctx, "substrate_line", None)
        auto_detect_baseline = substrate_line is None
        auto_detect_apex = True  # Always refine apex

        if substrate_line:
            p1, p2 = substrate_line
            baseline_y = float((p1[1] + p2[1]) / 2)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
        else:
            baseline_y = float(np.max(y))
            tilt_deg = 0.0

        axis_x = float(np.median(x))
        if getattr(ctx, "apex_point", None) is not None:
            apex_xy = (
                float(ctx.apex_point[0]),
                float(ctx.apex_point[1]),
            )
        else:
            apex_i = int(np.argmin(y))
            apex_xy = (float(x[apex_i]), float(y[apex_i]))

        # Get scale
        if ctx.scale and ctx.scale.get("px_per_mm"):
            px_per_mm = ctx.scale.get("px_per_mm", 1.0)
        elif hasattr(ctx, "px_per_mm") and ctx.px_per_mm:
            px_per_mm = ctx.px_per_mm
        else:
            px_per_mm = 1.0
            
        # Get contact angle method
        contact_angle_method = getattr(ctx, "contact_angle_method", "tangent")
        if contact_angle_method not in ["tangent", "circle_fit", "spherical_cap"]:
            contact_angle_method = "tangent"
        
        # Get contact points (may have been set by contour_refinement)
        use_contact_points = getattr(ctx, "contact_points", None)
        
        from .metrics import compute_sessile_metrics

        # Compute metrics (will be stored in ctx.results by compute_metrics stage)
        metrics = compute_sessile_metrics(
            xy,
            px_per_mm=px_per_mm,
            substrate_line=substrate_line,
            apex=apex_xy,
            contact_point_tolerance_px=20.0,
            auto_detect_baseline=auto_detect_baseline,
            auto_detect_apex=auto_detect_apex,
            contact_angle_method=contact_angle_method,
            contact_points=use_contact_points,
        )
        
        # Store metrics temporarily for compute_metrics stage
        ctx._sessile_metrics = metrics

        from menipy.models.geometry import Geometry

        ctx.geometry = Geometry(
            axis_x=axis_x, baseline_y=baseline_y, apex_xy=apex_xy, tilt_deg=tilt_deg
        )
        return ctx

    def do_calibration(self, ctx: Context) -> Optional[Context]:
        """Set up pixel-to-mm scaling."""
        ctx.scale = ctx.scale or {"px_per_mm": 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Optional[Context]:
        """Fit spherical Young-Laplace profile."""
        integrator = get_solver(self.solver_name, fallback=young_laplace_default)
        assert integrator is not None, f"Solver {self.solver_name} not found and no fallback available."
        
        # New ODE solver requires [R0_mm, beta]. beta ~ 0.5 for a typical drop
        cfg = FitConfig(
            x0=[20.0, 0.1],
            bounds=([1.0, -10.0], [2000.0, 10.0]),
            loss="soft_l1",
            distance="pointwise",
            param_names=["R0_mm", "beta"],
        )
        
        common_solver.run(ctx, integrator=integrator, config=cfg)
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Optional[Context]:
        """Aggregate fit results and sessile metrics."""
        fit = ctx.fit or {}
        names = fit.get("param_names") or []
        params = fit.get("params", [])
        res = {f"fit_{n}": p for n, p in zip(names, params)}
        res.update({"residuals": fit.get("residuals", {})})

        residuals = fit.get("residuals") or {}
        try:
            rmse = float(residuals.get("rmse"))
        except (TypeError, ValueError):
            rmse = float("nan")
        if np.isfinite(rmse) and rmse > 25.0:
            res["fit_warning"] = "profile_fit_unreliable"
        
        # Merge sessile metrics from geometric_features stage
        if hasattr(ctx, "_sessile_metrics") and ctx._sessile_metrics:
            res.update(ctx._sessile_metrics)
        
        # Merge with existing results
        if hasattr(ctx, "results") and ctx.results:
            ctx.results.update(res)
        else:
            ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = ensure_contour(ctx)
        
        cmds = []
        if len(xy) > 0:
            cmds.append({
                "type": "polyline",
                "points": xy.tolist(),
                "closed": False,
                "color": "green",
                "thickness": 2
            })
            
        if hasattr(ctx, "measurement_sequence") and ctx.measurement_sequence is not None:
            cmds.append({
                "type": "text", 
                "p": (10, 15), 
                "text": f"Measurement #{ctx.measurement_sequence}",
                "color": "white",
                "scale": 0.7,
                "thickness": 2
            })
            
        if ctx.geometry:
            geom = ctx.geometry
            if geom.apex_xy:
                cmds.append({
                    "type": "circle",
                    "center": geom.apex_xy,
                    "radius": 5,
                    "color": "blue",
                    "thickness": -1
                })
            if geom.axis_x is not None:
                # Need an arbitrary Y for a vertical line (overlay handles lines in px space)
                shape = (1000, 1000)
                if ctx.image is not None and hasattr(ctx.image, "shape"):
                    shape = ctx.image.shape
                cmds.append({
                    "type": "line",
                    "p1": (geom.axis_x, 0),
                    "p2": (geom.axis_x, shape[0]),
                    "color": "cyan",
                    "thickness": 1
                })
            if geom.baseline_y is not None:
                shape = (1000, 1000)
                if ctx.image is not None and hasattr(ctx.image, "shape"):
                    shape = ctx.image.shape
                cmds.append({
                    "type": "line",
                    "p1": (0, geom.baseline_y),
                    "p2": (shape[1], geom.baseline_y),
                    "color": "yellow",
                    "thickness": 1
                })
                
        if ctx.results:
            y_offset = 30
            for key in ["diameter_mm", "height_mm", "contact_angle_deg", "volume_uL"]:
                if key in ctx.results:
                    val = ctx.results[key]
                    if isinstance(val, (int, float)):
                        cmds.append({
                            "type": "text", 
                            "p": (10, y_offset), 
                            "text": f"{key.replace('_', ' ').title()}: {val:.2f}",
                            "color": "white",
                            "scale": 0.6,
                            "thickness": 2
                        })
                        y_offset += 25
                        
        return ovl.run(ctx, commands=cmds, alpha=0.6)
