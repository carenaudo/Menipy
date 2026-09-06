"""Stages.

Module implementation."""

from __future__ import annotations

import numpy as np

from menipy.common import overlay as ovl
from menipy.common import solver as common_solver
from menipy.common.plugin_loader import get_solver
from menipy.math.apex import detect_apex
from menipy.models.context import Context
from menipy.models.fit import FitConfig
from menipy.models.geometry import Geometry
from menipy.pipelines.base import PipelineBase
from menipy.pipelines.utils import ensure_contour

# Get solver from registry (loaded at startup)
young_laplace_sphere = get_solver("toy_young_laplace")
assert young_laplace_sphere is not None, "Fallback solver not found."


class CapillaryRisePipeline(PipelineBase):
    """Capillary rise: contour → baseline & meniscus apex → height gauge → toy radius fit."""

    name = "capillary_rise"

    # UI metadata for plugin-centric configuration
    ui_metadata = {
        "display_name": "Capillary Rise",
        "icon": "capillary_rise.svg",
        "color": "#9B59B6",
        "stages": [
            "acquisition",
            "contour_extraction",
            "calibration",
            "profile_fitting",
        ],
        "calibration_params": [
            "tube_diameter_mm",
            "fluid_density_kg_m3",
            "contact_angle_deg",
        ],
        "primary_metrics": [
            "capillary_height_mm",
            "surface_tension_mN_m",
            "contact_angle_deg",
        ],
    }

    def do_acquisition(self, ctx: Context) -> Context | None:
        return ctx

    def do_preprocessing(self, ctx: Context) -> Context | None:
        return ctx

    def do_geometric_features(self, ctx: Context) -> Context | None:
        """Extract baseline, apex, axis and rise height."""
        xy = ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]
        baseline_y = float(np.max(y))  # assume tube base at bottom of image
        apex_res = detect_apex(xy, mode="capillary_rise", refine=True)
        apex_xy = apex_res.point
        ctx.apex_point = (int(round(apex_xy[0])), int(round(apex_xy[1])))
        h_px = float(baseline_y - apex_xy[1])  # rise height in pixels

        # axis by median x (tube centerline)
        axis_x = float(np.median(x))

        ctx.geometry = Geometry(
            baseline_y=baseline_y,
            apex_xy=apex_xy,
            axis_x=axis_x,
        )
        ctx.h_px = h_px
        return ctx

    def do_calibration(self, ctx: Context) -> Context | None:
        """Set up pixel-to-mm scaling from tube diameter or scale settings."""
        scale = ctx.scale or {}
        px_per_mm = float(scale.get("px_per_mm", 0.0))

        tube_diameter_mm = getattr(ctx, "tube_diameter_mm", None)
        if px_per_mm <= 0 and tube_diameter_mm and tube_diameter_mm > 0:
            # Estimate tube width in pixels from contour extent
            xy = ensure_contour(ctx)
            if xy.size > 0:
                tube_width_px = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
                if tube_width_px > 5:
                    px_per_mm = tube_width_px / float(tube_diameter_mm)

        if px_per_mm <= 0:
            px_per_mm = float(getattr(ctx, "px_per_mm", 0.0) or 1.0)

        ctx.scale = {"px_per_mm": px_per_mm if px_per_mm > 0 else 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Context | None:
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Context | None:
        """Fit spherical Young-Laplace profile."""
        cfg = FitConfig(
            x0=[15.0],
            bounds=([1.0], [2000.0]),
            loss="soft_l1",
            distance="pointwise",
            param_names=["R0_mm"],
        )
        common_solver.run(ctx, integrator=young_laplace_sphere, config=cfg)
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Context | None:
        """Aggregate fit results, physical capillary rise height, and surface tension via Jurin's Law."""
        from menipy.math.jurin import (
            jurin_surface_tension,
            rayleigh_corrected_capillary_height,
        )

        fit = ctx.fit or {}
        names = fit.get("param_names") or []
        params = fit.get("params", [])
        res = dict(zip(names, params))
        h_px = float(getattr(ctx, "h_px", 0.0) or 0.0)
        res["h_px"] = h_px
        res["residuals"] = fit.get("residuals", {})

        scale = ctx.scale or {}
        px_per_mm = float(scale.get("px_per_mm", 1.0))
        h_mm = h_px / px_per_mm if px_per_mm > 0 else 0.0
        res["h_mm"] = h_mm

        # Determine tube radius
        tube_diam_mm = getattr(ctx, "tube_diameter_mm", None)
        if tube_diam_mm is not None and tube_diam_mm > 0:
            r_tube_mm = float(tube_diam_mm) / 2.0
        else:
            xy = ensure_contour(ctx)
            if xy.size > 0 and px_per_mm > 0:
                width_px = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
                r_tube_mm = (width_px / 2.0) / px_per_mm
            else:
                r_tube_mm = 0.5  # default 1 mm tube

        res["r_tube_mm"] = r_tube_mm

        # Physics
        physics = ctx.physics or {}
        rho1 = float(physics.get("rho1", 1000.0))
        rho2 = float(physics.get("rho2", 1.2))
        delta_rho = rho1 - rho2
        g = float(physics.get("g", 9.80665))

        # Contact angle
        theta_deg = float(getattr(ctx, "contact_angle_deg", 0.0) or 0.0)
        theta_rad = np.radians(theta_deg)
        res["theta_deg"] = theta_deg

        # Apply Lord Rayleigh (1915) meniscus correction
        h_m = h_mm * 1e-3
        r_m = r_tube_mm * 1e-3
        h_eff_m = rayleigh_corrected_capillary_height(h_m, r_m)
        h_eff_mm = h_eff_m * 1e3
        res["h_eff_mm"] = h_eff_mm

        # Compute surface tension via Jurin's law
        if h_eff_m > 0 and r_m > 0 and delta_rho > 0:
            gamma_n_m = jurin_surface_tension(
                h_m=h_eff_m,
                rho_kg_m3=delta_rho,
                g=g,
                tube_radius_m=r_m,
                contact_angle_rad=theta_rad,
            )
            res["gamma_mN_m"] = gamma_n_m * 1e3
            res["surface_tension_mN_m"] = res["gamma_mN_m"]

        ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Context | None:
        xy = ensure_contour(ctx)

        baseline_y = (
            int(round(ctx.geometry.baseline_y))
            if ctx.geometry and ctx.geometry.baseline_y is not None
            else 0
        )
        axis_x = (
            int(round(ctx.geometry.axis_x))
            if ctx.geometry and ctx.geometry.axis_x is not None
            else 0
        )
        apex_xy = (
            ctx.geometry.apex_xy
            if ctx.geometry and ctx.geometry.apex_xy is not None
            else (0, 0)
        )
        apex_x, apex_y = apex_xy
        h_px = float(getattr(ctx, "h_px", 0))

        text = f"R0≈{ctx.results.get('R0_mm','?')} mm | h≈{h_px:.0f}px"
        x0 = int(axis_x)
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
                "p1": (0, baseline_y),
                "p2": (int(np.max(xy[:, 0]) + 10), baseline_y),
                "color": "green",
                "thickness": 2,
            },
            {
                "type": "line",
                "p1": (x0, baseline_y),
                "p2": (x0, int(apex_y)),
                "color": "cyan",
                "thickness": 2,
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
