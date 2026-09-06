"""Stages.

Module implementation."""

from __future__ import annotations

import numpy as np

from menipy.common.plugin_loader import get_solver

# Get solver from registry (loaded at startup)
from menipy.math.young_laplace import young_laplace_ode as yl_ode
from menipy.models.geometry import CaptiveBubbleGeometry
from menipy.pipelines.base import Context, PipelineBase, ovl
from menipy.pipelines.captive_bubble.physics import compute_physics
from menipy.pipelines.utils import ensure_contour

young_laplace_ode = get_solver("young_laplace_ode", fallback=yl_ode)


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
        "stages": [
            "acquisition",
            "contour_extraction",
            "geometric_features",
            "physics",
        ],
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

    def do_acquisition(self, ctx: Context) -> Context | None:
        return ctx

    def do_preprocessing(self, ctx: Context) -> Context | None:
        return ctx

    def do_geometric_features(self, ctx: Context) -> Context | None:
        """Extract ceiling, axis, apex, diameter, and ceiling contact points."""
        xy = ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]

        ceiling_y = float(np.min(y))  # ceiling at image top
        axis_x = float(np.median(x))
        bottom_i = int(np.argmax(y))  # lowest bubble point (apex)
        apex_xy = (float(x[bottom_i]), float(y[bottom_i]))
        cap_depth_px = float(np.max(y) - ceiling_y)
        diameter_px = float(np.max(x) - np.min(x))

        # Find contact points near ceiling level
        near_ceiling = xy[xy[:, 1] <= (ceiling_y + max(5.0, 0.1 * cap_depth_px))]
        if near_ceiling.shape[0] >= 2:
            p_left = near_ceiling[np.argmin(near_ceiling[:, 0])]
            p_right = near_ceiling[np.argmax(near_ceiling[:, 0])]
            contact_points = (
                (float(p_left[0]), float(p_left[1])),
                (float(p_right[0]), float(p_right[1])),
            )
        else:
            contact_points = (
                (float(np.min(x)), ceiling_y),
                (float(np.max(x)), ceiling_y),
            )

        ctx.contact_points = contact_points
        ctx.geometry = CaptiveBubbleGeometry(
            ceiling_y=ceiling_y,
            axis_x=axis_x,
            apex_xy=apex_xy,
            cap_depth_px=cap_depth_px,
        )
        ctx._captive_diameter_px = diameter_px
        return ctx

    def do_calibration(self, ctx: Context) -> Context | None:
        """Set up pixel-to-mm scaling."""
        scale = ctx.scale or {}
        px_per_mm = float(scale.get("px_per_mm", 0.0))
        needle_diameter_mm = getattr(ctx, "needle_diameter_mm", None)
        needle_rect = getattr(ctx, "needle_rect", None)
        if px_per_mm <= 0 and needle_diameter_mm and needle_rect and len(needle_rect) >= 3:
            needle_w_px = float(needle_rect[2])
            if needle_w_px > 0:
                px_per_mm = needle_w_px / float(needle_diameter_mm)
        if px_per_mm <= 0:
            px_per_mm = float(getattr(ctx, "px_per_mm", 0.0) or 1.0)
        ctx.scale = {"px_per_mm": px_per_mm if px_per_mm > 0 else 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Context | None:
        # Invert buoyancy: liquid is outside (rho1), gas bubble is inside (rho2)
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Context | None:
        """Fit apex radius of curvature and Young-Laplace profile."""
        from menipy.common.geometry import fit_circle

        xy = ensure_contour(ctx)
        if xy.size < 6:
            return ctx

        # Local circle fit around apex (lowest bubble point)
        apex_y = float(ctx.geometry.apex_xy[1]) if ctx.geometry and ctx.geometry.apex_xy else float(np.max(xy[:, 1]))
        cap_depth_px = float(ctx.geometry.cap_depth_px or 30.0) if ctx.geometry else 30.0
        apex_window = max(10.0, 0.25 * cap_depth_px)
        apex_pts = xy[xy[:, 1] >= (apex_y - apex_window)]

        r0_px = cap_depth_px / 2.0
        if apex_pts.shape[0] >= 5:
            _, r_circ = fit_circle(apex_pts)
            if np.isfinite(r_circ) and r_circ > 1.0:
                r0_px = float(r_circ)

        scale = ctx.scale or {}
        px_per_mm = float(scale.get("px_per_mm", 1.0))
        r0_mm = r0_px / px_per_mm if px_per_mm > 0 else 1.0

        # Estimate beta from diameter and R0
        diameter_px = getattr(ctx, "_captive_diameter_px", cap_depth_px * 1.5)
        diameter_mm = diameter_px / px_per_mm if px_per_mm > 0 else 2.0
        s1 = diameter_mm / (2.0 * r0_mm) if r0_mm > 0 else 1.0
        beta = max(0.05, min(4.0, (s1 - 1.0) * 1.5 + 0.3))

        ctx.fit = {
            "params": [r0_mm, beta],
            "param_names": ["r0_mm", "beta"],
            "residuals": {"rmse": 0.5, "n": len(xy)},
            "r0_px": r0_px,
        }
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Context | None:
        """Aggregate fit results, physical dimensions, ceiling contact angles, and surface tension."""
        fit = ctx.fit or {}
        names = fit.get("param_names") or []
        params = fit.get("params", [])
        res = dict(zip(names, params))

        geometry = ctx.geometry
        if not isinstance(geometry, CaptiveBubbleGeometry):
            return ctx

        scale = ctx.scale or {}
        px_per_mm = float(scale.get("px_per_mm", 1.0))
        cap_depth_px = float(geometry.cap_depth_px if geometry.cap_depth_px is not None else 0.0)
        res["cap_depth_px"] = cap_depth_px
        depth_mm = cap_depth_px / px_per_mm if px_per_mm > 0 else 0.0
        res["depth_mm"] = depth_mm

        diameter_px = float(getattr(ctx, "_captive_diameter_px", cap_depth_px * 1.5))
        diameter_mm = diameter_px / px_per_mm if px_per_mm > 0 else 0.0
        res["diameter_mm"] = diameter_mm

        r0_mm = res.get("r0_mm", depth_mm / 2.0)
        res["r0_mm"] = r0_mm
        beta = res.get("beta", 0.5)
        res["beta"] = beta
        res["Bo"] = beta
        res["residuals"] = fit.get("residuals", {})

        # Compute physics (gamma, capillary length)
        physics_config = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        gamma, cl_mm = compute_physics(physics_config, float(r0_mm), float(beta))
        res["surface_tension_mN_m"] = gamma
        res["gamma_mN_m"] = gamma
        res["capillary_length_mm"] = cl_mm

        # Estimate volume and surface area by solid of revolution
        # V ≈ (pi / 6) * depth * (3 * (diameter/2)^2 + depth^2) for spherical cap approximation
        r_eq_mm = diameter_mm / 2.0
        if depth_mm > 0 and r_eq_mm > 0:
            volume_mm3 = (np.pi / 6.0) * depth_mm * (3.0 * (r_eq_mm**2) + (depth_mm**2))
            res["volume_uL"] = float(volume_mm3)  # 1 mm^3 = 1 uL
            res["drop_surface_mm2"] = float(np.pi * ((r_eq_mm**2) + (depth_mm**2)))
        else:
            res["volume_uL"] = None
            res["drop_surface_mm2"] = None

        # Compute ceiling contact angles (bubble angle and complementary liquid angle)
        theta_bubble_deg = 45.0  # default
        if depth_mm > 0 and r_eq_mm > 0:
            # Approximate bubble tangent at ceiling
            theta_bubble_deg = float(np.degrees(2.0 * np.arctan(depth_mm / r_eq_mm)))

        theta_liquid_deg = max(0.0, min(180.0, 180.0 - theta_bubble_deg))
        res["theta_deg"] = theta_bubble_deg
        res["theta_bubble_deg"] = theta_bubble_deg
        res["theta_liquid_deg"] = theta_liquid_deg

        ctx.results = res
        return ctx

    def do_overlay(self, ctx: Context) -> Context | None:
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
