# menipy/pipelines/pendant/stages.py (test-specific overrides)
from __future__ import annotations

from typing import Optional
from pathlib import Path
import numpy as np

from menipy.pipelines.base import PipelineBase
from menipy.models.context import Context
from menipy.models.fit import FitConfig
from menipy.models.config import EdgeDetectionSettings
from menipy.models.geometry import Contour
from menipy.common import solver as common_solver
from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl 
from menipy.common.plugin_loader import get_solver
from menipy.pipelines.utils import ensure_contour

# Get solvers from registry (loaded at startup)
young_laplace_adsa = get_solver("young_laplace_adsa")
calculate_surface_tension = get_solver("calculate_surface_tension")



class PendantPipeline(PipelineBase):
    """Pendant drop pipeline (simplified): contour → axis/apex → toy Y–L radius fit."""

    name = "pendant"

    # UI metadata for plugin-centric configuration
    ui_metadata = {
        "display_name": "Pendant Drop",
        "icon": "pendant.svg",
        "color": "#7ED321",
        "stages": ["acquisition", "edge_detection", "geometry", "physics"],
        "calibration_params": ["needle_diameter_mm", "drop_density_kg_m3", "fluid_density_kg_m3"],
        "primary_metrics": ["surface_tension_mN_m", "volume_uL", "beta"]
    }

    def do_acquisition(self, ctx: Context) -> Optional[Context]:
        """Load frames from disk or wrap existing image in frames."""
        # Debug: log what we received using pipeline logger
        self.logger.info(f"[do_acquisition] ctx.image type: {type(getattr(ctx, 'image', None))}, ctx.image_path: {getattr(ctx, 'image_path', None)}")
        
        # If frames already exist, nothing to do
        if getattr(ctx, "frames", None) and len(ctx.frames) > 0:
            self.logger.info(f"[do_acquisition] Frames already exist: {len(ctx.frames)}")
            return ctx

        # Check if we have a direct image (numpy array)
        image = getattr(ctx, "image", None)
        if image is not None and isinstance(image, np.ndarray):
            from menipy.models.frame import Frame
            frame = Frame(image=image)
            ctx.frames = [frame]
            ctx.frame = frame
            self.logger.info(f"[do_acquisition] Wrapped ctx.image in frame, shape: {image.shape}")
            return ctx

        # Try to load from file path
        image_path = getattr(ctx, "image_path", None)
        if not image_path:
            # No image source available
            self.logger.warning("[do_acquisition] No image or image_path available!")
            return ctx

        try:
            from menipy.common import acquisition as acq
            self.logger.info(f"[do_acquisition] Loading from file: {image_path}")
            frames = acq.from_file([image_path])
        except Exception as e:
            self.logger.error(f"[do_acquisition] Failed to load from file: {e}")
            frames = []

        if frames:
            ctx.frames = frames
            ctx.frame = frames[0]
            ctx.image = frames[0]
            self.logger.info(f"[do_acquisition] Loaded {len(frames)} frames from disk")
        return ctx

    def do_preprocessing(self, ctx: Context) -> Optional[Context]:
        """Run preprocessing with automatic feature detection."""
        from menipy.pipelines.pendant.preprocessing import do_preprocessing
        return do_preprocessing(ctx)

    def do_geometry(self, ctx: Context) -> Optional[Context]:
        xy = ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]
        axis_x = float(np.median(x))
        apex_i = int(np.argmax(y))  # pendant apex at bottom
        apex_xy = (float(x[apex_i]), float(y[apex_i]))
        
        from menipy.models.geometry import Geometry
        ctx.geometry = Geometry(
            axis_x=axis_x,
            apex_xy=apex_xy
        )
        return ctx

    def do_scaling(self, ctx: Context) -> Optional[Context]:
        """Calculate px_per_mm from needle calibration and scale contour to mm."""
        # Check if scaling already set
        if ctx.scale and ctx.scale.get("px_per_mm", 1.0) != 1.0:
            return ctx
        
        # Try to calculate px_per_mm from needle calibration
        needle_diameter_mm = getattr(ctx, "needle_diameter_mm", None)
        needle_rect = getattr(ctx, "needle_rect", None)
        
        px_per_mm = 1.0  # Default (no scaling)
        
        if needle_diameter_mm and needle_rect:
            # needle_rect is typically (x, y, width, height) in pixels
            try:
                if hasattr(needle_rect, '__iter__') and len(needle_rect) >= 3:
                    needle_width_px = needle_rect[2]  # width in pixels
                    if needle_width_px > 0 and needle_diameter_mm > 0:
                        px_per_mm = needle_width_px / needle_diameter_mm
                        self.logger.info(f"Calibration: needle {needle_width_px}px = {needle_diameter_mm}mm → {px_per_mm:.2f} px/mm")
            except Exception as e:
                self.logger.warning(f"Could not calculate scale from needle: {e}")
        elif needle_diameter_mm:
            self.logger.warning(f"needle_diameter_mm={needle_diameter_mm} but no needle_rect provided for calibration")
        
        ctx.scale = {"px_per_mm": px_per_mm}
        
        # Scale the contour to mm if we have valid calibration
        if px_per_mm != 1.0 and ctx.contour and hasattr(ctx.contour, "xy"):
            from menipy.models.geometry import Contour
            xy_px = np.asarray(ctx.contour.xy, dtype=float)
            xy_mm = xy_px / px_per_mm
            ctx.contour = Contour(xy=xy_mm.tolist())
            self.logger.info(f"Scaled contour from pixels to mm (factor: 1/{px_per_mm:.2f})")
            
            # Also update geometry apex_xy if present
            if ctx.geometry and ctx.geometry.apex_xy:
                apex_x, apex_y = ctx.geometry.apex_xy
                ctx.geometry.apex_xy = (apex_x / px_per_mm, apex_y / px_per_mm)
                if ctx.geometry.axis_x:
                    ctx.geometry.axis_x = ctx.geometry.axis_x / px_per_mm
        
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_solver(self, ctx: Context) -> Optional[Context]:
        # Fit both apex radius (R0) and Bond number (beta)
        # beta = (delta_rho * g * R0^2) / gamma
        cfg = FitConfig(
            x0=[5.0, 0.3],  # Initial guess: R0=5mm, beta=0.3
            bounds=([0.1, 0.01], [50.0, 2.0]),  # R0: 0.1-50mm, beta: 0.01-2.0
            loss="soft_l1",
            distance="pointwise",
            param_names=["r0_mm", "beta"],
        )
        common_solver.run(ctx, integrator=young_laplace_adsa, config=cfg)
        return ctx

    def do_optimization(self, ctx: Context) -> Optional[Context]: return ctx

    def do_outputs(self, ctx: Context) -> Optional[Context]:
        fit = ctx.fit or {}
        names = fit.get("param_names") or []
        params = fit.get("params", [])
        
        # Base results from fit
        results = {n: p for n, p in zip(names, params)} | {"residuals": fit.get("residuals", {})}

        # --- Geometric Calculations ---
        # 1. Scale
        scale = ctx.scale or {}
        px_per_mm = scale.get("px_per_mm", 1.0)
        
        # 2. Contour Data
        try:
            xy = ensure_contour(ctx)
            x, y = xy[:, 0], xy[:, 1]
            
            # 3. Height (mm)
            height_px = np.max(y) - np.min(y)
            results["height_mm"] = height_px / px_per_mm
            
            # 4. Diameter (mm) - derived from R0 fit for consistency with spherical model
            if "r0_mm" in results:
                results["diameter_mm"] = 2 * results["r0_mm"]
            
            # 5. Volume (uL) - Disk integration method
            axis_x = ctx.geometry.axis_x if ctx.geometry else np.mean(x)
            r_px = np.abs(x - axis_x)
            
            sort_idx = np.argsort(y)
            y_sorted = y[sort_idx]
            r_sorted = r_px[sort_idx]
            
            vol_px3 = np.pi * np.trapz(r_sorted**2, y_sorted)
            results["volume_uL"] = abs(vol_px3) / (px_per_mm**3)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate geometric stats: {e}")

        # --- Surface Tension Calculation ---
        # gamma = (delta_rho * g * R0^2) / beta
        try:
            if "r0_mm" in results and "beta" in results:
                physics = ctx.physics or {}
                rho1 = physics.get("rho1", 1000.0)  # liquid density kg/m³
                rho2 = physics.get("rho2", 1.2)      # ambient density kg/m³
                g = physics.get("g", 9.80665)
                delta_rho = rho1 - rho2
                
                gamma = calculate_surface_tension(
                    R0_mm=results["r0_mm"],
                    beta=results["beta"],
                    delta_rho_kg_m3=delta_rho,
                    g=g
                )
                results["surface_tension_mN_m"] = gamma
        except Exception as e:
            self.logger.warning(f"Failed to calculate surface tension: {e}")

        ctx.results = results
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = ensure_contour(ctx)
        if not ctx.geometry:
            return ctx
        axis_x = int(round(ctx.geometry.axis_x)) if ctx.geometry.axis_x is not None else 0
        apex_x, apex_y = ctx.geometry.apex_xy if ctx.geometry.apex_xy is not None else (0, 0)
        text = f"R0≈{ctx.results.get('r0_mm','?')} mm"
        measurement_text = f"Measurement #{ctx.measurement_sequence}" if hasattr(ctx, 'measurement_sequence') and ctx.measurement_sequence else ""
        cmds = [
            # Measurement number overlay
            {"type": "text", "p": (10, 25), "text": measurement_text, "color": "white", "scale": 0.7, "thickness": 2},
            {"type": "polyline", "points": xy.tolist(), "closed": True, "color": "yellow", "thickness": 2},
            {"type": "line", "p1": (axis_x, 0), "p2": (axis_x, int(np.max(xy[:, 1]) + 10)), "color": "cyan", "thickness": 1},
            {"type": "cross", "p": (int(apex_x), int(apex_y)), "color": "red", "size": 6, "thickness": 2},
            {"type": "text", "p": (10, 20), "text": text, "color": "white", "scale": 0.55},
        ]
        return ovl.run(ctx, commands=cmds, alpha=0.6)

    def do_validation(self, ctx: Context) -> Optional[Context]:
        ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
        ctx.qa = {"ok": ok}
        return ctx
