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
from menipy.common.plugins import _load_module_from_path
from menipy.models.config import EdgeDetectionSettings
from menipy.models.geometry import Contour
from menipy.common import solver as common_solver
from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl 
from menipy.common.plugins import _load_module_from_path

# Load the toy solver from plugins
_repo_root = Path(__file__).resolve().parents[4]
_toy_path = _repo_root / "plugins" / "toy_young_laplace.py"
_toy_mod = _load_module_from_path(_toy_path, "adsa_plugins.toy_young_laplace")
young_laplace_sphere = getattr(_toy_mod, "toy_young_laplace")
from menipy.models.config import EdgeDetectionSettings
from menipy.models.fit import FitConfig

def _ensure_contour(ctx: Context) -> np.ndarray:
    if getattr(ctx, "contour", None) is not None and hasattr(ctx.contour, "xy"):
        return np.asarray(ctx.contour.xy, dtype=float)

    # Ensure we have at least one frame to run edge detection on
    frames = getattr(ctx, "frames", None)
    if (not frames or len(frames) == 0) and getattr(ctx, "image_path", None):
        try:
            from menipy.common import acquisition as acq
            loaded = acq.from_file([ctx.image_path])
        except Exception:
            loaded = []
        if loaded:
            ctx.frames = loaded
            ctx.frame = loaded[0]
            ctx.image = loaded[0]

    edged.run(ctx, settings=ctx.edge_detection_settings or EdgeDetectionSettings(method="canny"))
    return np.asarray(ctx.contour.xy, dtype=float)

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
        """Load frames from disk if the context only has a path reference."""
        if getattr(ctx, "frames", None):
            return ctx

        image_path = getattr(ctx, "image_path", None)
        if not image_path:
            return ctx

        try:
            from menipy.common import acquisition as acq
            frames = acq.from_file([image_path])
        except Exception:
            frames = []

        if frames:
            ctx.frames = frames
            ctx.frame = frames[0]
            ctx.image = frames[0]
        return ctx
    def do_preprocessing(self, ctx: Context) -> Optional[Context]: return ctx

    def do_geometry(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
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
        ctx.scale = ctx.scale or {"px_per_mm": 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_solver(self, ctx: Context) -> Optional[Context]:
        cfg = FitConfig(
            x0=[25.0],
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
        ctx.results = {n: p for n, p in zip(names, params)} | {"residuals": fit.get("residuals", {})}
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = _ensure_contour(ctx)
        if not ctx.geometry:
            return ctx
        axis_x = int(round(ctx.geometry.axis_x)) if ctx.geometry.axis_x is not None else 0
        apex_x, apex_y = ctx.geometry.apex_xy if ctx.geometry.apex_xy is not None else (0, 0)
        text = f"R0≈{ctx.results.get('R0_mm','?')} mm"
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
