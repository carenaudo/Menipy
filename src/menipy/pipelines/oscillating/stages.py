"""Stages.

Module implementation."""


# pipeline/pendant/stages.py (test-specific overrides)
from __future__ import annotations

from typing import Optional, List
import numpy as np

from menipy.pipelines.base import PipelineBase
from menipy.models.context import Context
from menipy.models.fit import FitConfig
from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl
from menipy.common import solver as common_solver
from pathlib import Path
from menipy.common.plugins import _load_module_from_path

_repo_root = Path(__file__).resolve().parents[4]
_toy_path = _repo_root / "plugins" / "toy_young_laplace.py"
_toy_mod = _load_module_from_path(_toy_path, "adsa_plugins.toy_young_laplace")
young_laplace_sphere = getattr(_toy_mod, "toy_young_laplace")


def _contour_from_frame(ctx: Context, frame) -> np.ndarray:
    """Run edge detection on a single frame and return Nx2 contour (float)."""
    # Save/restore current frame(s)
    original = ctx.frames
    ctx.frames = [frame]
    edged.run(ctx, method="canny")
    xy = np.asarray(ctx.contour.xy, dtype=float)
    ctx.frames = original
    return xy


def _area_equiv_radius(xy: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Return area-equivalent radius (px) and centroid (cx, cy) for polygon xy."""
    x, y = xy[:, 0], xy[:, 1]
    # polygon area (signed); centroid
    a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(a) < 1e-6:
        return 0.0, (float(np.mean(x)), float(np.mean(y)))
    cx = (1.0 / (6.0 * a)) * np.sum(
        (x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)
    )
    cy = (1.0 / (6.0 * a)) * np.sum(
        (y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)
    )
    r_eq = np.sqrt(abs(a) / np.pi)
    return float(r_eq), (float(cx), float(cy))


class OscillatingPipeline(PipelineBase):
    """Oscillating drop: contour per frame → radius time series → (toy) fit on frame 0 + frequency estimate."""

    name = "oscillating"

    # UI metadata for plugin-centric configuration
    ui_metadata = {
        "display_name": "Oscillating Drop",
        "icon": "oscillating.svg",
        "color": "#F5A623",
        "stages": [
            "acquisition",
            "preprocessing",
            "contour_extraction",
            "geometric_features",
            "physics",
        ],
        "calibration_params": [
            "needle_length_mm",
            "drop_density_kg_m3",
            "fluid_density_kg_m3",
        ],
        "primary_metrics": [
            "surface_tension_mN_m",
            "oscillation_frequency",
            "damping_ratio",
        ],
    }

    def do_acquisition(self, ctx: Context) -> Optional[Context]:
        return ctx

    def do_preprocessing(self, ctx: Context) -> Optional[Context]:
        return ctx

    def do_contour_extraction(self, ctx: Context) -> Optional[Context]:
        """Extract contour from each frame for oscillation analysis."""
        frames = (
            ctx.frames
            if isinstance(ctx.frames, list)
            else ([ctx.frames] if ctx.frames is not None else [])
        )
        contours: List[object] = []
        for f in frames[:]:  # safe slice
            xy = _contour_from_frame(ctx, f)
            C = type("Contour", (), {})()
            C.xy = xy
            contours.append(C)
        ctx.contours_by_frame = contours or None

        # Provide a current contour (frame 0) for downstream stages
        if contours:
            ctx.contour = contours[0]
        return ctx

    def do_geometric_features(self, ctx: Context) -> Optional[Context]:
        """Extract axis, apex and radius series from contours."""
        # Use frame 0 for geometry refs; also build r_eq(t)
        if getattr(ctx, "contours_by_frame", None):
            series = []
            centers = []
            for C in ctx.contours_by_frame:
                r, (cx, cy) = _area_equiv_radius(np.asarray(C.xy))
                series.append(r)
                centers.append((cx, cy))
            ctx.geometry = (ctx.geometry or {}) | {
                "r_eq_series_px": series,
                "centers_px": centers,
            }
            # Assign a few handy refs from frame 0
            xy0 = np.asarray(ctx.contours_by_frame[0].xy)
        else:
            # Single contour fallback
            edged.run(ctx, method="canny")
            xy0 = np.asarray(ctx.contour.xy)

        x0, y0 = xy0[:, 0], xy0[:, 1]
        axis_x = float(np.median(x0))
        apex_i = int(np.argmin(y0))  # top-most point (visual guide only)
        apex_xy = (float(x0[apex_i]), float(y0[apex_i]))

        # Store frame-0 equivalent radius/center for overlay
        r0, (c0x, c0y) = _area_equiv_radius(xy0)
        ctx.geometry = (ctx.geometry or {}) | {
            "axis_x": axis_x,
            "apex_xy": apex_xy,
            "r0_eq_px": float(r0),
            "c0_xy": (float(c0x), float(c0y)),
        }
        return ctx

    def do_calibration(self, ctx: Context) -> Optional[Context]:
        """Set up pixel-to-mm scaling."""
        ctx.scale = ctx.scale or {"px_per_mm": 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        # include fps if known (used to estimate f0)
        ctx.physics = ctx.physics or {}
        ctx.physics.setdefault("fps", 100.0)  # default if unknown
        ctx.physics.setdefault("rho1", 1000.0)
        ctx.physics.setdefault("rho2", 1.2)
        ctx.physics.setdefault("g", 9.80665)
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Optional[Context]:
        """Fit R0 from frame 0 contour."""
        # Simple: fit R0_mm on frame-0 only (toy model)
        cfg = FitConfig(
            x0=[30.0],
            bounds=([1.0], [2000.0]),
            loss="soft_l1",
            distance="pointwise",
            param_names=["R0_mm"],
        )
        common_solver.run(ctx, integrator=young_laplace_sphere, config=cfg)
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Optional[Context]:
        """Aggregate fit results, compute frequency from oscillation data."""
        # First estimate oscillation frequency from r_eq(t)
        series = (ctx.geometry or {}).get("r_eq_series_px")
        fps = (ctx.physics or {}).get("fps", None)
        f0 = None
        if series and fps and len(series) >= 8:
            arr = np.asarray(series, dtype=float)
            arr = arr - np.mean(arr)
            # FFT magnitude (one-sided), ignore DC
            n = int(2 ** np.ceil(np.log2(len(arr))))
            mag = np.abs(np.fft.rfft(arr, n=n))
            freqs = np.fft.rfftfreq(n, d=1.0 / float(fps))
            if len(freqs) > 1:
                mag[0] = 0.0
                k = int(np.argmax(mag))
                f0 = float(freqs[k])
        
        # Collect fit results
        fit = ctx.fit or {}
        names = list(fit.get("param_names") or [])
        params = list(fit.get("params", []))
        
        # Add frequency if found
        if f0 is not None:
            names.append("f0_Hz")
            params.append(f0)
        
        results = {n: p for n, p in zip(names, params)}
        results["residuals"] = fit.get("residuals", {})
        # Export a couple of geometry refs
        if ctx.geometry:
            results["r0_eq_px"] = ctx.geometry.get("r0_eq_px")
        ctx.results = results
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = np.asarray(ctx.contour.xy, dtype=float)
        cx, cy = ctx.geometry.get(
            "c0_xy", (float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1])))
        )
        r = float(ctx.geometry.get("r0_eq_px", 10.0))
        axis_x = int(round(ctx.geometry.get("axis_x", cx)))
        text = f"R0≈{ctx.results.get('R0_mm','?')} mm"
        f0 = ctx.results.get("f0_Hz", None)
        if f0 is not None:
            text += f" | f0≈{f0:.2f} Hz"
        cmds = [
            {
                "type": "polyline",
                "points": xy.tolist(),
                "closed": True,
                "color": "yellow",
                "thickness": 2,
            },
            {
                "type": "circle",
                "center": (int(cx), int(cy)),
                "radius": int(r),
                "color": "magenta",
                "thickness": 1,
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
                "p": (int(cx), int(cy)),
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
        """do validation.

        Parameters
        ----------
        ctx : type
        Description.

        Returns
        -------
        type
        Description.
        """
        ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
        ctx.qa = {"ok": ok}
        return ctx
