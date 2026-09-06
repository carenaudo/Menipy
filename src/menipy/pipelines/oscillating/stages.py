"""Stages.

Module implementation."""

# pipeline/pendant/stages.py (test-specific overrides)
from __future__ import annotations

from pathlib import Path

import numpy as np

from menipy.common import edge_detection as edged
from menipy.common import overlay as ovl
from menipy.common import solver as common_solver
from menipy.common._module_loader import load_module_from_path
from menipy.math.apex import detect_apex
from menipy.models.context import Context
from menipy.models.fit import FitConfig
from menipy.models.geometry import Contour, Geometry
from menipy.pipelines.base import PipelineBase

_repo_root = Path(__file__).resolve().parents[4]
_toy_path = _repo_root / "plugins" / "toy_young_laplace.py"
_toy_mod = load_module_from_path(_toy_path, "menipy_plugins.toy_young_laplace")
young_laplace_sphere = _toy_mod.toy_young_laplace


def _contour_from_frame(ctx: Context, frame) -> np.ndarray:
    """Run edge detection on a single frame and return Nx2 contour (float)."""
    # Save/restore current frame(s)
    original = ctx.frames
    ctx.frames = [frame]
    from menipy.models.config import EdgeDetectionSettings

    edged.run(ctx, settings=EdgeDetectionSettings(method="canny"))
    if ctx.contour is not None and ctx.contour.xy is not None:
        xy = np.asarray(ctx.contour.xy, dtype=float)
    else:
        xy = np.empty((0, 2), dtype=float)
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
            "needle_diameter_mm",
            "drop_density_kg_m3",
            "fluid_density_kg_m3",
        ],
        "primary_metrics": [
            "surface_tension_mN_m",
            "oscillation_frequency",
            "damping_ratio",
        ],
    }

    def do_acquisition(self, ctx: Context) -> Context | None:
        return ctx

    def do_preprocessing(self, ctx: Context) -> Context | None:
        return ctx

    def do_contour_extraction(self, ctx: Context) -> Context | None:
        """Extract contour from each frame for oscillation analysis."""
        frames = (
            ctx.frames
            if isinstance(ctx.frames, list)
            else ([ctx.frames] if ctx.frames is not None else [])
        )
        contours: list[object] = []
        for f in frames[:]:  # safe slice
            xy = _contour_from_frame(ctx, f)
            C = Contour(xy=xy)
            contours.append(C)
        ctx.contours_by_frame = contours

        # Provide a current contour (frame 0) for downstream stages
        if contours:
            ctx.contour = contours[0]
        return ctx

    def do_geometric_features(self, ctx: Context) -> Context | None:
        """Extract axis, apex and radius series from contours."""
        # Use frame 0 for geometry refs; also build r_eq(t)
        if getattr(ctx, "contours_by_frame", None):
            series = []
            centers = []
            for C in ctx.contours_by_frame:
                r, (cx, cy) = _area_equiv_radius(np.asarray(C.xy))
                series.append(r)
                centers.append((cx, cy))

            ctx.r_eq_series_px = series
            ctx.centers_px = centers

            # Assign a few handy refs from frame 0
            xy0 = np.asarray(ctx.contours_by_frame[0].xy)
        elif ctx.contour is not None and ctx.contour.xy is not None:
            xy0 = np.asarray(ctx.contour.xy, dtype=float)
        else:
            # Single contour fallback
            from menipy.models.config import EdgeDetectionSettings

            edged.run(ctx, settings=EdgeDetectionSettings(method="canny"))
            if ctx.contour is not None and ctx.contour.xy is not None:
                xy0 = np.asarray(ctx.contour.xy, dtype=float)
            else:
                xy0 = np.empty((0, 2), dtype=float)

        x0 = xy0[:, 0] if xy0.shape[0] > 0 else np.empty(0)
        axis_x = float(np.median(x0)) if x0.size > 0 else 0.0
        apex_xy = detect_apex(xy0, mode="sessile", refine=True).point if xy0.shape[0] >= 3 else (0.0, 0.0)

        # Store frame-0 equivalent radius/center for overlay
        r0, (c0x, c0y) = _area_equiv_radius(xy0)

        ctx.geometry = Geometry(
            axis_x=axis_x,
            apex_xy=apex_xy,
        )
        ctx.r0_eq_px = float(r0)
        ctx.c0_xy = (float(c0x), float(c0y))

        return ctx

    def do_calibration(self, ctx: Context) -> Context | None:
        """Set up pixel-to-mm scaling."""
        ctx.scale = ctx.scale or {"px_per_mm": 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Context | None:
        # include fps if known (used to estimate f0)
        ctx.physics = ctx.physics or {}
        ctx.physics.setdefault("fps", 100.0)  # default if unknown
        ctx.physics.setdefault("rho1", 1000.0)
        ctx.physics.setdefault("rho2", 1.2)
        ctx.physics.setdefault("g", 9.80665)
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Context | None:
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

    def do_compute_metrics(self, ctx: Context) -> Context | None:
        """Aggregate fit results, compute frequency, Rayleigh-Lamb surface tension, and rheology metrics."""
        from menipy.math.rheology import (
            dilational_viscoelasticity,
            harmonic_sine_fit,
            rayleigh_lamb_surface_tension,
        )

        series = getattr(ctx, "r_eq_series_px", None)
        physics = ctx.physics or {}
        fps = float(physics.get("fps", 100.0))
        rho1 = float(physics.get("rho1", 1000.0))
        rho2 = float(physics.get("rho2", 1.2))

        scale = ctx.scale or {}
        px_per_mm = float(scale.get("px_per_mm", 1.0))

        f0 = None
        snr = None
        peak_width_Hz = None
        n_frames = len(series) if series else 1

        if series and len(series) >= 8:
            arr = np.asarray(series, dtype=float)
            y0 = float(np.mean(arr))
            arr_detrended = arr - y0

            # Windowed FFT (Hann window)
            window = np.hanning(len(arr))
            n = int(2 ** np.ceil(np.log2(len(arr))))
            mag = np.abs(np.fft.rfft(arr_detrended * window, n=n))
            freqs = np.fft.rfftfreq(n, d=1.0 / fps)

            if len(freqs) > 1:
                mag[0] = 0.0
                k = int(np.argmax(mag))
                f0 = float(freqs[k])
                noise_floor = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1e-6
                snr = float(mag[k] / max(1e-6, noise_floor))
                df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.5
                peak_width_Hz = float(df * 2.0)

            # Refined harmonic sine fitting
            time_s = np.arange(len(arr)) / fps
            y0_fit, amp_fit, f_fit, phi_fit = harmonic_sine_fit(time_s, arr, frequency_hz=f0)
            if np.isfinite(f_fit) and f_fit > 0:
                f0 = f_fit

        # Collect fit results from stage 0 profile fitting
        fit = ctx.fit or {}
        names = list(fit.get("param_names") or [])
        params = list(fit.get("params", []))

        results = dict(zip(names, params))
        results["residuals"] = fit.get("residuals", {})

        r0_eq_px = float(getattr(ctx, "r0_eq_px", 10.0) or (series[0] if series else 10.0))
        results["r0_eq_px"] = r0_eq_px
        r0_eq_mm = r0_eq_px / px_per_mm if px_per_mm > 0 else 1.0
        results["r0_eq_mm"] = r0_eq_mm

        results["fps"] = fps
        results["n_frames"] = n_frames
        results["window"] = "hann"
        results["estimator"] = "fft_harmonic"

        if f0 is not None:
            results["f0_Hz"] = float(f0)
        else:
            results["f0_Hz"] = 1.0  # fallback

        if snr is not None:
            results["snr"] = float(snr)
        if peak_width_Hz is not None:
            results["peak_width_Hz"] = float(peak_width_Hz)

        # Rayleigh-Lamb natural droplet oscillation surface tension
        if f0 is not None and f0 > 0 and r0_eq_mm > 0:
            r0_m = r0_eq_mm * 1e-3
            gamma_n_m = rayleigh_lamb_surface_tension(
                frequency_hz=f0,
                radius_m=r0_m,
                rho_drop_kg_m3=rho1,
                rho_medium_kg_m3=rho2,
                mode_n=2,
            )
            gamma_mN_m = gamma_n_m * 1e3
            results["gamma_mN_m"] = float(gamma_mN_m)
            results["surface_tension_mN_m"] = float(gamma_mN_m)

        # Interfacial dilational rheology (moduli E, E', E'', eta_d)
        if series and len(series) >= 8 and px_per_mm > 0:
            r_arr_mm = np.asarray(series, dtype=float) / px_per_mm
            r_mean_mm = float(np.mean(r_arr_mm))
            delta_r_mm = float((np.max(r_arr_mm) - np.min(r_arr_mm)) / 2.0)
            a0_mm2 = float(4.0 * np.pi * (r_mean_mm**2))
            delta_a_mm2 = float(8.0 * np.pi * r_mean_mm * delta_r_mm)

            # Expected surface tension response from Rayleigh-Lamb or dynamic response
            delta_gamma = 5.0  # default perturbation 5 mN/m if no explicit sensor
            phase_shift_rad = float(np.radians(getattr(ctx, "oscillation_phase_deg", 15.0) or 15.0))
            rheo = dilational_viscoelasticity(
                A0=a0_mm2,
                delta_A=delta_a_mm2,
                delta_gamma=delta_gamma,
                phase_shift_rad=phase_shift_rad,
                frequency_hz=float(f0 or 1.0),
            )
            results.update(rheo)

        ctx.results = results
        return ctx

    def do_overlay(self, ctx: Context) -> Context | None:
        if ctx.contour is not None and ctx.contour.xy is not None:
            xy = np.asarray(ctx.contour.xy, dtype=float)
        else:
            xy = np.empty((0, 2), dtype=float)

        if xy.size > 0:
            cx, cy = getattr(
                ctx, "c0_xy", (float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1])))
            )
        else:
            cx, cy = getattr(ctx, "c0_xy", (0.0, 0.0))
        r = float(getattr(ctx, "r0_eq_px", 10.0))

        geom_axis_x = getattr(ctx.geometry, "axis_x", cx) if ctx.geometry else cx
        axis_x = int(round(geom_axis_x))

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
