"""Stages.

Module implementation."""


# src/menipy/pipelines/pendant/stages.py
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import trapezoid

from menipy.pipelines.base import PipelineBase
from menipy.models.context import Context
from menipy.common.geometry import fit_circle
from menipy.common import overlay as ovl
from menipy.common import registry
from menipy.models.surface_tension import (
    bond_number as bond_number_from_gamma,
    jennings_pallas_beta,
    surface_tension as surface_tension_n_per_m,
)
from menipy.models.drop_extras import vmax_uL, worthington_number
from menipy.pipelines.pendant.strict_young_laplace import (
    PendantStrictFitInput,
    build_pendant_profile_envelope_mm,
    fit_pendant_young_laplace_strict,
)
from menipy.pipelines.utils import ensure_contour
from menipy.pipelines.pendant import approximations as _pendant_approximations  # noqa: F401


DEFAULT_PENDANT_APPROXIMATION_METHODS = [
    "selected_plane",
    "multi_selected_plane",
    "volume_apex_lookup",
]


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


def _clip_contour_at_pendant_contacts(
    xy: np.ndarray, contact_points: object | None
) -> np.ndarray:
    """Remove needle-shaft contour points above the pendant contact level."""
    if contact_points is None:
        return xy

    try:
        contacts = np.asarray(contact_points, dtype=float).reshape(-1, 2)
    except Exception:
        return xy

    if contacts.shape[0] < 2 or xy.size == 0:
        return xy

    contact_y = float(np.min(contacts[:2, 1]))
    clipped = xy[xy[:, 1] >= contact_y]
    if clipped.shape[0] < 3:
        return xy

    return np.vstack([contacts[:2], clipped])


def _pendant_max_width(
    xy: np.ndarray, contact_points: object | None = None
) -> tuple[float, tuple[tuple[int, int], tuple[int, int]]]:
    """Return maximum horizontal contour width in pixels and its line."""
    if xy.size == 0:
        return 0.0, ((0, 0), (0, 0))

    measured = xy
    if contact_points is not None:
        try:
            contacts = np.asarray(contact_points, dtype=float).reshape(-1, 2)
            if contacts.shape[0] >= 2:
                contact_y = float(np.min(contacts[:2, 1]))
                candidate = xy[xy[:, 1] >= contact_y]
                if candidate.shape[0] >= 2:
                    measured = candidate
        except Exception:
            measured = xy

    row_keys = np.rint(measured[:, 1]).astype(int)
    best_width = 0.0
    best_line = ((0, 0), (0, 0))
    for row in np.unique(row_keys):
        xs = measured[row_keys == row, 0]
        if xs.size < 2:
            continue
        width = float(np.max(xs) - np.min(xs))
        if width > best_width:
            x_min = float(np.min(xs))
            x_max = float(np.max(xs))
            best_width = width
            best_line = ((int(round(x_min)), int(row)), (int(round(x_max)), int(row)))

    if best_width <= 0.0 and measured.shape[0] >= 2:
        x_min = float(np.min(measured[:, 0]))
        x_max = float(np.max(measured[:, 0]))
        y_mid = int(round(float(np.median(measured[:, 1]))))
        best_width = x_max - x_min
        best_line = ((int(round(x_min)), y_mid), (int(round(x_max)), y_mid))

    return best_width, best_line


def _pendant_apex_radius_px(
    xy: np.ndarray, apex_xy: tuple[float, float] | None, window_px: float = 20.0
) -> float:
    """Fit a local circle around the pendant apex and return its pixel radius."""
    if apex_xy is None or xy.size == 0:
        return 0.0

    apex_y = float(apex_xy[1])
    apex_pts = xy[(xy[:, 1] - apex_y) > -float(window_px)]
    if apex_pts.shape[0] < 3:
        return 0.0

    _, radius = fit_circle(apex_pts)
    if not np.isfinite(radius) or radius <= 0:
        return 0.0
    return float(radius)


def _profile_fit_unreliable(fit: dict, rmse_threshold_px: float = 25.0) -> bool:
    """Return True when the profile fit residual is too large for reporting."""
    residuals = fit.get("residuals") or {}
    rmse = residuals.get("rmse")
    if rmse is None:
        return False
    try:
        rmse_value = float(rmse)
    except (TypeError, ValueError):
        return False
    return np.isfinite(rmse_value) and rmse_value > rmse_threshold_px


def _radial_profile_integrals(profile_mm: np.ndarray) -> tuple[float, float]:
    """Return ``(volume_uL, surface_mm2)`` for a radial ``(r, z)`` profile."""
    profile = np.asarray(profile_mm, dtype=float).reshape(-1, 2)
    if profile.shape[0] < 3:
        return 0.0, 0.0

    profile = profile[np.all(np.isfinite(profile), axis=1)]
    profile = profile[profile[:, 0] >= 0]
    if profile.shape[0] < 3:
        return 0.0, 0.0

    order = np.argsort(profile[:, 1])
    z_mm = profile[order, 1]
    r_mm = profile[order, 0]
    z_mm, unique_idx = np.unique(z_mm, return_index=True)
    r_mm = r_mm[unique_idx]
    if z_mm.shape[0] < 3 or float(np.ptp(z_mm)) <= 0:
        return 0.0, 0.0

    volume_uL = float(np.pi * trapezoid(r_mm**2, z_mm))
    dr_dz = np.gradient(r_mm, z_mm, edge_order=2)
    surface_mm2 = float(
        2.0 * np.pi * trapezoid(r_mm * np.sqrt(1.0 + dr_dz**2), z_mm)
    )
    return abs(volume_uL), abs(surface_mm2)


def _append_pendant_dimensionless_numbers(ctx: Context, results: dict) -> None:
    """Populate Bond and Worthington numbers from public pendant outputs."""
    gamma_mn_m = results.get("surface_tension_mN_m")
    r0_mm = results.get("r0_mm")
    volume_uL = results.get("volume_uL")
    if gamma_mn_m is None or r0_mm is None:
        return
    try:
        physics = ctx.physics or {}
        rho1 = float(physics.get("rho1", 1000.0))
        rho2 = float(physics.get("rho2", 1.2))
        g = float(physics.get("g", 9.80665))
        delta_rho = rho1 - rho2
        gamma_n_m = float(gamma_mn_m) / 1000.0
        results["bond_number"] = bond_number_from_gamma(
            delta_rho, g, float(r0_mm), gamma_n_m
        )
        needle_diam_mm = getattr(ctx, "needle_diameter_mm", None)
        if needle_diam_mm and needle_diam_mm > 0 and volume_uL is not None:
            vmax = vmax_uL(gamma_n_m, float(needle_diam_mm), delta_rho, g)
            results["vmax_uL"] = vmax
            results["worthington_number"] = worthington_number(float(volume_uL), vmax)
    except Exception:
        return


def _enabled_pendant_approximators(ctx: Context) -> list[str]:
    methods = getattr(ctx, "pendant_approximation_methods", None)
    if methods is None:
        return list(DEFAULT_PENDANT_APPROXIMATION_METHODS)
    return [str(method) for method in methods if str(method)]


def _run_pendant_approximators(
    ctx: Context, results: dict, profile_mm: np.ndarray | None
) -> None:
    """Run enabled pendant approximation plugins and merge diagnostic keys."""
    if profile_mm is None:
        return
    ctx.results = results
    for name in _enabled_pendant_approximators(ctx):
        fn = registry.PENDANT_APPROXIMATORS.get(name)
        if fn is None:
            results[f"approx_{name}_status"] = "plugin_not_registered"
            continue
        try:
            approx = fn(ctx, profile_mm, ctx.physics or {})
        except Exception as exc:
            results[f"approx_{name}_status"] = "plugin_exception"
            results[f"approx_{name}_error"] = str(exc)
            continue
        if isinstance(approx, dict):
            results.update(approx)


def _promote_pendant_fallback(results: dict) -> None:
    """Use the best available approximation only when strict Y-L was rejected."""
    if results.get("surface_tension_method") == "young_laplace_strict":
        return

    candidates = (
        (
            "multi_selected_plane",
            "approx_multi_selected_plane_surface_tension_mN_m",
            "approx_multi_selected_plane_beta",
            "approx_multi_selected_plane_status",
        ),
        (
            "selected_plane",
            "approx_selected_plane_surface_tension_mN_m",
            "approx_selected_plane_beta",
            "approx_selected_plane_status",
        ),
        (
            "volume_apex_lookup",
            "approx_volume_apex_surface_tension_mN_m",
            "approx_volume_apex_beta",
            "approx_volume_apex_status",
        ),
    )
    for method, gamma_key, beta_key, status_key in candidates:
        gamma = results.get(gamma_key)
        status = results.get(status_key)
        if status != "ok" or gamma is None:
            continue
        try:
            gamma_value = float(gamma)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(gamma_value) or gamma_value <= 0:
            continue
        results["surface_tension_mN_m"] = gamma_value
        beta = results.get(beta_key)
        if beta is not None:
            try:
                beta_value = float(beta)
                if np.isfinite(beta_value) and beta_value > 0:
                    results["beta"] = beta_value
            except (TypeError, ValueError):
                pass
        results["surface_tension_method"] = method
        return


class PendantPipeline(PipelineBase):
    """Pendant drop pipeline (simplified): contour → axis/apex → toy Y–L radius fit."""

    name = "pendant"
    solver_name: str = "young_laplace_ode"
    preprocessor_name: str | None = None
    edge_detector_name: str | None = None

    # UI metadata for plugin-centric configuration
    ui_metadata = {
        "display_name": "Pendant Drop",
        "icon": "pendant.svg",
        "color": "#7ED321",
        "stages": ["acquisition", "contour_extraction", "geometric_features", "physics"],
        "calibration_params": [
            "needle_diameter_mm",
            "drop_density_kg_m3",
            "fluid_density_kg_m3",
        ],
        "primary_metrics": ["surface_tension_mN_m", "volume_uL", "beta"],
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
        from menipy.pipelines.pendant.preprocessing import do_preprocessing
        return do_preprocessing(ctx)

    def do_contour_extraction(self, ctx: Context) -> Optional[Context]:
        """Extract droplet contour using edge detection."""
        detected = getattr(ctx, "drop_contour", None)
        if detected is None:
            detected = getattr(ctx, "detected_contour", None)
        if detected is not None:
            from menipy.models.geometry import Contour

            xy = _contour_to_xy(detected)
            xy = _clip_contour_at_pendant_contacts(
                xy, getattr(ctx, "contact_points", None)
            )
            ctx.contour = Contour(xy=xy)
            return ctx

        if self.edge_detector_name and self.edge_detector_name in registry.EDGE_DETECTORS:
            fn = registry.EDGE_DETECTORS[self.edge_detector_name]
            return fn(ctx) or ctx
        return super().do_contour_extraction(ctx)

    def do_geometric_features(self, ctx: Context) -> Optional[Context]:
        """Extract axis and apex location from contour."""
        xy = ensure_contour(ctx)
        x, y = xy[:, 0], xy[:, 1]
        axis_x = float(np.median(x))
        apex_i = int(np.argmax(y))  # pendant apex at bottom
        apex_xy = (float(x[apex_i]), float(y[apex_i]))

        from menipy.models.geometry import Geometry

        ctx.geometry = Geometry(axis_x=axis_x, apex_xy=apex_xy)
        return ctx

    def do_calibration(self, ctx: Context) -> Optional[Context]:
        """Calculate px_per_mm from needle calibration."""
        if ctx.scale and ctx.scale.get("px_per_mm", 0.0) > 0:
            return ctx

        needle_diameter_mm = getattr(ctx, "needle_diameter_mm", None)
        needle_rect = getattr(ctx, "needle_rect", None)

        px_per_mm = float(getattr(ctx, "px_per_mm", 0.0) or 0.0)

        if px_per_mm <= 0 and needle_diameter_mm and needle_rect:
            # needle_rect is typically (x, y, width, height) in pixels
            try:
                if hasattr(needle_rect, "__iter__") and len(needle_rect) >= 3:
                    needle_width_px = needle_rect[2]  # width in pixels
                    if needle_width_px > 0 and needle_diameter_mm > 0:
                        px_per_mm = float(needle_width_px) / float(needle_diameter_mm)
                        self.logger.info(
                            f"Calibration: needle {needle_width_px}px = {needle_diameter_mm}mm → {px_per_mm:.2f} px/mm"
                        )
            except Exception as e:
                self.logger.warning(f"Could not calculate scale from needle: {e}")
        elif px_per_mm <= 0 and needle_diameter_mm:
            self.logger.warning(
                f"needle_diameter_mm={needle_diameter_mm} but no needle_rect provided for calibration"
            )

        ctx.scale = {"px_per_mm": px_per_mm if px_per_mm > 0 else 1.0}
        return ctx

    def do_physics(self, ctx: Context) -> Optional[Context]:
        ctx.physics = ctx.physics or {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
        return ctx

    def do_profile_fitting(self, ctx: Context) -> Optional[Context]:
        """Fit a calibrated Young-Laplace profile to extract R0 and beta."""
        try:
            xy = ensure_contour(ctx)
            scale = ctx.scale or {}
            px_per_mm = float(scale.get("px_per_mm", 0.0))
            if px_per_mm <= 0:
                raise ValueError("missing positive px_per_mm calibration")

            if ctx.geometry is None or ctx.geometry.axis_x is None:
                self.do_geometric_features(ctx)
            if ctx.geometry is None or ctx.geometry.axis_x is None:
                raise ValueError("missing pendant geometry")

            apex_xy = ctx.geometry.apex_xy
            if apex_xy is None:
                apex_i = int(np.argmax(xy[:, 1]))
                apex_xy = (float(xy[apex_i, 0]), float(xy[apex_i, 1]))

            diameter_px, _ = _pendant_max_width(xy, getattr(ctx, "contact_points", None))
            r0_seed_px = _pendant_apex_radius_px(xy, apex_xy)
            r0_seed_mm = r0_seed_px / px_per_mm if r0_seed_px > 0 else 0.0
            if r0_seed_mm <= 0:
                r0_seed_mm = max((diameter_px / px_per_mm) / 4.0, 0.1)

            beta_seed = 0.3
            if diameter_px > 0 and r0_seed_px > 0:
                beta_seed = float(jennings_pallas_beta(diameter_px / (2.0 * r0_seed_px)))

            needle_radius_mm = None
            needle_diameter_mm = getattr(ctx, "needle_diameter_mm", None)
            if needle_diameter_mm and needle_diameter_mm > 0:
                needle_radius_mm = float(needle_diameter_mm) / 2.0
            elif getattr(ctx, "needle_rect", None) is not None:
                try:
                    needle_width_px = float(ctx.needle_rect[2])
                    if needle_width_px > 0:
                        needle_radius_mm = needle_width_px / px_per_mm / 2.0
                except Exception:
                    needle_radius_mm = None
            elif getattr(ctx, "contact_points", None) is not None:
                try:
                    contacts = np.asarray(ctx.contact_points, dtype=float).reshape(-1, 2)
                    if contacts.shape[0] >= 2:
                        contact_r_px = np.max(np.abs(contacts[:2, 0] - ctx.geometry.axis_x))
                        needle_radius_mm = float(contact_r_px) / px_per_mm
                except Exception:
                    needle_radius_mm = None

            ctx.fit = fit_pendant_young_laplace_strict(
                PendantStrictFitInput(
                    contour_px=xy,
                    axis_x_px=float(ctx.geometry.axis_x),
                    apex_y_px=float(apex_xy[1]),
                    px_per_mm=px_per_mm,
                    r0_seed_mm=r0_seed_mm,
                    beta_seed=beta_seed,
                    physics=ctx.physics or {},
                    needle_radius_mm=needle_radius_mm,
                )
            )
        except Exception as exc:
            ctx.fit = {
                "params": [],
                "param_names": ["r0_mm", "beta", "x_offset_mm", "z_offset_mm"],
                "residuals": {
                    "rmse": float("nan"),
                    "max_abs": float("nan"),
                    "dof": 0,
                    "r": [],
                    "units": "mm",
                },
                "solver": {
                    "backend": "scipy.least_squares",
                    "method": "trf",
                    "iterations": 0,
                    "success": False,
                    "message": str(exc),
                },
                "strict_fit_success": False,
                "strict_fit_warning": "fit_exception",
            }
            self.logger.warning(f"Strict Young-Laplace fit failed: {exc}")
        return ctx

    def do_compute_metrics(self, ctx: Context) -> Optional[Context]:
        """Compute surface tension, volume, and other derived metrics."""
        fit = ctx.fit or {}
        names = fit.get("param_names") or []
        params = fit.get("params", [])

        # Keep profile-fit values as diagnostics. The current fitter compares
        # pixel contour coordinates against a millimetre model, so fitted
        # parameters are not reliable enough for reported pendant metrics.
        results = {"residuals": fit.get("residuals", {})}
        for name, value in zip(names, params):
            results[f"fit_{name}"] = value
        for key in (
            "strict_r0_mm",
            "strict_beta",
            "strict_surface_tension_mN_m",
            "strict_rmse_mm",
            "strict_fit_success",
            "strict_fit_warning",
            "strict_residual_threshold_mm",
            "strict_fit_stop_reason",
            "strict_model_coverage_height_mm",
            "strict_observed_height_mm",
            "strict_observed_diameter_mm",
            "strict_x_offset_mm",
            "strict_z_offset_mm",
        ):
            if key in fit:
                results[key] = fit[key]
        if fit.get("strict_fit_warning"):
            results["fit_warning"] = "young_laplace_fit_unreliable"
        elif _profile_fit_unreliable(fit):
            results["fit_warning"] = "young_laplace_fit_unreliable"

        # --- Geometric Calculations ---
        # 1. Scale
        scale = ctx.scale or {}
        px_per_mm = float(scale.get("px_per_mm", 1.0))

        # 2. Contour Data
        try:
            xy = ensure_contour(ctx)
            x, y = xy[:, 0], xy[:, 1]
            axis_x = ctx.geometry.axis_x if ctx.geometry else np.mean(x)
            apex_xy = ctx.geometry.apex_xy if ctx.geometry else None
            apex_y = float(apex_xy[1]) if apex_xy is not None else float(np.max(y))
            envelope_mm = build_pendant_profile_envelope_mm(
                xy,
                axis_x_px=float(axis_x),
                apex_y_px=apex_y,
                px_per_mm=px_per_mm,
            )

            # 3. Height (mm)
            if envelope_mm.shape[0] >= 3:
                results["height_mm"] = float(np.max(envelope_mm[:, 1]))
            else:
                height_px = np.max(y) - np.min(y)
                results["height_mm"] = height_px / px_per_mm

            # 4. Diameter (mm) from calibrated contour width, not fitted R0.
            diameter_px, diameter_line = _pendant_max_width(
                xy, getattr(ctx, "contact_points", None)
            )
            results["diameter_px"] = diameter_px
            results["diameter_mm"] = diameter_px / px_per_mm if px_per_mm > 0 else 0.0
            results["diameter_line"] = diameter_line

            # Jennings-Pallas geometric surface tension estimate. Keep contour
            # coordinates in pixels for overlays, then calibrate scalar lengths.
            r0_px = _pendant_apex_radius_px(xy, apex_xy)
            if diameter_px > 0 and r0_px > 0 and px_per_mm > 0:
                r0_mm = r0_px / px_per_mm
                s1 = diameter_px / (2.0 * r0_px)
                beta = jennings_pallas_beta(s1)
                results["r0_px"] = r0_px
                results["r0_mm"] = r0_mm
                results["s1"] = s1
                results["beta"] = beta
                results["surface_tension_method"] = "jennings_pallas_geometric"
                results["geometric_r0_mm"] = r0_mm
                results["geometric_beta"] = beta

                if np.isfinite(beta) and abs(beta) > 1e-12:
                    physics = ctx.physics or {}
                    rho1 = float(physics.get("rho1", 1000.0))
                    rho2 = float(physics.get("rho2", 1.2))
                    g = float(physics.get("g", 9.80665))
                    gamma_n_m = surface_tension_n_per_m(
                        rho1 - rho2, g, r0_mm, beta
                    )
                    results["surface_tension_mN_m"] = gamma_n_m * 1000.0
                    results["geometric_surface_tension_mN_m"] = (
                        results["surface_tension_mN_m"]
                    )

            if fit.get("strict_fit_success"):
                strict_r0 = fit.get("strict_r0_mm")
                strict_beta = fit.get("strict_beta")
                strict_gamma = fit.get("strict_surface_tension_mN_m")
                if (
                    strict_r0 is not None
                    and strict_beta is not None
                    and strict_gamma is not None
                ):
                    results["r0_mm"] = float(strict_r0)
                    results["beta"] = float(strict_beta)
                    results["surface_tension_mN_m"] = float(strict_gamma)
                    results["surface_tension_method"] = "young_laplace_strict"
                    results.pop("fit_warning", None)

            profile_for_integrals = None
            if fit.get("strict_fit_success") and fit.get("model_radial_profile_mm"):
                profile_for_integrals = np.asarray(
                    fit.get("model_radial_profile_mm"), dtype=float
                )
            elif envelope_mm.shape[0] >= 3:
                profile_for_integrals = envelope_mm

            if profile_for_integrals is not None:
                volume_uL, drop_surface_mm2 = _radial_profile_integrals(
                    profile_for_integrals
                )
                results["volume_uL"] = volume_uL
                results["drop_surface_mm2"] = drop_surface_mm2
                _run_pendant_approximators(ctx, results, profile_for_integrals)
                _promote_pendant_fallback(results)

            _append_pendant_dimensionless_numbers(ctx, results)

        except Exception as e:
            self.logger.warning(f"Failed to calculate geometric stats: {e}")

        ctx.results = results
        return ctx

    def do_overlay(self, ctx: Context) -> Optional[Context]:
        xy = ensure_contour(ctx)
        if not ctx.geometry:
            return ctx
        axis_x = (
            int(round(ctx.geometry.axis_x)) if ctx.geometry.axis_x is not None else 0
        )
        apex_x, apex_y = (
            ctx.geometry.apex_xy if ctx.geometry.apex_xy is not None else (0, 0)
        )
        text = f"R0≈{ctx.results.get('r0_mm','?')} mm"
        measurement_text = (
            f"Measurement #{ctx.measurement_sequence}"
            if hasattr(ctx, "measurement_sequence") and ctx.measurement_sequence
            else ""
        )
        cmds = [
            # Measurement number overlay
            {
                "type": "text",
                "p": (10, 25),
                "text": measurement_text,
                "color": "white",
                "scale": 0.7,
                "thickness": 2,
            },
            {
                "type": "polyline",
                "points": xy.tolist(),
                "closed": True,
                "color": "yellow",
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
        model_profile_px = (ctx.fit or {}).get("model_profile_px")
        if model_profile_px:
            try:
                model_xy = np.asarray(model_profile_px, dtype=float)
                if model_xy.ndim == 2 and model_xy.shape[0] >= 2:
                    cmds.append(
                        {
                            "type": "polyline",
                            "points": model_xy.tolist(),
                            "closed": False,
                            "color": "green",
                            "thickness": 2,
                        }
                    )
            except Exception:
                pass
        # Store commands for UI-side rendering (e.g., Qt painter toggles)
        ctx.overlay_commands = cmds
        return ovl.run(ctx, commands=cmds, alpha=0.6)
