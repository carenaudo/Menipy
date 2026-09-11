"""Two-zone clothoid spline in the pendant pipeline.

Runs :func:`menipy.common.pendant_spline.fit_pendant_contour` on the pipeline
contour and contact points, refines the edge on the image when the contour
lies inside it, and exposes the result two ways:

* as the ``clothoid_zones`` contour model of the strict Young-Laplace fit
  (``Context.pendant_contour_model``): the strict fit then runs on the spline's
  samples, seeded with the anchored profile's apex radius, Bond number and
  symmetry axis;
* as the ``clothoid_zones`` pendant approximator: the anchored profile's
  surface tension, the spline's curvature (Laplace-slope) surface tension and
  the needle angle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from menipy.common.pendant_spline import (
    PendantSplineFit,
    fit_pendant_contour,
    refine_pendant_on_image,
)
from menipy.common.registry import register_pendant_approximator
from menipy.pipelines.utils import image_for_contour

CONTOUR_MODELS = ("raw", "clothoid_zones")
STRICT_SAMPLE_STEP_PX = 3.0


def _physics(physics: dict[str, Any] | None) -> tuple[float, float]:
    physics = physics or {}
    rho1 = float(physics.get("rho1", 1000.0))
    rho2 = float(physics.get("rho2", 1.2))
    return rho1 - rho2, float(physics.get("g", 9.80665))


def fit_clothoid_zones(ctx: Any, contour_xy: np.ndarray, px_per_mm: float) -> tuple[PendantSplineFit | None, dict[str, Any]]:
    """Fit the two-zone spline to a pipeline contour.

    Parameters
    ----------
    ctx : Context
        Pipeline context; ``contact_points`` must hold the two needle contacts.
    contour_xy : np.ndarray
        Drop contour in image pixels, shape ``(N, 2)``.
    px_per_mm : float
        Image scale.

    Returns
    -------
    fit : PendantSplineFit or None
        The fitted model, or ``None`` when it could not be fitted.
    diagnostics : dict
        JSON-serializable summary; ``accepted`` is ``False`` with
        ``rejection_reasons`` when the fit failed or was rejected.
    """
    contacts = getattr(ctx, "contact_points", None)
    try:
        contacts = np.asarray(contacts, float).reshape(-1, 2)[:2] if contacts is not None else None
    except (TypeError, ValueError):
        contacts = None
    if contacts is None or len(contacts) < 2:
        return None, {"accepted": False, "rejection_reasons": ["missing_contact_points"]}
    delta_rho, g = _physics(getattr(ctx, "physics", None))
    kwargs = {"px_per_mm": px_per_mm, "delta_rho": delta_rho, "g": g}
    xy = np.asarray(contour_xy, float).reshape(-1, 2)
    try:
        fit, _ = fit_pendant_contour(xy, contacts[0], contacts[1], **kwargs)
        image = image_for_contour(ctx, xy)
        if image is not None:
            fit, _ = refine_pendant_on_image(image, fit, contacts[0], contacts[1], **kwargs)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, {"accepted": False, "rejection_reasons": ["fit_failed"], "error": str(exc)}
    diagnostics = fit.to_diagnostics()
    diagnostics["image_refined"] = image is not None
    diagnostics["model_contour_xy"] = fit.sample(2.0).tolist()
    return fit, diagnostics


def strict_inputs_from_fit(fit: PendantSplineFit, px_per_mm: float) -> dict[str, Any]:
    """Contour, seeds and axis for the strict fit from an accepted spline.

    Parameters
    ----------
    fit : PendantSplineFit
        Accepted two-zone fit.
    px_per_mm : float
        Image scale.

    Returns
    -------
    dict
        Keyword arguments for ``PendantStrictFitInput`` (everything but the
        physics and needle radius).
    """
    point, up = fit.axis_image
    return {
        "contour_px": fit.sample(STRICT_SAMPLE_STEP_PX),
        "axis_x_px": float(point[0]),
        "apex_y_px": float(point[1]),
        "px_per_mm": px_per_mm,
        "r0_seed_mm": fit.shape.apex_radius_px / px_per_mm,
        "beta_seed": fit.shape.bond,
        "axis_origin_px": (float(point[0]), float(point[1])),
        "axis_direction_xy": (float(up[0]), float(up[1])),
    }


def clothoid_zones(ctx: Any, profile_mm: np.ndarray, physics: dict[str, Any]) -> dict[str, Any]:
    """Surface tension and needle angle from the two-zone clothoid spline.

    Reuses the fit of the ``clothoid_zones`` contour model when the profile
    stage ran it; otherwise fits the context contour.

    Parameters
    ----------
    ctx : Context
        Pipeline context (contour, contact points, scale).
    profile_mm : np.ndarray
        Radial profile (unused: the spline needs both sides of the image contour).
    physics : dict
        Densities and gravity.

    Returns
    -------
    dict
        ``approx_clothoid_zones_*`` keys.
    """
    prefix = "approx_clothoid_zones"
    px_per_mm = float((getattr(ctx, "scale", None) or {}).get("px_per_mm", 0.0) or 0.0)
    diag = (getattr(ctx, "fit", None) or {}).get("clothoid_zones")
    if not diag or "surface_tension_mN_m" not in diag:
        contour = getattr(ctx, "contour", None)
        if contour is None or getattr(contour, "xy", None) is None or px_per_mm <= 0:
            return {f"{prefix}_status": "missing_contour_or_scale"}
        _, diag = fit_clothoid_zones(ctx, np.asarray(contour.xy, float), px_per_mm)
    if "surface_tension_mN_m" not in diag:
        return {f"{prefix}_status": "fit_failed",
                f"{prefix}_rejection_reasons": list(diag.get("rejection_reasons") or [])}
    laplace = diag.get("laplace") or {}
    return {
        f"{prefix}_status": "ok" if diag.get("accepted") else "rejected",
        f"{prefix}_rejection_reasons": list(diag.get("rejection_reasons") or []),
        f"{prefix}_surface_tension_mN_m": diag["surface_tension_mN_m"],
        f"{prefix}_beta": diag.get("bond"),
        f"{prefix}_r0_mm": diag["apex_radius_px"] / px_per_mm if px_per_mm > 0 else None,
        f"{prefix}_laplace_surface_tension_mN_m": laplace.get("surface_tension_mN_m"),
        f"{prefix}_needle_angle_deg": diag.get("needle_angle_deg"),
    }


register_pendant_approximator("clothoid_zones", clothoid_zones)
