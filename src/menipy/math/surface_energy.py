"""Surface free energy calculation — OWRK and Wu harmonic mean methods.

Implements:
- **OWRK** (Owens–Wendt–Rabel–Kaelble): Rabel linearisation with ordinary
  least-squares regression.
- **Wu** harmonic mean: non-linear 2-liquid system solved with
  ``scipy.optimize.fsolve``.

Both methods decompose the solid surface free energy into dispersive (γ_S^d)
and polar (γ_S^p) components.

References
----------
Owens, D. K. & Wendt, R. C. (1969). J. Appl. Polym. Sci. 13, 1741–1747.
Rabel, W. (1971). Farbe und Lack 77(10), 997–1005.
Wu, S. (1982). Polymer Interface and Adhesion, Marcel Dekker.
ISO 19403-2 — Wettability — Part 2: Surface free energy.
DIN 55660-2 — Wettability — Part 2: OWRK method.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from menipy.common.liquid_db import ProbeLiquid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class OWRKResult:
    """Result container for the OWRK (Rabel) linear regression."""

    gamma_s_d: float
    """Dispersive component of solid SFE in mN/m."""
    gamma_s_p: float
    """Polar component of solid SFE in mN/m."""
    gamma_s_total: float
    """Total solid surface free energy (γ_S^d + γ_S^p) in mN/m."""

    slope: float
    """Regression slope m = √(γ_S^p)."""
    intercept: float
    """Regression intercept b = √(γ_S^d)."""

    r_squared: float | None = None
    """Coefficient of determination (None when N = 2)."""

    se_slope: float | None = None
    """Standard error of the slope."""
    se_intercept: float | None = None
    """Standard error of the intercept."""
    se_gamma_s_d: float | None = None
    """Propagated standard error of γ_S^d."""
    se_gamma_s_p: float | None = None
    """Propagated standard error of γ_S^p."""
    se_gamma_s_total: float | None = None
    """Propagated standard error of total SFE."""

    x_coords: list[float] = field(default_factory=list)
    """Per-liquid OWRK x-coordinates: √(γ_L^p / γ_L^d)."""
    y_coords: list[float] = field(default_factory=list)
    """Per-liquid OWRK y-coordinates: γ_L(1+cosθ) / (2√γ_L^d)."""
    y_predicted: list[float] = field(default_factory=list)
    """Regression-line predicted y-values."""
    residuals: list[float] = field(default_factory=list)
    """Per-liquid residuals (y_obs − y_pred)."""

    liquid_names: list[str] = field(default_factory=list)
    """Names of probe liquids used."""
    contact_angles_deg: list[float] = field(default_factory=list)
    """Measured contact angles (degrees) per liquid."""
    warnings: list[str] = field(default_factory=list)
    """Validation warnings (e.g. ``"negative_slope_constrained"``)."""

    method: str = "owrk"


@dataclass
class WuResult:
    """Result container for the Wu harmonic mean method."""

    gamma_s_d: float
    """Dispersive component of solid SFE in mN/m."""
    gamma_s_p: float
    """Polar component of solid SFE in mN/m."""
    gamma_s_total: float
    """Total solid SFE in mN/m."""

    liquid_names: list[str] = field(default_factory=list)
    """Names of the two probe liquids used."""
    contact_angles_deg: list[float] = field(default_factory=list)
    """Measured contact angles (degrees) for the two liquids."""
    warnings: list[str] = field(default_factory=list)
    """Validation warnings."""

    method: str = "wu"


@dataclass
class SFEResult:
    """Combined result container holding OWRK and/or Wu results."""

    owrk: OWRKResult | None = None
    wu: WuResult | None = None
    substrate_name: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON export."""
        result: dict = {
            "substrate": self.substrate_name,
            "warnings": self.warnings,
        }
        if self.owrk is not None:
            result["owrk"] = {
                "gamma_s_dispersive_mN_m": round(self.owrk.gamma_s_d, 2),
                "gamma_s_polar_mN_m": round(self.owrk.gamma_s_p, 2),
                "gamma_s_total_mN_m": round(self.owrk.gamma_s_total, 2),
                "slope": round(self.owrk.slope, 4),
                "intercept": round(self.owrk.intercept, 4),
                "r_squared": (
                    round(self.owrk.r_squared, 4)
                    if self.owrk.r_squared is not None
                    else None
                ),
                "se_gamma_s_dispersive": (
                    round(self.owrk.se_gamma_s_d, 3)
                    if self.owrk.se_gamma_s_d is not None
                    else None
                ),
                "se_gamma_s_polar": (
                    round(self.owrk.se_gamma_s_p, 3)
                    if self.owrk.se_gamma_s_p is not None
                    else None
                ),
                "se_gamma_s_total": (
                    round(self.owrk.se_gamma_s_total, 3)
                    if self.owrk.se_gamma_s_total is not None
                    else None
                ),
                "data_points": [
                    {
                        "liquid": name,
                        "contact_angle_deg": round(ca, 2),
                        "x": round(x, 4),
                        "y": round(y, 4),
                        "y_predicted": round(yp, 4),
                        "residual": round(r, 4),
                    }
                    for name, ca, x, y, yp, r in zip(
                        self.owrk.liquid_names,
                        self.owrk.contact_angles_deg,
                        self.owrk.x_coords,
                        self.owrk.y_coords,
                        self.owrk.y_predicted,
                        self.owrk.residuals,
                    )
                ],
                "warnings": self.owrk.warnings,
            }
        if self.wu is not None:
            result["wu"] = {
                "gamma_s_dispersive_mN_m": round(self.wu.gamma_s_d, 2),
                "gamma_s_polar_mN_m": round(self.wu.gamma_s_p, 2),
                "gamma_s_total_mN_m": round(self.wu.gamma_s_total, 2),
                "liquid_names": self.wu.liquid_names,
                "contact_angles_deg": [round(a, 2) for a in self.wu.contact_angles_deg],
                "warnings": self.wu.warnings,
            }
        return result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_sfe_inputs(
    liquids: list[ProbeLiquid],
    contact_angles_deg: list[float],
) -> list[str]:
    """Validate surface energy inputs and return warning/error messages.

    Parameters
    ----------
    liquids : list[ProbeLiquid]
        Probe liquids (must have matching length with *contact_angles_deg*).
    contact_angles_deg : list[float]
        Measured advancing contact angles in degrees.

    Returns
    -------
    list[str]
        List of warning or error strings.  An empty list indicates valid input.
    """
    msgs: list[str] = []

    if len(liquids) != len(contact_angles_deg):
        msgs.append(
            f"Mismatch: {len(liquids)} liquids but "
            f"{len(contact_angles_deg)} contact angles"
        )
        return msgs

    if len(liquids) < 2:
        msgs.append("At least 2 probe liquids are required for SFE analysis")
        return msgs

    # Angle range
    for _i, (liq, ca) in enumerate(zip(liquids, contact_angles_deg)):
        if ca <= 0.0 or ca >= 180.0:
            msgs.append(
                f"{liq.name}: contact angle {ca}° is outside valid range (0°, 180°)"
            )
        if ca > 170.0:
            msgs.append(f"{liq.name}: contact angle {ca}° is near 180° — check data")

    # Duplicate liquids
    names = [liq.name.lower() for liq in liquids]
    if len(names) != len(set(names)):
        msgs.append("Duplicate liquid names detected — each liquid should appear once")

    # Polarity span check
    x_vals = [liq.owrk_x for liq in liquids]
    has_dispersive = any(x < 0.1 for x in x_vals)
    has_polar = any(x > 1.0 for x in x_vals)
    if not has_dispersive:
        msgs.append(
            "No purely dispersive liquid (x ≈ 0) included — OWRK intercept "
            "will be poorly constrained.  Consider adding diiodomethane or "
            "1-bromonaphthalene."
        )
    if not has_polar:
        msgs.append(
            "No highly polar liquid (x > 1.0) included — OWRK slope will be "
            "poorly constrained.  Consider adding water."
        )

    # Per-liquid consistency
    for liq in liquids:
        for w in liq.validate():
            msgs.append(w)

    return msgs


# ---------------------------------------------------------------------------
# OWRK (Rabel linearisation + OLS)
# ---------------------------------------------------------------------------


def compute_owrk(
    liquids: list[ProbeLiquid],
    contact_angles_deg: list[float],
) -> OWRKResult:
    """Compute solid surface energy via the OWRK/Rabel method.

    Parameters
    ----------
    liquids : list[ProbeLiquid]
        Probe liquids with known γ_L, γ_L^d, γ_L^p.
    contact_angles_deg : list[float]
        Measured advancing contact angles (degrees), one per liquid.

    Returns
    -------
    OWRKResult
        Regression results including γ_S^d, γ_S^p, γ_S, R², and uncertainties.

    Raises
    ------
    ValueError
        If fewer than 2 liquids are provided.
    """
    n = len(liquids)
    if n < 2:
        raise ValueError("OWRK requires at least 2 probe liquids")
    if n != len(contact_angles_deg):
        raise ValueError(
            f"Length mismatch: {n} liquids vs {len(contact_angles_deg)} angles"
        )

    warnings: list[str] = []

    # Compute Rabel coordinates
    x_arr = np.empty(n, dtype=np.float64)
    y_arr = np.empty(n, dtype=np.float64)

    for i, (liq, theta_deg) in enumerate(zip(liquids, contact_angles_deg)):
        theta_rad = math.radians(theta_deg)
        cos_theta = math.cos(theta_rad)

        if liq.gamma_d <= 0:
            raise ValueError(
                f"{liq.name}: dispersive component γ_d must be positive (got {liq.gamma_d})"
            )

        x_arr[i] = (liq.gamma_p / liq.gamma_d) ** 0.5
        y_arr[i] = liq.gamma_total * (1.0 + cos_theta) / (2.0 * liq.gamma_d**0.5)

    # OLS linear regression: y = m*x + b
    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)
    ss_xx = np.sum((x_arr - x_mean) ** 2)
    ss_xy = np.sum((x_arr - x_mean) * (y_arr - y_mean))

    if ss_xx < 1e-12:
        raise ValueError(
            "All probe liquids have the same OWRK x-coordinate — "
            "linear regression is degenerate.  Include liquids with "
            "different polarity ratios."
        )

    m = float(ss_xy / ss_xx)
    b = float(y_mean - m * x_mean)

    # Handle negative slope (non-physical polar component)
    negative_slope_constrained = False
    if m < 0:
        warnings.append(
            "Negative OWRK slope detected (m < 0): polar component is "
            "non-physical.  Setting γ_S^p = 0 and re-fitting intercept."
        )
        negative_slope_constrained = True
        m = 0.0
        b = float(y_mean)  # horizontal line through mean

    # Handle negative intercept
    if b < 0:
        warnings.append(
            f"Negative OWRK intercept (b = {b:.3f}): dispersive component "
            f"would be non-physical.  This indicates invalid input data or "
            f"gross measurement error."
        )

    # Surface energy components
    gamma_s_p = m * m
    gamma_s_d = b * b if b >= 0 else 0.0
    gamma_s_total = gamma_s_d + gamma_s_p

    # Predicted values and residuals
    y_pred = m * x_arr + b
    residuals = y_arr - y_pred

    # Quality metrics (only meaningful for N >= 3)
    r_squared: float | None = None
    se_m: float | None = None
    se_b: float | None = None
    se_gsd: float | None = None
    se_gsp: float | None = None
    se_gs: float | None = None

    if n >= 3 and not negative_slope_constrained:
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_arr - y_mean) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None

        if r_squared is not None and r_squared < 0.90:
            warnings.append(
                f"Low R² = {r_squared:.3f} (< 0.90) — check for contamination, "
                f"droplet evaporation, or surface heterogeneity."
            )

        # Standard errors
        if n > 2:
            s_yx = (ss_res / (n - 2)) ** 0.5
            se_m = float(s_yx / ss_xx**0.5)
            se_b = float(s_yx * (1.0 / n + x_mean**2 / ss_xx) ** 0.5)

            # Propagated uncertainties: γ_S^p = m², γ_S^d = b²
            se_gsp = 2.0 * abs(m) * se_m if m != 0 else se_m
            se_gsd = 2.0 * abs(b) * se_b if b > 0 else se_b

            # Covariance(b, m) = -x̄ · SE(m)²
            cov_bm = -x_mean * se_m**2
            se_gs = (se_gsd**2 + se_gsp**2 + 8.0 * abs(b * m) * cov_bm) ** 0.5
            # Guard against sqrt of negative (can happen with extreme covariance)
            var_gs = se_gsd**2 + se_gsp**2 + 8.0 * abs(b * m) * cov_bm
            se_gs = var_gs**0.5 if var_gs > 0 else 0.0

    return OWRKResult(
        gamma_s_d=gamma_s_d,
        gamma_s_p=gamma_s_p,
        gamma_s_total=gamma_s_total,
        slope=m,
        intercept=b,
        r_squared=r_squared,
        se_slope=se_m,
        se_intercept=se_b,
        se_gamma_s_d=se_gsd,
        se_gamma_s_p=se_gsp,
        se_gamma_s_total=se_gs,
        x_coords=x_arr.tolist(),
        y_coords=y_arr.tolist(),
        y_predicted=y_pred.tolist(),
        residuals=residuals.tolist(),
        liquid_names=[liq.name for liq in liquids],
        contact_angles_deg=list(contact_angles_deg),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Wu harmonic mean
# ---------------------------------------------------------------------------


def compute_wu(
    liquids: list[ProbeLiquid],
    contact_angles_deg: list[float],
) -> WuResult:
    """Compute solid surface energy via the Wu harmonic mean method.

    The harmonic mean method requires exactly 2 probe liquids and solves
    a system of 2 non-linear equations:

        γ_L(1 + cosθ) = 4·[γ_S^d·γ_L^d/(γ_S^d + γ_L^d)
                          + γ_S^p·γ_L^p/(γ_S^p + γ_L^p)]

    Parameters
    ----------
    liquids : list[ProbeLiquid]
        Exactly 2 probe liquids.  If more are provided, only the first 2 are
        used and a warning is emitted.
    contact_angles_deg : list[float]
        Contact angles in degrees (one per liquid).

    Returns
    -------
    WuResult
        Surface energy components from the harmonic mean method.
    """
    from scipy.optimize import fsolve

    warnings: list[str] = []

    if len(liquids) < 2:
        raise ValueError("Wu harmonic mean requires at least 2 probe liquids")

    if len(liquids) > 2:
        warnings.append(
            f"Wu harmonic mean uses exactly 2 liquids; only the first 2 of "
            f"{len(liquids)} will be used."
        )

    liq1, liq2 = liquids[0], liquids[1]
    ca1, ca2 = contact_angles_deg[0], contact_angles_deg[1]

    wa1 = liq1.gamma_total * (1.0 + math.cos(math.radians(ca1)))
    wa2 = liq2.gamma_total * (1.0 + math.cos(math.radians(ca2)))

    def _system(params: np.ndarray) -> list[float]:
        gs_d, gs_p = float(params[0]), float(params[1])
        # Guard against zero denominators
        denom_d1 = gs_d + liq1.gamma_d
        denom_p1 = gs_p + liq1.gamma_p
        denom_d2 = gs_d + liq2.gamma_d
        denom_p2 = gs_p + liq2.gamma_p

        # Avoid division by zero for purely dispersive liquids (gamma_p = 0)
        harm_d1 = (4.0 * gs_d * liq1.gamma_d / denom_d1) if denom_d1 > 1e-12 else 0.0
        harm_p1 = (4.0 * gs_p * liq1.gamma_p / denom_p1) if denom_p1 > 1e-12 else 0.0
        harm_d2 = (4.0 * gs_d * liq2.gamma_d / denom_d2) if denom_d2 > 1e-12 else 0.0
        harm_p2 = (4.0 * gs_p * liq2.gamma_p / denom_p2) if denom_p2 > 1e-12 else 0.0

        eq1 = harm_d1 + harm_p1 - wa1
        eq2 = harm_d2 + harm_p2 - wa2
        return [eq1, eq2]

    # Initial guess: mid-range values
    x0 = np.array([20.0, 10.0])
    solution, info, ier, mesg = fsolve(_system, x0, full_output=True)

    gs_d_sol, gs_p_sol = float(solution[0]), float(solution[1])

    if ier != 1:
        warnings.append(f"Wu solver did not converge: {mesg}")

    # Physical validity check: try alternative starting point if negative
    if gs_d_sol < 0 or gs_p_sol < 0:
        # Retry with different initial guess
        for x0_alt in [np.array([40.0, 5.0]), np.array([10.0, 20.0]),
                        np.array([30.0, 1.0])]:
            sol2, _, ier2, _ = fsolve(_system, x0_alt, full_output=True)
            gd2, gp2 = float(sol2[0]), float(sol2[1])
            if ier2 == 1 and gd2 >= 0 and gp2 >= 0:
                gs_d_sol, gs_p_sol = gd2, gp2
                break

    # Select physically valid root
    if gs_d_sol < 0:
        warnings.append(
            f"Wu dispersive component is negative ({gs_d_sol:.2f} mN/m) — "
            f"non-physical result, check input data."
        )
        gs_d_sol = 0.0
    if gs_p_sol < 0:
        warnings.append(
            f"Wu polar component is negative ({gs_p_sol:.2f} mN/m) — "
            f"non-physical result, check input data."
        )
        gs_p_sol = 0.0

    # Validate that dispersive component doesn't exceed max liquid dispersive
    max_liq_d = max(liq1.gamma_d, liq2.gamma_d)
    if gs_d_sol > max_liq_d * 2:
        warnings.append(
            f"Wu dispersive component ({gs_d_sol:.1f} mN/m) is unusually "
            f"high relative to probe liquids — check measurement quality."
        )

    return WuResult(
        gamma_s_d=gs_d_sol,
        gamma_s_p=gs_p_sol,
        gamma_s_total=gs_d_sol + gs_p_sol,
        liquid_names=[liq1.name, liq2.name],
        contact_angles_deg=[ca1, ca2],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Combined convenience function
# ---------------------------------------------------------------------------


def compute_surface_energy(
    liquids: list[ProbeLiquid],
    contact_angles_deg: list[float],
    *,
    method: str = "both",
    substrate_name: str | None = None,
) -> SFEResult:
    """Compute solid surface free energy using specified method(s).

    Parameters
    ----------
    liquids : list[ProbeLiquid]
        Probe liquids with known surface tension components.
    contact_angles_deg : list[float]
        Measured advancing contact angles (degrees).
    method : str
        ``"owrk"``, ``"wu"``, or ``"both"`` (default).
    substrate_name : str | None
        Optional name for the substrate (used in reporting).

    Returns
    -------
    SFEResult
        Combined result container.
    """
    all_warnings = validate_sfe_inputs(liquids, contact_angles_deg)

    owrk_result: OWRKResult | None = None
    wu_result: WuResult | None = None

    if method in ("owrk", "both"):
        try:
            owrk_result = compute_owrk(liquids, contact_angles_deg)
        except ValueError as exc:
            all_warnings.append(f"OWRK computation failed: {exc}")

    if method in ("wu", "both"):
        if len(liquids) >= 2:
            try:
                wu_result = compute_wu(liquids, contact_angles_deg)
            except ValueError as exc:
                all_warnings.append(f"Wu computation failed: {exc}")
        else:
            all_warnings.append("Wu method requires at least 2 liquids")

    return SFEResult(
        owrk=owrk_result,
        wu=wu_result,
        substrate_name=substrate_name,
        warnings=all_warnings,
    )
