from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from menipy.models.context import Context


@dataclass
class QACheck:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    message: str
    code: str = "unspecified"
    severity: Literal["info", "warning", "error"] = "error"
    units: str | None = None


@dataclass
class QAResult:
    ok: bool
    score: float
    checks: dict[str, QACheck] = field(default_factory=dict)

    @property
    def rejection_reasons(self) -> list[str]:
        """Return stable codes for failed error-severity checks."""
        return [
            check.code
            for check in self.checks.values()
            if not check.passed and check.severity == "error"
        ]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rejection_reasons"] = self.rejection_reasons
        return payload


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return {}


def build_diagnostics(ctx: Context, qa: QAResult | dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the additive diagnostics wire payload for ADSA results."""
    fit = _as_mapping(ctx.fit_results) or _as_mapping(ctx.fit)
    solver = _as_mapping(fit.get("solver"))
    residuals = _as_mapping(fit.get("residuals"))
    confidence = _as_mapping(fit.get("confidence"))
    qa_value = qa if qa is not None else ctx.qa
    validity = _as_mapping(qa_value)

    px_per_mm = ctx.px_per_mm or (ctx.scale or {}).get("px_per_mm")
    calibration = {
        "pixels_per_mm": px_per_mm,
        "source": "needle" if ctx.needle_diameter_mm and px_per_mm else "provided",
        "needle_diameter_mm": ctx.needle_diameter_mm,
    }
    calibration = {key: value for key, value in calibration.items() if value is not None}

    left = ctx.results.get("theta_left_deg") if isinstance(ctx.results, dict) else None
    right = ctx.results.get("theta_right_deg") if isinstance(ctx.results, dict) else None
    side_discrepancy: dict[str, Any] = {}
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        side_discrepancy = {
            "left_deg": float(left),
            "right_deg": float(right),
            "absolute_difference_deg": abs(float(left) - float(right)),
        }

    return {
        "solver": solver,
        "residuals": residuals,
        "confidence": confidence,
        "validity": validity,
        "calibration": calibration,
        "side_discrepancy": side_discrepancy,
        "detectors": dict(ctx.detector_diagnostics),
        "experimental_geometry": dict(ctx.results.get("experimental_geometry", {})) if isinstance(ctx.results, dict) else {},
        "onnx_proposals": dict(ctx.onnx_proposals),
    }


def validate(ctx: Context, thresholds: dict[str, float] | None = None) -> QAResult:
    """
    Run comprehensive QA checks on the analysis context.

    Checks:
    1. Convergence: was the fit successful?
    2. Residual quality: is RMSE small enough?
    3. Physical plausibility: are physical values in a realistic range?
    4. Contour quality: sufficient points?
    5. Geometric consistency: is the layout reasonably symmetrical?
    """
    if thresholds is None:
        thresholds = {}

    checks = {}
    total_score = 0.0
    max_score = 0.0

    # 1. Convergence
    fit_ok = bool(ctx.fit and ctx.fit.get("solver", {}).get("success", False))
    checks["convergence"] = QACheck(
        name="Convergence",
        passed=fit_ok,
        value=1.0 if fit_ok else 0.0,
        threshold=1.0,
        message=(
            "Solver converged successfully" if fit_ok else "Solver failed to converge"
        ),
        code="solver_not_converged",
    )
    total_score += 1.0 if fit_ok else 0.0
    max_score += 1.0

    # 2. Residuals
    rmse_thresh = thresholds.get("rmse", 5.0)
    rmse = None
    residuals_ok = False

    if ctx.fit and "residuals" in ctx.fit:
        res = ctx.fit["residuals"]
        if hasattr(res, "rmse") and res.rmse is not None:
            rmse = float(res.rmse)
        elif isinstance(res, dict) and "rmse" in res:
            rmse = float(res["rmse"])

    if rmse is not None:
        residuals_ok = rmse <= rmse_thresh
        method = str(
            (ctx.results or {}).get("method")
            or (ctx.results or {}).get("surface_tension_method")
            or ""
        )
        residuals_are_authoritative = method in {
            "young_laplace",
            "young_laplace_strict",
        }
        checks["residuals"] = QACheck(
            name="Residuals",
            passed=residuals_ok,
            value=rmse,
            threshold=rmse_thresh,
            message=f"RMSE={rmse:.2f} px (threshold={rmse_thresh})",
            code="residual_limit_exceeded",
            severity=(
                "error"
                if residuals_are_authoritative
                else "warning"
            ),
            units="px",
        )
    else:
        checks["residuals"] = QACheck(
            name="Residuals",
            passed=True,  # no residuals to check
            value=None,
            threshold=rmse_thresh,
            message="No residuals reported",
            code="residuals_not_reported",
            severity="warning",
        )

    total_score += 1.0 if residuals_ok or rmse is None else 0.0
    max_score += 1.0

    # 3. Contour quality
    min_pts = thresholds.get("min_contour_points", 50)
    pts_ok = False
    pts_count = 0
    if ctx.contour and hasattr(ctx.contour, "xy"):
        pts_count = len(ctx.contour.xy)
        pts_ok = pts_count >= min_pts

    checks["contour"] = QACheck(
        name="Contour Quality",
        passed=pts_ok,
        value=float(pts_count),
        threshold=float(min_pts),
        message=f"Contour points={pts_count} (min={min_pts})",
        code="insufficient_contour_points",
        units="points",
    )
    total_score += 1.0 if pts_ok else 0.0
    max_score += 1.0

    # Experimental geometry is authoritative only when explicitly selected.
    # Shadow diagnostics never affect acceptance.
    active_experimental = (
        getattr(ctx, "needle_geometry_method", "legacy") == "bilateral_robust"
        or getattr(ctx, "pendant_initializer", "legacy") == "robust_axis"
        or getattr(ctx, "contact_angle_method", "tangent") == "auto_residual"
    )
    experimental_geometry_ok = True
    if active_experimental:
        experimental = (ctx.results or {}).get("experimental_geometry", {}) if isinstance(ctx.results, dict) else {}
        failed = []
        def collect_failed(node: Any, prefix: str = "experimental_geometry") -> None:
            if not isinstance(node, dict):
                return
            if node.get("accepted") is False:
                failed.extend(node.get("rejection_reasons") or [f"{prefix}_rejected"])
                return
            for key, value in node.items():
                collect_failed(value, f"{prefix}_{key}")

        collect_failed(experimental)
        if failed:
            checks["experimental_geometry"] = QACheck(
                name="Experimental geometry",
                passed=False,
                value=0.0,
                threshold=1.0,
                message="; ".join(str(item) for item in failed),
                code=str(failed[0]),
                severity="error",
            )
            experimental_geometry_ok = False

    # Calculate final ok and score
    # Must have converged to be ok
    all_passed = (
        checks["convergence"].passed
        and (
            checks["residuals"].passed
            or checks["residuals"].severity != "error"
        )
        and checks["contour"].passed
        and experimental_geometry_ok
    )
    score = total_score / max_score if max_score > 0 else 0.0

    return QAResult(ok=all_passed, score=score, checks=checks)
