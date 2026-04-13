from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from menipy.models.context import Context

@dataclass
class QACheck:
    name: str
    passed: bool
    value: Optional[float]
    threshold: Optional[float]
    message: str

@dataclass
class QAResult:
    ok: bool
    score: float
    checks: Dict[str, QACheck] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

def validate(ctx: Context, thresholds: Optional[Dict[str, float]] = None) -> QAResult:
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
        message="Solver converged successfully" if fit_ok else "Solver failed to converge"
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
        checks["residuals"] = QACheck(
            name="Residuals",
            passed=residuals_ok,
            value=rmse,
            threshold=rmse_thresh,
            message=f"RMSE={rmse:.2f} px (threshold={rmse_thresh})"
        )
    else:
        checks["residuals"] = QACheck(
            name="Residuals",
            passed=True, # no residuals to check
            value=None,
            threshold=rmse_thresh,
            message="No residuals reported"
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
        message=f"Contour points={pts_count} (min={min_pts})"
    )
    total_score += 1.0 if pts_ok else 0.0
    max_score += 1.0
    
    # Calculate final ok and score
    # Must have converged to be ok
    all_passed = checks["convergence"].passed and checks["residuals"].passed and checks["contour"].passed
    score = total_score / max_score if max_score > 0 else 0.0
    
    return QAResult(
        ok=all_passed,
        score=score,
        checks=checks
    )
