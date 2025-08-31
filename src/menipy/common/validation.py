# src/adsa/common/validation.py
from __future__ import annotations

def run(ctx):
    """Basic QA gates; expand with Bond number / residual thresholds."""
    r = (ctx.fit or {}).get("residuals", {})
    ctx.qa = {
        "rmse_ok": r.get("rmse", 1e9) < 5.0,
        "max_abs_ok": r.get("max_abs", 1e9) < 10.0,
    }
    return ctx
