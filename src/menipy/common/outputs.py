# src/adsa/common/outputs.py
"""
Output formatting and normalization for pipeline results.
"""
from __future__ import annotations


def run(ctx):
    """Normalize fit dict to a small 'results' object used by the GUI/CLI."""
    f = ctx.fit or {}
    ctx.results = {
        "gamma_mN_per_m": f.get("gamma_mN_per_m"),
        "param_names": f.get("param_names"),
        "params": f.get("params"),
        "rmse": (f.get("residuals") or {}).get("rmse"),
    }
    return ctx
