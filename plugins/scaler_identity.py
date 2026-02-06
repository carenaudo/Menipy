"""Identity scaling plugin that provides unit scaling.

Registers a no-op scaler that ensures the context has a scale attribute set to 1.0.
"""
from menipy.common.registry import register_scaler


def identity_scaler(ctx):
    """Placeholder docstring for identity_scaler.
    
    TODO: Complete docstring with full description.
    
    Parameters
    ----------
    ctx : type
        Description of ctx.
    
    Returns
    -------
    type
        Description of return value.
    """
    """Set the scale attribute to 1.0 (identity scaling).
    
    Parameters
    ----------
    ctx : AnalysisContext
        Context object to scale.
    
    Returns
    -------
    AnalysisContext
        Context with scale set to 1.0.
    """
    # no-op scaler: sets ctx.scale if missing
    if not hasattr(ctx, "scale"):
        ctx.scale = 1.0
    return ctx


register_scaler("identity", identity_scaler)
