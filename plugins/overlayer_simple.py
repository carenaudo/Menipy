"""Simple overlay plugin for adding minimal visual markers to results.

Registers a basic overlay handler that attaches visual metadata to the context.
"""
from menipy.common.registry import register_overlayer


def add_simple_overlay(ctx):
    """Placeholder docstring for add_simple_overlay.
    
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
    """Add a simple text overlay to the analysis context.
    
    Parameters
    ----------
    ctx : AnalysisContext
        Context object to attach overlay metadata to.
    
    Returns
    -------
    AnalysisContext
        Context with overlay list appended.
    """
    # attach a tiny overlay (non-visual for this example)
    ctx.overlay = getattr(ctx, "overlay", None) or []
    ctx.overlay.append({"type": "simple", "text": "overlay"})
    return ctx


register_overlayer("simple", add_simple_overlay)
