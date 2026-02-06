"""Image preprocessing plugin that applies Gaussian blur to frames.

Registers a blur preprocessor that applies a Gaussian blur filter to the first frame
if OpenCV is available. Useful for noise reduction before edge detection.
"""
try:
    import cv2
except Exception:
    cv2 = None

from menipy.common.registry import register_preprocessor


def blur_preprocessor(ctx):
    """Placeholder docstring for blur_preprocessor.
    
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
    """Apply Gaussian blur to the first frame for noise reduction.
    
    Parameters
    ----------
    ctx : AnalysisContext
        Context object with frames attribute.
    
    Returns
    -------
    AnalysisContext
        Context with first frame blurred if OpenCV is available.
    """
    if getattr(ctx, "frames", None) is None:
        return ctx
    # naive: blur the first frame
    frame = ctx.frames[0]
    if cv2 is not None and hasattr(frame, "ndim"):
        b = cv2.GaussianBlur(frame, (5, 5), 0)
        ctx.frames[0] = b
    return ctx


register_preprocessor("blur", blur_preprocessor)
