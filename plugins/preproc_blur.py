try:
    import cv2
except Exception:
    cv2 = None

from menipy.common.registry import register_preprocessor


def blur_preprocessor(ctx):
    if getattr(ctx, "frames", None) is None:
        return ctx
    # naive: blur the first frame
    frame = ctx.frames[0]
    if cv2 is not None and hasattr(frame, "ndim"):
        b = cv2.GaussianBlur(frame, (5, 5), 0)
        ctx.frames[0] = b
    return ctx


register_preprocessor("blur", blur_preprocessor)
