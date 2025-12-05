from __future__ import annotations

from menipy.common.registry import register_acquisition
from menipy.common import acquisition as acq

try:
    from menipy.models.frame import Frame
except Exception:  # pragma: no cover - optional import
    Frame = None  # type: ignore


def acquire_from_file(ctx):
    """
    Load image frames from disk so downstream stages receive numpy arrays
    instead of raw string paths.
    """
    path = getattr(ctx, "image_path", None) or getattr(ctx, "image", None)
    if not path:
        return ctx

    frames = acq.from_file([path])
    if not frames:
        return ctx

    # Store numpy frames in the context
    ctx.frames = frames
    first = frames[0]
    ctx.frame = first
    ctx.image = first
    if Frame is not None:
        ctx.current_frame = Frame(image=first)
    return ctx


register_acquisition("file", acquire_from_file)
