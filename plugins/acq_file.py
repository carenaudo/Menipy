from menipy.common.registry import register_acquisition

def acquire_from_dummy(ctx):
    """Simple acquisition that sets ctx.frames from ctx.image_path if present."""
    if getattr(ctx, "image_path", None):
        ctx.frames = [ctx.image_path]
    return ctx

register_acquisition("file", acquire_from_dummy)
