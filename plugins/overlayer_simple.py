from menipy.common.registry import register_overlayer

def add_simple_overlay(ctx):
    # attach a tiny overlay (non-visual for this example)
    ctx.overlay = getattr(ctx, 'overlay', None) or []
    ctx.overlay.append({"type": "simple", "text": "overlay"})
    return ctx

register_overlayer("simple", add_simple_overlay)
