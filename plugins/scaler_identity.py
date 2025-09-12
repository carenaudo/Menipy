from menipy.common.registry import register_scaler

def identity_scaler(ctx):
    # no-op scaler: sets ctx.scale if missing
    if not hasattr(ctx, 'scale'):
        ctx.scale = 1.0
    return ctx

register_scaler("identity", identity_scaler)
