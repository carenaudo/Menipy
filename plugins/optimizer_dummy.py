from menipy.common.registry import register_optimizer


def noop_optimizer(ctx):
    """No-op optimizer for testing purposes. Simply marks context as optimized."""
    # simple optimizer stub: no-op but attaches an 'optimized' flag
    ctx.optimized = True
    return ctx


register_optimizer("noop", noop_optimizer)
