from menipy.common.registry import register_optimizer


def noop_optimizer(ctx):
    # simple optimizer stub: no-op but attaches an 'optimized' flag
    ctx.optimized = True
    return ctx


register_optimizer("noop", noop_optimizer)
