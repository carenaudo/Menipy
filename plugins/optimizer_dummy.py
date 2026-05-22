def noop_optimizer(ctx):
    """No-op optimizer for testing purposes. Simply marks context as optimized."""
    # simple optimizer stub: no-op but attaches an 'optimized' flag
    ctx.optimized = True
    return ctx


OPTIMIZERS = {"noop": noop_optimizer}
