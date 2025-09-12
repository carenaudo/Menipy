from menipy.common.registry import register_physics

def dummy_physics(ctx):
    # attach a placeholder physics dict
    ctx.physics = {"method": "dummy", "params": {}}
    return ctx

register_physics("dummy", dummy_physics)
