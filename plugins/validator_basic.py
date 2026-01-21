from menipy.common.registry import register_validator


def basic_validator(ctx):
    # set a simple QA flag
    ctx.qa = getattr(ctx, "qa", {})
    ctx.qa["valid"] = True
    return ctx


register_validator("basic", basic_validator)
