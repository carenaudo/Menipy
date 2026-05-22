"""Basic validation plugin for QA (quality assurance) checks.

Registers a simple validator that sets a basic validation flag in the context.
"""


def basic_validator(ctx):
    """Placeholder docstring for basic_validator.

    TODO: Complete docstring with full description.

    Parameters
    ----------
    ctx : type
        Description of ctx.

    Returns
    -------
    type
        Description of return value.
    """
    """Set a basic validation flag to mark results as valid.

    Parameters
    ----------
    ctx : AnalysisContext
        Context object to validate.

    Returns
    -------
    AnalysisContext
        Context with qa validation flag set.
    """
    # set a simple QA flag
    ctx.qa = getattr(ctx, "qa", {})
    ctx.qa["valid"] = True
    return ctx


VALIDATORS = {"basic": basic_validator}
