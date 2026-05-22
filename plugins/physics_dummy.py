"""Dummy physics plugin for testing physics computation stages.

Registers a placeholder physics handler that provides a mock physics dictionary.
"""


def dummy_physics(ctx):
    """Placeholder docstring for dummy_physics.

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
    """Add placeholder physics metadata to the context.

    Parameters
    ----------
    ctx : AnalysisContext
        Context object to attach physics data to.

    Returns
    -------
    AnalysisContext
        Context with physics dictionary set.
    """
    # attach a placeholder physics dict
    ctx.physics = {"method": "dummy", "params": {}}
    return ctx


PHYSICS = {"dummy": dummy_physics}
