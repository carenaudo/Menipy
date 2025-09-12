"""Utilities to apply registered plugin functions into pipeline instances.

The main helper `apply_registered_stages(pipeline, merge_strategy='override')`
merges functions from `menipy.common.registry` into the provided pipeline
instance. `merge_strategy` can be:
- 'override': plugin-provided stage replaces existing pipeline stage
- 'prepend': plugin stage is called before the pipeline's stage
- 'append': plugin stage is called after the pipeline's stage

This is intentionally small and conservative: it does not attempt to import
plugins automatically (discovery should be done elsewhere), it simply looks up
registered utilities and wires them onto a PipelineBase instance.
"""
from __future__ import annotations
from typing import Callable

from menipy.common import registry


def _wrap_chain(first: Callable, second: Callable) -> Callable:
    """Return a function that runs first(ctx) then second(ctx).

    Both callables follow the stage hook signature (ctx) -> Optional[ctx]. If
    the first returns a non-None context, that context is passed to the
    second. The wrapper returns the context from the second (or first if second
    returns None).
    """
    def wrapped(ctx):
        c = first(ctx)
        if c is None:
            c = ctx
        return second(c) or c
    return wrapped


def apply_registered_stages(pipeline, merge_strategy: str = "override") -> None:
    """Apply registered stage utilities to a PipelineBase instance.

    This function mutates the pipeline instance in-place. It looks up the
    pipeline name on `pipeline.name` and applies stage callables registered in
    `registry.PIPELINE_STAGES` as well as utilities in the stage-specific
    registries (preprocessors, scalers, etc.) by matching names. The exact
    merge behaviour is controlled via `merge_strategy`.
    """
    name = getattr(pipeline, "name", None)
    if not name:
        return

    # apply per-pipeline stage entries if the registry exposes them
    if hasattr(registry, "PIPELINE_STAGES"):
        stage_map = getattr(registry, "PIPELINE_STAGES").get(name, {})
        for stage_name, fn in stage_map.items():
            attr = f"do_{stage_name}"
            existing = getattr(pipeline, attr, None)
            if existing is None or merge_strategy == "override":
                setattr(pipeline, attr, fn)
            elif merge_strategy == "prepend":
                setattr(pipeline, attr, _wrap_chain(fn, existing))
            elif merge_strategy == "append":
                setattr(pipeline, attr, _wrap_chain(existing, fn))

    # Helper: map registry names to pipeline stage method names
    registry_to_stage = {
        "preprocessors": "preprocessing",
        "acquisitions": "acquisition",
        "geometries": "geometry",
        "scalers": "scaling",
        "physics": "physics",
        "optimizers": "optimization",
        "outputs": "outputs",
        "overlayers": "overlay",
        "validators": "validation",
    }

    snapshot = registry.get_registry_snapshot()
    for reg_name, stage_name in registry_to_stage.items():
        utils = snapshot.get(reg_name, {})
        # if multiple utils exist, apply them in alphabetical order
        for util_name in sorted(utils.keys()):
            fn = utils[util_name]
            attr = f"do_{stage_name}"
            existing = getattr(pipeline, attr, None)
            if existing is None or merge_strategy == "override":
                setattr(pipeline, attr, fn)
            elif merge_strategy == "prepend":
                setattr(pipeline, attr, _wrap_chain(fn, existing))
            elif merge_strategy == "append":
                setattr(pipeline, attr, _wrap_chain(existing, fn))
