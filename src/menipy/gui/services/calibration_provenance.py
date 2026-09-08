"""Describe GUI calibration without changing detector or solver thresholds."""

from menipy.models.calibration import CalibrationProvenance


def describe_calibration(parameters, ctx, warnings):
    supplied = parameters.get("calibration_provenance") or {}
    supplied = dict(supplied)
    supplied["component_confidence"] = {
        name: value
        for name, value in supplied.get("component_confidence", {}).items()
        if isinstance(value, (int, float))
    }
    origin = supplied.get("origin")
    scale = (getattr(ctx, "scale", {}) or {}).get("px_per_mm")
    scale = scale or getattr(ctx, "px_per_mm", None)
    scale = scale or (parameters.get("scale") or {}).get("px_per_mm")
    if origin is None:
        origin = (
            "manual" if (parameters.get("scale") or {}).get("px_per_mm") else "missing"
        )
        if parameters.get("needle_rect") and scale:
            origin = "measured"
    if origin == "missing" and getattr(ctx, "dynamic_sessile_result", None) is not None:
        temporal = ctx.dynamic_sessile_result.calibration
        scale = temporal.get("px_per_mm")
        if scale:
            origin = "manual" if temporal.get("method") == "explicit" else "measured"
    combined = list(dict.fromkeys([*supplied.get("warnings", []), *warnings]))
    for component, confidence in supplied.get("component_confidence", {}).items():
        if component != "overall" and confidence < (
            0.75 if component == "substrate" else 0.5
        ):
            combined.append(
                f"Low {component} confidence: {confidence:.0%}; review this component."
            )
    result = CalibrationProvenance(
        origin=origin,
        px_per_mm=scale,
        component_confidence=supplied.get("component_confidence", {}),
        warnings=combined,
    )
    if not result.physical_values_enabled:
        result.warnings.append(
            f"Scale is {origin}; physical values are withheld. Supply valid calibration and rerun."
        )
    return result
