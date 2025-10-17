# Results Panel Integration — Pipeline‑Aware Checklist

This guide standardizes how the GUI Results Panel renders results for each pipeline using the schema contracts in `docs/contracts/*_results.md`.

## Goals

- Detect active pipeline and select its schema.
- Render canonical keys with clear labels, units, and precision.
- Show diagnostics when available (residuals, timings, method tags).
- Export CSV/JSON with provenance and schema metadata.

## Prerequisites

- Schema docs present for all pipelines:
  - Pendant: `docs/contracts/pendant_results.md`
  - Sessile: `docs/contracts/sessile_results.md`
  - Captive bubble: `docs/contracts/captive_bubble_results.md`
  - Oscillating: `docs/contracts/oscillating_results.md`
  - Capillary rise: `docs/contracts/capillary_rise_results.md`

## Implementation Checklist

1) Identify the active pipeline
- Source the pipeline name from the execution context (e.g., `ctx.pipeline` or controller parameter). Normalize to lowercase keys: `pendant`, `sessile`, `captive_bubble`, `oscillating`, `capillary_rise`.

2) Select schema map
- Maintain a mapping of pipeline → ordered list of (key, label, unit, precision). Keys must use canonical names from the schema docs.
- Keep a separate list/section for diagnostics (e.g., residuals summary, timings, method/estimator tags).

3) Format values
- Apply unit suffixes and precision per field.
- Hide missing optional fields; display only present keys in schema order.
- For unknown keys in `results`, show them in an “Additional Fields” section with raw values for transparency.

4) Diagnostics section (optional tab or collapsible)
- Pendant: residuals summary, Bond number (if present), stage timings, provenance (image path, px_per_mm).
- Sessile: angle method tag, per‑side uncertainties/RMSE, baseline tilt, timings, provenance.
- Captive bubble: residuals, ceiling tilt/confidence, timings, provenance.
- Oscillating: spectrum stats (SNR, peak width), fps, estimator/window, timings.
- Capillary rise: detection confidences, baseline tilt, timings, provenance.

5) Exports
- CSV/JSON should include: `schema_version` (from the selected schema), `pipeline`, and provenance: calibration (`px_per_mm`), densities/`g` if applicable, method/estimator tags, and `image_path`.
- Ensure stable column order for CSV by following the schema ordering and append diagnostics/provenance columns at the end.

6) Testing
- Add UI smoke tests to verify label/unit switching across pipelines.
- Add serialization tests to validate presence of `schema_version`/`pipeline` and stable column ordering.

## Notes

- Label text for the table should come from a centralized label map per pipeline. Avoid hard‑coding scattered labels in widgets.
- Precision may be user‑configurable in the future; keep a single place to adjust display formatting.
- Keep mapping tolerant to schema growth: ignore unrecognized keys or display them in the Additional Fields section without breaking.

## Minimal Field Sets per Pipeline (for quick reference)

- Pendant: `diameter_mm`, `height_mm`, `r0_mm`, `beta`, `surface_tension_mN_m`, `volume_uL`, `drop_surface_mm2`.
- Sessile: `diameter_mm`, `height_mm`, `theta_left_deg`, `theta_right_deg`, (`contact_angle_deg` legacy), `volume_uL`, `drop_surface_mm2`, `baseline_tilt_deg`.
- Captive bubble: `depth_mm`, `diameter_mm`, `r0_mm`, (`gamma_mN_m` optional), `volume_uL`, `drop_surface_mm2`.
- Oscillating: `f0_Hz`, `r0_eq_mm`, (`gamma_mN_m` optional), plus diagnostics (`fps`, `snr`, `peak_width_Hz`, `n_frames`).
- Capillary rise: `h_mm`, `r_tube_mm`, (`gamma_mN_m`/`theta_deg` optional).

