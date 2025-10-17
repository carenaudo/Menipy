# Capillary Rise Pipeline Results Schema

This document defines the capillary rise pipeline results contract used by the GUI Results Panel and export routines. It specifies canonical keys, units, and formatting to ensure consistency across runs and versions.

## Schema Overview

- schema_version: `1.0`
- pipeline: `capillary_rise`
- required keys:
  - `h_mm` (float) — Capillary rise height in millimetres (baseline to meniscus apex, perpendicular to baseline/tube axis as defined).
- recommended keys:
  - `r_tube_mm` (float) — Tube inner radius in millimetres (estimated from detected walls or provided by user).
- optional keys:
  - `gamma_mN_m` (float) — Surface tension in mN/m, inferred via Jurin’s law if `r_tube_mm` and `theta_deg` are known.
  - `theta_deg` (float) — Contact angle in degrees at the tube wall; may be estimated locally or provided.
  - `baseline_tilt_deg` (float) — Estimated tilt of the baseline/tube from horizontal.
  - `residuals` (object) — Implementation‑defined diagnostics (fit quality, detection confidences).
  - `timings_ms` (object) — Per‑stage timings populated by the pipeline runner.
  - `image_path` (string) — Source image path (for provenance in exports).

Notes
- Current implementation may only emit pixel height (`h_px`). Prefer `h_mm`; if only `h_px` is available, display clearly as pixels and avoid mixed units.
- Jurin’s law linkage: `h = 2 γ cosθ / (Δρ g r)`. Inference requires reliable `r_tube_mm` and either `γ` or `θ` to compute the other.

## Units and Formatting

- `h_mm`, `r_tube_mm`: display with 2–3 decimals; export full precision.
- `gamma_mN_m`: display with 1–2 decimals when present.
- `theta_deg`: display with 1 decimal.
- `baseline_tilt_deg`: 2 decimals.

## Example

```json
{
  "schema_version": "1.0",
  "pipeline": "capillary_rise",
  "h_mm": 12.47,
  "r_tube_mm": 0.50,
  "theta_deg": 18.5,
  "gamma_mN_m": 71.6,
  "baseline_tilt_deg": 0.34,
  "residuals": {"apex_rmse_px": 0.7, "wall_fit_conf": 0.92},
  "timings_ms": {"edge_detection": 9.8, "geometry": 6.1}
}
```

## Backward Compatibility

- GUI mapping must tolerate presence of only `h_px` and absence of tube radius/physics outputs; render available fields without errors.
- Increment `schema_version` when changing required fields; preserve canonical key names.

## Provenance and Exports

- CSV/JSON exports SHOULD include `schema_version`, `pipeline`, calibration (`px_per_mm`), densities (`Δρ`) and `g` (if known), tube radius source (detected/provided), and `image_path`.

---

Authoritative consumers
- Results Panel mapping
- Batch export writers
- Tests validating capillary rise results shape

