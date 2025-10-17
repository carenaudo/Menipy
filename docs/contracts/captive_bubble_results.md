# Captive Bubble Pipeline Results Schema

This document defines the captive-bubble pipeline results contract used by the GUI Results Panel and export routines. It normalizes keys, units, and formatting so results are consistent across runs and versions.

## Schema Overview

- schema_version: `1.0`
- pipeline: `captive_bubble`
- required keys:
  - `depth_mm` (float) — Bubble cap depth in millimetres (ceiling to lowest point, measured perpendicular to ceiling).
  - `diameter_mm` (float) — Equatorial diameter of the bubble in millimetres.
  - `r0_mm` (float) — Apex radius of curvature at the lowest point in millimetres.
- optional keys:
  - `gamma_mN_m` (float) — Surface tension in mN/m (if fitted via Young–Laplace).
  - `volume_uL` (float) — Bubble volume in microlitres (by revolution of profile).
  - `drop_surface_mm2` (float) — Bubble surface area in mm² (by revolution of profile).
  - `cap_depth_px` (float) — Cap depth in pixels (legacy/diagnostic; prefer `depth_mm`).
  - `Bo` (float) — Bond number `Δρ g r0² / γ` when `gamma_mN_m` is available.
  - `residuals` (object) — Fit residuals/diagnostics; implementation‑defined structure.
  - `timings_ms` (object) — Per‑stage timings populated by the pipeline runner.
  - `image_path` (string) — Source image path (for provenance in exports).

Notes
- Calibration (`px_per_mm`) and densities (`ρ_liquid`, `ρ_gas`) are required to produce physically meaningful `depth_mm`, `r0_mm`, `gamma_mN_m`.
- The current implementation may emit only pixel depth and `R0_mm` from a toy solver; the canonical schema prefers mm units and physical quantities.

## Units and Formatting

- `depth_mm`, `diameter_mm`, `r0_mm`: display with 2–3 decimals; export full precision.
- `gamma_mN_m`: display with 1–2 decimals when present.
- `volume_uL`: 2–3 decimals; adaptive formatting allowed.

## Example

```json
{
  "schema_version": "1.0",
  "pipeline": "captive_bubble",
  "depth_mm": 1.862,
  "diameter_mm": 3.415,
  "r0_mm": 1.098,
  "gamma_mN_m": 71.2,
  "volume_uL": 14.21,
  "drop_surface_mm2": 52.7,
  "Bo": 0.63,
  "residuals": {"rms": 0.19, "n": 480},
  "timings_ms": {"edge_detection": 10.8, "geometry": 5.2, "solver": 41.9}
}
```

## Backward Compatibility

- GUI mapping must tolerate missing optional fields and unknown extras.
- If only pixel depth is available (`cap_depth_px`), display it clearly as pixels and avoid mixing units; prefer mm when calibration is present.
- If future versions introduce new keys, increment `schema_version` and keep original keys stable.

## Provenance and Exports

- CSV/JSON exports SHOULD include `schema_version`, `pipeline`, calibration (`px_per_mm`), densities/temperature (if available), and `image_path`.

---

Authoritative consumers
- Results Panel mapping
- Batch export writers
- Tests validating captive-bubble results shape

