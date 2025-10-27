# Sessile Pipeline Results Schema

This document defines the sessile pipeline results contract used by the GUI Results Panel and export routines. It normalizes keys, units, and formatting for clarity and downstream compatibility.

## Schema Overview

- schema_version: `1.0`
- pipeline: `sessile`
- required keys (baseline):
  - `diameter_mm` (float) — Base diameter in millimetres (distance between contact points along substrate).
  - `height_mm` (float) — Height in millimetres (apex to substrate distance, perpendicular to substrate).
  - `theta_left_deg` (float) — Left contact angle in degrees (tangent method preferred).
  - `theta_right_deg` (float) — Right contact angle in degrees.
  - `volume_uL` (float) — Volume in microlitres.
  - `drop_surface_mm2` (float) — Drop surface area in mm².
  - `baseline_tilt_deg` (float) — Estimated substrate tilt in degrees.
  - `method` (string) — Angle method tag: `tangent`, `spherical_cap`, `circle_fit`, `young_laplace`.
- optional keys:
  - `contact_angle_deg` (float) — Single angle from spherical‑cap approximation (legacy/quick estimate).
  - `uncertainty_deg` (object) — Angle uncertainty per side: `{ "left": x, "right": y }`.
  - `contact_line` (object) — Contact line endpoints: `[[x1, y1], [x2, y2]]`.
  - `diameter_line` (object) — Visualization line; same format as `contact_line`.
  - `timings_ms` (object) — Per‑stage timings populated by the pipeline runner.
  - `image_path` (string) — Source image path (for provenance in exports).
  - `diagnostics` (object) — Optional diagnostics including residuals, fit quality metrics, etc.

Notes
- Current implementation may emit only `contact_angle_deg` and omit left/right angles; both schemas are supported. The Results Panel must handle either gracefully.
- Angle definitions are relative to the substrate; ensure coordinate tilt correction precedes measurement.

## Units and Formatting

- `diameter_mm`, `height_mm`: display with 2–3 decimals; export full precision.
- `theta_left_deg`, `theta_right_deg`, `contact_angle_deg`: display with 1 decimal (or per user settings).
- `volume_uL`: 2–3 decimals for typical sizes; adaptive allowed.
- `baseline_tilt_deg`: 2 decimals.

## Example (tangent method)

```json
{
  "schema_version": "1.0",
  "pipeline": "sessile",
  "diameter_mm": 2.941,
  "height_mm": 0.842,
  "theta_left_deg": 96.3,
  "theta_right_deg": 94.8,
  "method": "tangent",
  "uncertainty_deg": {"left": 0.7, "right": 0.6},
  "baseline_tilt_deg": 0.82,
  "volume_uL": 3.54,
  "drop_surface_mm2": 17.9,
  "contact_line": [[112, 356], [286, 347]],
  "diameter_line": [[118, 351], [280, 348]],
  "timings_ms": {"edge_detection": 11.2, "geometry": 7.1}
}
```

## Backward Compatibility

- GUI mapping must tolerate presence of only `contact_angle_deg` and absence of left/right angles.
- If the angle method is unknown/missing, display values without method tag and no uncertainty.
- New keys must not break existing consumers; increment `schema_version` when changing required fields.

## Provenance and Exports

- CSV/JSON exports SHOULD include `schema_version`, `pipeline`, calibration (`px_per_mm`), densities/temperature (if available), baseline endpoints, and `image_path`.

---

Authoritative consumers
- Results Panel mapping
- Batch export writers
- Tests validating sessile results shape

