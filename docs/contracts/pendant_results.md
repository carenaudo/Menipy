# Pendant Pipeline Results Schema

This document defines the pendant pipeline results contract used by the GUI Results Panel and export routines. It normalizes keys, units, and formatting so results are consistent across runs and versions.

## Schema Overview

- schema_version: `1.0`
- pipeline: `pendant`
- required keys:
  - `diameter_mm` (float) — Maximum equatorial diameter De in millimetres.
  - `height_mm` (float) — Total droplet height in millimetres (ROI frame).
  - `r0_mm` (float) — Apex radius of curvature R0 in millimetres.
  - `beta` (float) — Dimensionless form factor from Jennings–Pallas correlation.
  - `surface_tension_mN_m` (float) — Surface tension in mN/m (millinewtons per metre).
  - `volume_uL` (float) — Volume in microlitres.
  - `drop_surface_mm2` (float) — Drop surface area in mm².
- optional keys:
  - `needle_surface_mm2` (float) — Needle cross‑sectional area in mm² (if needle_diam provided).
  - `s1` (float) — Shape factor `De / (2 r0)`.
  - `Bo` (float) — Bond number `Δρ g r0² / γ`.
  - `residuals` (object) — Fit residuals/diagnostics; implementation‑defined structure.
  - `timings_ms` (object) — Per‑stage timings populated by the pipeline runner.
  - `image_path` (string) — Source image path (for provenance in exports).

Notes
- Current implementation emits `diameter_mm`, `height_mm`, `r0_mm`, `beta`, `surface_tension_mN_m`, `volume_uL`, `drop_surface_mm2`, and possibly `needle_surface_mm2`, `s1`.
- The plan previously referenced `De_mm`/`H_mm`; these are aliases of `diameter_mm`/`height_mm` and SHOULD NOT be added as separate keys. Use the canonical names above.

## Units and Formatting

- `diameter_mm`, `height_mm`, `r0_mm`: display with 2–3 decimals; export full precision.
- `surface_tension_mN_m`: display with 1–2 decimals.
- `volume_uL`: display with 2–3 decimals for small drops; adaptive formatting allowed.
- Angles are not part of pendant; omit unless explicitly defined in future extensions.

## Example

```json
{
  "schema_version": "1.0",
  "pipeline": "pendant",
  "diameter_mm": 3.482,
  "height_mm": 4.915,
  "r0_mm": 1.274,
  "s1": 1.366,
  "beta": 0.742,
  "surface_tension_mN_m": 71.7,
  "volume_uL": 12.84,
  "drop_surface_mm2": 45.1,
  "needle_surface_mm2": 0.50,
  "residuals": {"rms": 0.23, "n": 512},
  "timings_ms": {"edge_detection": 12.3, "geometry": 4.9, "solver": 38.6}
}
```

## Backward Compatibility

- GUI mapping must tolerate missing optional fields and unknown extras.
- If future versions introduce new keys, increment `schema_version` and keep original keys stable.

## Provenance and Exports

- CSV/JSON exports SHOULD include `schema_version`, `pipeline`, calibration (`px_per_mm`), densities, temperature (if available), and `image_path`.

---

Authoritative consumers
- Results Panel mapping
- Batch export writers
- Tests validating pendant results shape

