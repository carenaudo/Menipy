# Pendant Pipeline Results Schema

This document defines the pendant pipeline results contract used by the GUI Results Panel and export routines. It normalizes keys, units, and formatting so results are consistent across runs and versions.

## Schema Overview

- schema_version: `1.0`
- pipeline: `pendant`
- required keys:
  - `diameter_mm` (float) — Maximum equatorial diameter De in millimetres.
  - `height_mm` (float) — Total droplet height in millimetres (ROI frame).
  - `r0_mm` (float) — Apex radius of curvature R0 in millimetres.
  - `beta` (float) — Public dimensionless Young-Laplace/Bond parameter when strict fit passes, otherwise the Jennings-Pallas fallback form factor.
  - `surface_tension_mN_m` (float) — Surface tension in mN/m (millinewtons per metre).
  - `volume_uL` (float) — Volume in microlitres.
  - `drop_surface_mm2` (float) — Drop surface area in mm².
- optional keys:
  - `needle_surface_mm2` (float) — Needle cross‑sectional area in mm² (if needle_diam provided).
  - `s1` (float) — Shape factor `De / (2 r0)`.
  - `bond_number` (float) — Bond number `Δρ g r0² / γ`.
  - `worthington_number` (float) — `volume_uL / vmax_uL`.
  - `vmax_uL` (float) — Maximum detachment volume used for Worthington number.
  - `surface_tension_method` (string) — One of `young_laplace_strict`, `multi_selected_plane`, `selected_plane`, `volume_apex_lookup`, or `jennings_pallas_geometric`.
  - `strict_r0_mm`, `strict_beta`, `strict_surface_tension_mN_m`, `strict_rmse_mm`, `strict_fit_stop_reason` — Strict Young-Laplace diagnostics in calibrated units.
  - `geometric_r0_mm`, `geometric_beta`, `geometric_surface_tension_mN_m` — Jennings-Pallas comparison values.
  - `approx_selected_plane_surface_tension_mN_m`, `approx_selected_plane_beta`, `approx_selected_plane_status` — Single selected-plane approximation result.
  - `approx_multi_selected_plane_surface_tension_mN_m`, `approx_multi_selected_plane_std_mN_m`, `approx_multi_selected_plane_planes` — Multi-plane approximation result and per-plane diagnostics.
  - `approx_volume_apex_surface_tension_mN_m`, `approx_volume_apex_beta`, `approx_volume_apex_status` — Volume plus apex-curvature lookup approximation.
  - `residuals` (object) — Fit residuals/diagnostics; implementation‑defined structure.
  - `timings_ms` (object) — Per‑stage timings populated by the pipeline runner.
  - `image_path` (string) — Source image path (for provenance in exports).

### Common diagnostics envelope

New runs may include the additive `diagnostics` object with `solver`,
`residuals`, `confidence`, `validity`, `calibration`, `side_discrepancy`, and
`detectors`. The established `residuals`, `strict_rmse_mm`, and
`strict_fit_stop_reason` fields remain available during migration.

When explicitly enabled, `diagnostics.experimental_geometry` may contain a
robust pendant axis initializer with `axis_origin_px`, `axis_direction_xy`,
`r0_seed_mm`, `beta_seed`, coverage, asymmetry, and rejection reasons. The
legacy vertical-axis fields remain valid when the initializer is disabled.

Phase-C shadow runs may add `diagnostics.onnx_proposals`. Proposal masks and
contours are non-authoritative and do not alter calibration, strict fitting,
surface tension, acceptance, or rejection reasons.

Persisted measurements include `accepted`, `rejection_reasons`, and
diagnostics. A rejected solve remains auditable but its physical candidate
values are not exported inside `results` as valid measurements. These fields
are optional, so schema version `1.0` is unchanged.

Notes
- Current implementation emits `diameter_mm`, `height_mm`, `r0_mm`, `beta`, `surface_tension_mN_m`, `volume_uL`, and possibly `needle_surface_mm2`, `s1`, strict fit diagnostics, and geometric comparison values.
- Public `r0_mm`, `beta`, and `surface_tension_mN_m` come from strict Young-Laplace only when the fit succeeds and passes residual gates. Otherwise public surface tension falls back to the first enabled approximation with `ok` status in this order: multi-selected-plane, selected-plane, volume-apex lookup, then legacy Jennings-Pallas.
- Accepted strict pendant profiles are truncated at the observed contact/needle height before loop branches are generated for display or reporting.
- Approximation plugins are comparison methods for scientific review. They should be exported when present but treated as optional diagnostics by consumers.
- Method references: Berry et al. 2015 for pendant selected-plane and ADSA workflow (https://doi.org/10.1016/j.jcis.2015.05.012), Jůza et al. 2026 for selected/multiple selected-plane notation (https://link.springer.com/article/10.1007/s00396-025-05513-5), and Yeow et al. 2008 for volume plus apex-curvature estimation (https://doi.org/10.1016/j.colsurfa.2007.07.025).
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
  "surface_tension_method": "young_laplace_strict",
  "volume_uL": 12.84,
  "drop_surface_mm2": 45.1,
  "bond_number": 0.31,
  "worthington_number": 0.42,
  "approx_volume_apex_surface_tension_mN_m": 70.9,
  "approx_volume_apex_status": "ok",
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
