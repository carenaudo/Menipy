# Oscillating Pipeline Results Schema

This document defines the oscillating pipeline results contract used by the GUI Results Panel and export routines. It normalizes keys, units, and formatting so results are consistent across runs and versions.

## Schema Overview

- schema_version: `1.0`
- pipeline: `oscillating`
- required keys:
  - `f0_Hz` (float) — Dominant oscillation frequency in hertz, estimated from the time series.
- recommended keys:
  - `r0_eq_mm` (float) — Frame‑0 area‑equivalent radius in millimetres (or average equilibrium radius if available).
- optional keys:
  - `gamma_mN_m` (float) — Surface tension in mN/m (if physics mapping is enabled and valid).
  - `r0_eq_px` (float) — Frame‑0 area‑equivalent radius in pixels (diagnostic; prefer `r0_eq_mm`).
  - `fps` (float) — Frames per second used to estimate `f0_Hz`.
  - `snr` (float) — Signal‑to‑noise ratio around the dominant peak (unitless).
  - `peak_width_Hz` (float) — Peak half‑power width in hertz (confidence indicator).
  - `n_frames` (int) — Number of frames used in the analysis window.
  - `window` (string) — Window function applied (e.g., `hann`).
  - `estimator` (string) — Estimation method (e.g., `fft`, `prony`, `ar`).
  - `residuals` (object) — Implementation‑defined diagnostics.
  - `timings_ms` (object) — Per‑stage timings.
  - `image_path` (string) — Source (first frame) or clip identifier for provenance.

Notes
- The current implementation estimates `f0_Hz` via FFT of `r_eq_series_px` and may expose `r0_eq_px`. The canonical schema prefers `r0_eq_mm` when calibration is present.
- Physics mapping (`gamma_mN_m` from Rayleigh–Lamb) is optional and must be guarded by validity checks (small amplitude/low viscosity).

## Units and Formatting

- `f0_Hz`: display with 2 decimals; export full precision.
- `r0_eq_mm`: display with 2–3 decimals; export full precision.
- `gamma_mN_m`: display with 1–2 decimals when present.
- `peak_width_Hz`: 2 decimals.

## Example

```json
{
  "schema_version": "1.0",
  "pipeline": "oscillating",
  "f0_Hz": 5.37,
  "r0_eq_mm": 1.26,
  "fps": 120.0,
  "snr": 18.4,
  "peak_width_Hz": 0.22,
  "n_frames": 480,
  "window": "hann",
  "estimator": "fft",
  "timings_ms": {"edge_detection": 92.1, "geometry": 14.7, "optimization": 3.8}
}
```

## Backward Compatibility

- GUI mapping must tolerate presence of `r0_eq_px` instead of `r0_eq_mm` and absence of optional diagnostics.
- If `fps` is unknown, still display `f0_Hz` if available but flag potential inaccuracy in diagnostics.
- Increment `schema_version` when changing required fields; preserve existing keys where possible.

## Provenance and Exports

- CSV/JSON exports SHOULD include `schema_version`, `pipeline`, `fps`, calibration (`px_per_mm`), estimator/windowing, and `image_path` or clip ID.

---

Authoritative consumers
- Results Panel mapping
- Batch export writers
- Tests validating oscillating results shape

