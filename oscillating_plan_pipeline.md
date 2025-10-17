# Oscillating Drop Pipeline — Current State and Enhancement Plan

This plan outlines the “oscillating” pipeline (processing + GUI), identifies gaps, and proposes an actionable roadmap. It mirrors the pendant/sessile/captive-bubble plans for consistency.

## Scope

- Pipeline: `oscillating` (time-series analysis + geometry + physics)
- Baseline: repo at time of analysis

---

## 1) Current Implementation Summary

### 1.1 Key Components

- `src/menipy/pipelines/oscillating/stages.py` — `OscillatingPipeline(PipelineBase)` simplified implementation:
  - Edge detection per-frame: builds `contours_by_frame` by running common edge detection frame-by-frame; sets `ctx.contour` to frame 0 for downstream.
  - Geometry: computes area-equivalent radius per frame `r_eq_series_px` and frame-0 center; stores axis/apex for overlays.
  - Physics: defaults (`fps`, `rho1`, `rho2`, `g`).
  - Solver: toy Young–Laplace fit on frame 0 for `R0_mm` (placeholder; not central to oscillation analysis).
  - Optimization: estimates dominant frequency `f0_Hz` via FFT of detrended `r_eq_series_px` using provided/assumed `fps`.
  - Outputs: includes `R0_mm`, optional `f0_Hz`, residuals; exports `r0_eq_px`.
  - Overlay: draws contour, equivalent circle at frame 0, symmetry axis, and text with `R0` and `f0`.

- Other modules under `oscillating/` exist but are empty placeholders (`geometry.py`, `solver.py`, `overlay.py`, etc.); logic is concentrated in `stages.py`.

- GUI integration: Uses the shared pipeline infrastructure; no dedicated functional analyzer path is defined for oscillating (staged path only).

### 1.2 Data Flow

- Acquisition → preprocessing → per-frame edge detection → geometry (per-frame radii and refs) → physics (params, fps) → solver (toy) → optimization (FFT-based `f0`) → outputs → overlay → validation.
- Inputs: video frames or list of images, `fps` if not inferable, calibration (`px_per_mm`), densities.
- Outputs: `f0_Hz` (dominant frequency), `R0_mm` (toy), optional series stored in geometry.

### 1.3 Algorithms (current)

- `r_eq_series_px`: polygon area-based equivalent radius per frame; FFT to estimate dominant frequency; store frame-0 circle and axis for overlays.
- No de-noising/detrending beyond mean removal; no windowing or bandpass; no damping model fit; no radius-to-mm conversion for the time series.

### 1.4 Gaps / Inconsistencies

- Concentrated logic in `stages.py`; missing modularization into `geometry.py` (time-series extraction), `optimization.py` (spectral analysis), and `physics.py` (model-based inference).
- Sensitivity to edge noise and frame-to-frame contour jitter; no smoothing/windowing; no handling of missing/failed frames.
- Frequency estimate lacks confidence intervals and peak validation; no alternative estimators (e.g., Prony/ESPRIT/CLEAN) for short/noisy series.
- Physics: no Rayleigh–Lamb linkage between `f0` and `γ`, radius, and density; no damping/viscosity modeling.
- Units: `r_eq_series_px` not converted to mm; calibration not enforced for physical inference.
- GUI: no spectral plot or time-series preview panel; ResultsPanel not pipeline-aware for oscillating metrics.

---

## 2) Improvement Opportunities

### 2.1 Performance

- Reuse contours/edges; optionally downscale for edge detection and refine around the interface; cache intermediate results.
- Vectorize per-frame features; avoid repeated Python object churn; support chunked processing for long videos.

### 2.2 Accuracy and Robustness

- Preprocess `r_eq(t)`: de-noise (Savitzky–Golay or low-pass), detrend (remove drift), and normalize.
- Spectral estimation: apply windowing (Hann) prior to FFT; add peak picking with neighborhood validation; estimate uncertainty via peak width or bootstrapping.
- Alternative estimators for short/noisy series: Prony/AR, ESPRIT, or zero-crossing with correction; pick method adaptively based on SNR/length.
- Convert radii to mm and propagate calibration uncertainty; validate `fps` accuracy.
- Handle missing/failed frames gracefully (interpolate or drop with masking).

### 2.3 Scalability and Architecture

- Move time-series extraction to `geometry.py`; spectral estimation to `optimization.py`; physics mapping to `physics.py`; overlays to `overlay.py`.
- Define a results schema for oscillating pipeline; add to `docs/contracts` and ResultsPanel mapping.
- Provide a small set of configuration knobs (window length, filter params, estimator choice).

### 2.4 UX and Workflow

- Add a time-series/spectrum preview (plot `r_eq(t)` and |FFT|); allow selecting analysis window; display `f0` with confidence.
- Overlay: optionally show evolution (ghost contours) or selected frames; annotate center and equivalent circle.
- Batch: per-clip progress, summaries, and export of `f0`, confidence, and settings.

### 2.5 Error Handling and Observability

- Validate presence of `fps` and `px_per_mm` when physical inference is requested; warn otherwise.
- Expose diagnostics: SNR around `f0`, peak width, series length; stage timings.

---

## 3) Proposed Implementation Plan

### 3.1 Goals

- Robust `f0` estimation with uncertainty; optional physics-based inference of `γ` (if appropriate) from `f0` and radius per Rayleigh–Lamb small-amplitude theory.
- Clean separation of concerns; GUI previews (time series + spectrum) and ResultsPanel integration.

### 3.2 Architectural Changes

- Create/implement: `oscillating/geometry.py` (extract `r_eq_series_mm`, centers), `oscillating/optimization.py` (FFT/windowing + alternative estimators), `oscillating/physics.py` (Rayleigh–Lamb link), `oscillating/overlay.py` (frame selections).
- Keep `stages.py` as orchestration; call module functions; maintain compatibility with `PipelineBase`.

### 3.3 Feature Roadmap (Phased)

Phase 0 — Contracts and wiring (1–2 days)

- Draft results schema (`docs/contracts/oscillating_results.md`): keys, units, formatting; map in ResultsPanel.
- Ensure discovery and staged runner paths are aligned; add minimal settings (windowing, smoothing) with defaults.

Phase 1 — Time-series extraction and preprocessing (2–4 days)

- Compute `r_eq_series_mm` and centers; add smoothing/detrending; handle missing frames.
- Validate on sample clips; add unit tests for polygon area and conversions.

Phase 2 — Spectral estimation and `f0` (3–5 days)

- Implement windowed FFT peak-picking with confidence; add alternative estimator (Prony or AR) for short series; reconcile outputs.
- Provide uncertainty estimates (peak width/bootstrapping) and SNR measures.

Phase 3 — Physics mapping (optional) (3–5 days)

- Rayleigh–Lamb small-amplitude relation (see Physics below) to infer `γ` from `f0` and equilibrium radius `R` for selected mode `n` (typically n=2 for shape oscillations); include liquid density and mm scaling.
- Guard with assumptions (small amplitude, inviscid/low-viscosity, axisymmetry); document limitations; add toggles in settings.

Phase 4 — UX, diagnostics, exports (2–4 days)

- Add time-series/spectrum preview widgets; show `f0` and confidence; allow trimming window.
- ResultsPanel mapping; CSV/JSON export with provenance (fps, px_per_mm, estimator, windowing).
- Diagnostics tab: series length, SNR, peak width, timings.

### 3.4 API Contracts and Data Model

- Inputs: frames (list or video), `fps`, `px_per_mm`, densities (if physics mapping enabled).
- Intermediates: `contours_by_frame`, `r_eq_series_mm`, centers.
- Outputs: `{f0_Hz, r0_eq_mm, gamma_mN_m?}` plus diagnostics `{snr, peak_width_Hz, n_frames}` and `residuals?`.

### 3.5 Dependencies

- Numerical/signal processing: numpy/scipy; optional `spectrum` libs if available; avoid heavy deps if not needed.

### 3.6 Estimates (rough, engineer-days)

- P0: 1–2; P1: 2–4; P2: 3–5; P3: 3–5 (optional); P4: 2–4 → Total: 11–20 days (lower if physics mapping omitted).

---

## 4) Risks, Dependencies, Prerequisites

- `fps` inaccuracies and dropped frames bias `f0`; require verification and allow manual override.
- Edge jitter and illumination changes perturb `r_eq(t)`; smoothing and robust contours selection are needed.
- Physics mapping validity limited to small amplitudes and low viscosity; communicate assumptions clearly.

---

## 5) Physics Background and Research Notes (Oscillations)

- Rayleigh–Lamb small-amplitude oscillations (inviscid approximation)
  - For an isolated spherical drop of radius `R` in a fluid of density `ρ`, the natural frequency of the `n`-th mode is
    `ω_n^2 = n(n-1)(n+2) γ / (ρ R^3)`; for `n=2`, `ω_2 = sqrt(8 γ / (ρ R^3))` and `f_2 = ω_2 / (2π)`.
  - This provides a relation to infer `γ` from measured `f0` and estimated equilibrium radius `R` (from frame-0 or average radius), assuming small oscillations and negligible viscosity.
- Damping and viscosity
  - Viscous damping shifts frequency and introduces decay; for higher viscosity or larger amplitudes, corrections are needed (beyond scope for initial implementation). Consider fitting an exponentially decaying sinusoid to estimate damping ratio and corrected `f0`.
- Practical considerations
  - Use radius in metres for physics; propagate `px_per_mm` uncertainty; choose `ρ` appropriate for the drop vs. surrounding medium.
  - Ensure `f0` corresponds to the shape oscillation mode (n=2) rather than centroid motion or illumination artifacts.

---

## 6) Results Panel Integration (Oscillating)

- Goals
  - Display `f0_Hz` with confidence and (optionally) `gamma_mN_m` if physics mapping is enabled; show `r0_eq_mm`.
  - Provide a diagnostics tab with spectrum, SNR, peak width, and timings.

- Tasks
  - Add `docs/contracts/oscillating_results.md` and update ResultsPanel mapping for oscillating metrics.
  - Add plotting hooks in the GUI for time-series/spectrum previews.
  - Cross-link: see `docs/contracts/oscillating_results.md` for canonical keys/units and diagnostics fields.
  - Implement pipeline-aware switching: detect active pipeline from context, pick corresponding schema map, and render `f0_Hz`, `r0_eq_mm`, and diagnostics with unit-aware formatting; include `schema_version`, `pipeline`, `fps`, and `estimator` in exports.

---

Comments and prioritization feedback welcome.
