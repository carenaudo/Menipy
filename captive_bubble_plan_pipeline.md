# Captive Bubble Pipeline — Current State and Enhancement Plan

This document captures the state of the “captive bubble” pipeline and proposes improvements across accuracy, performance, UX, and integration. Structure mirrors the pendant/sessile plans for cross-pipeline consistency.

## Scope

- Pipeline: `captive_bubble` (processing + GUI integration)
- Baseline: repo at time of analysis

---

## 1) Current Implementation Summary

### 1.1 Key Components

- Core base/discovery
  - `src/menipy/pipelines/base.py` — stage skeleton, orchestration, timings, errors.
  - `src/menipy/pipelines/discover.py` — dynamic discovery used by GUI/CLI.

- Captive bubble pipeline
  - `src/menipy/pipelines/captive_bubble/stages.py` — `CaptiveBubblePipeline(PipelineBase)` simplified implementation only (no separate `geometry.py`/`metrics.py`).
    - Ensures contour via common edge detection (`canny` default) when needed.
    - Geometry: ceiling `y` as min(y), axis `x` as median(x), visual apex as max(y); computes cap depth in pixels.
    - Scaling: default `px_per_mm=1.0`.
    - Physics: defaults for densities and `g`.
    - Solver: toy Young–Laplace sphere fit returning `R0_mm` (placeholder for wiring).
    - Outputs: returns `R0_mm`, `cap_depth_px`, and residuals.
    - Overlay: draws contour, ceiling line, symmetry axis, apex cross, and text with `R0` and cap depth.
    - Validation: solver success check.

- GUI integration
  - Follows the same base services/controllers as other pipelines; no dedicated functional analyzer path is present.

### 1.2 Data Flow

- Full pipeline path only (no separate functional analyzer): acquisition/preprocess/edge_detection → geometry → scaling → physics → solver → outputs → overlay → validation.
- Results consumed by GUI via `ctx.results`; overlay via `ovl.run` commands.

### 1.3 Algorithms (current)

- Ceiling location via image min(y) of contour; axis via median(x); apex via max(y). Depth = max(y) − min(y).
- No physical conversion of cap depth to mm nor inverse surface tension inference beyond the toy solver.

### 1.4 Gaps / Inconsistencies

- Single-file stage implementation; lacks dedicated `geometry`, `metrics`, and `overlay` modules for testability and reuse.
- Geometry heuristics are fragile (min/max/median); no robustness to tilt or partial contours.
- Missing mm conversions for depth and bubble dimensions; calibration not enforced.
- No results schema contract; Results Panel not tailored for captive-bubble metrics.
- Toy solver does not reflect bubble physics (bubble under a ceiling entails sign/density considerations and different boundary conditions).

---

## 2) Improvement Opportunities

### 2.1 Performance

- Unify contour extraction with preprocessing/edge settings (already using common path); keep it consistent with other pipelines.
- Vectorize dimension extraction and add optional smoothing near the apex/neck to stabilize measures.
- Add ROI-driven downscaling for preview; compute at full resolution for final results.

### 2.2 Accuracy and Robustness

- Ceiling detection: robust baseline detection at the ceiling (RANSAC line fit near min(y) band), with tilt correction and confidence.
- Axis estimation: robust symmetry axis using minimal radius-variance criterion; fallback to median x.
- Cap geometry: compute bubble height, equatorial diameter, local curvature radii; convert to mm.
- Surface tension inference: integrate a proper Young–Laplace solver adapted to a gas bubble attached to a solid ceiling (sign of Δρ reversed vs. pendant; boundary conditions at ceiling contact). Keep a correlation/estimate mode if literature supports it for captive bubbles.
- Units/scale: require nonzero `px_per_mm` and proper densities (`ρ_liquid`, `ρ_gas`).

### 2.3 Scalability and Architecture

- Split logic into `geometry.py`, `metrics.py`, and `overlay.py`; keep `stages.py` as thin orchestration.
- Define a captive-bubble results schema and document it under `docs/contracts`.
- Align with the Results Panel pipeline-awareness work for consistent display and exports.

### 2.4 UX and Workflow

- Ceiling tools: auto-detect ceiling line, manual adjust, and lock; show confidence.
- Overlay: annotate ceiling, axis, apex, depth (mm), and optional fitted profile.
- Consistent Analyze/Run behavior (staged path), with progress/diagnostics.

### 2.5 Error Handling and Observability

- Input validation (ROI, scale, densities) at stage boundaries; clear GUI messages.
- Diagnostics: residuals, fit status, timings; display in a diagnostics tab.

---

## 3) Proposed Implementation Plan

### 3.1 Goals

- Robust geometry and surface-tension estimation for captive bubbles with tilt-aware ceiling detection and consistent units.
- Results Panel and export parity with other pipelines.

### 3.2 Architectural Changes

- Add `src/menipy/pipelines/captive_bubble/{geometry.py,metrics.py,overlay.py,solver.py}`; migrate logic out of `stages.py`.
- Introduce a results schema doc: `docs/contracts/captive_bubble_results.md`; update Results Panel mapping.

### 3.3 Feature Roadmap (Phased)

Phase 0 — Contracts and wiring (1–2 days)

- Draft results schema and labels (depth_mm, height_mm, diameter_mm, R0_mm, gamma_mN_m?, volume_uL?).
- Ensure discovery exposes the pipeline; add missing imports/guards analogous to other pipelines.

Phase 1 — Geometry and ceiling detection (2–4 days)

- Implement ceiling line detection (RANSAC) and tilt correction; compute depth_mm and equatorial diameter_mm.
- Robust apex/axis estimation; smoothing near apex as needed.

Phase 2 — Metrics and units (2–3 days)

- Convert all geometric measures to mm; compute curvature at apex and related descriptors.
- Validate measures on sample images; add unit tests.

Phase 3 — Physical solver integration (5–8 days)

- Adapt Young–Laplace solver for captive-bubble boundary conditions and sign conventions; select residual metric and initialization strategy.
- Expose diagnostics (iterations, RMS residual, success) and keep correlation/estimate as a fast alternative if supported.

Phase 4 — UX, diagnostics, and exports (2–3 days)

- Overlay enhancements (mm annotations; optional fitted profile overlay); diagnostics tab integration.
- Results Panel mapping and CSV/JSON exports with provenance.

### 3.4 API Contracts and Data Model

- Inputs: `image_path|frames`, `roi`, `px_per_mm`, densities (`ρ_liquid`, `ρ_gas`), `g`.
- Intermediates: `contour.xy`, `geometry.ceiling_y`, `geometry.axis_x`, `geometry.apex_xy`.
- Outputs: `{depth_mm, diameter_mm, height_mm?, R0_mm, gamma_mN_m?}` plus diagnostics (`residuals`, `timings_ms`).

### 3.5 Dependencies

- Imaging: OpenCV/skimage for edges and RANSAC; numerical: numpy/scipy for fitting.

### 3.6 Estimates (rough, engineer-days)

- P0: 1–2; P1: 2–4; P2: 2–3; P3: 5–8; P4: 2–3 → Total: 12–20 days.

---

## 4) Risks, Dependencies, Prerequisites

- Ceiling reflections and texture can mislead line detection; tune RANSAC and provide user override.
- Solver sensitivity to initialization; careful parameterization needed (Bond number range, apex radius bounds).
- Calibration/density accuracy is critical for physical outputs; require and validate in UI.

---

## 5) Physics Background and Research Notes (Captive Bubble)

- Governing equations
  - Young–Laplace still applies; sign/direction for hydrostatic pressure difference is inverted relative to pendant (gas bubble under ceiling). Boundary conditions at the ceiling contact influence the profile.
  - Axisymmetric profile parameterization via arc length with apex at the lowest point; integrate upward to the ceiling with appropriate conditions.

- Dimensionless numbers
  - Bond number `Bo = Δρ g R0^2 / γ` with `Δρ = ρ_liquid − ρ_gas`; capillary length `ℓ_c = sqrt(γ/(Δρ g))` still sets the gravitational scale.

- Measurement strategies
  - Geometric: depth (ceiling to apex), equatorial diameter, apex curvature, area/volume by revolution; convert to mm using calibration.
  - Physical: fit a Y–L profile to the observed contour to estimate `γ` or `R0` given `γ`.

- Assumptions and imaging
  - Static equilibrium; clean interface; flat ceiling (locally), orthogonal view, adequate resolution near the apex and ceiling region; reliable calibration and densities.

- Validation
  - Synthetic bubbles with known `γ`, `Δρ`, and added noise/tilt; reference liquids and known bubble sizes if available.

---

## 6) Results Panel Integration (Captive Bubble)

- Goals
  - ResultsPanel displays captive-bubble metrics/units when this pipeline is active; provide diagnostics (residuals, depth).

- Behavior
  - Core metrics: `depth_mm`, `diameter_mm`, `R0_mm`, optional `gamma_mN_m`, `volume_uL`, `drop_surface_mm2`.
  - Diagnostics: `residuals`, `timings_ms`, image/calibration provenance.

- Tasks
  - Define results schema in `docs/contracts/captive_bubble_results.md` and update Results Panel mapping.
  - Add formatting helpers for mm/mN/m fields; handle absence of solver outputs gracefully.
  - Cross-link: see `docs/contracts/captive_bubble_results.md` for canonical keys/units and diagnostics fields.
  - Implement pipeline-aware switching: detect active pipeline from context, pick corresponding schema map, render values with unit-aware formatting; ensure exports include `schema_version` and `pipeline`.

- Diagnostics tab outline
  - Summary: solver status, RMS residual, iterations; ceiling tilt/confidence.
  - Plots: residuals along arc length; optional fitted vs. measured overlay.
  - Timings and provenance: same pattern as other pipelines.

---

Questions and prioritization feedback welcome.
