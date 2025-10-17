# Capillary Rise Pipeline — Current State and Enhancement Plan

This plan documents the current implementation of the “capillary rise” pipeline, identifies improvement opportunities, and proposes a phased, actionable roadmap. It mirrors the structure used for pendant, sessile, captive-bubble, and oscillating pipelines to stay consistent across the suite.

## Scope

- Pipeline: `capillary_rise` (processing + GUI integration)
- Baseline: repo at time of analysis

---

## 1) Current Implementation Summary

### 1.1 Key Components

- Base/discovery
  - `src/menipy/pipelines/base.py` — stage skeleton, orchestration, timings, errors.
  - `src/menipy/pipelines/discover.py` — dynamic discovery used by GUI/CLI.

- Capillary rise pipeline
  - `src/menipy/pipelines/capillary_rise/stages.py` — `CapillaryRisePipeline(PipelineBase)` simplified implementation:
    - Ensures contour via edge detection (Canny default) if absent.
    - Geometry: `baseline_y = max(y)` (tube bottom), meniscus apex as `argmin(y)`, axis as median x, rise height `h_px = baseline_y − apex_y`.
    - Scaling: default `px_per_mm=1.0`.
    - Physics: default densities and `g`.
    - Solver: toy Young–Laplace “sphere” to return `R0_mm` (placeholder).
    - Outputs: merges fit params with `h_px`; overlay draws contour, baseline, axis gauge to apex, and text with `R0` and `h`.
    - Validation: solver success check.
  - Other capillary modules exist as stubs (`geometry.py`, `metrics.py`, etc.) but are not implemented; logic is concentrated in `stages.py`.

- GUI integration
  - Shares the same staged runner and controllers as other pipelines; ResultsPanel currently generic.

### 1.2 Data Flow

- Full staged path only: acquisition/preprocessing/edge_detection → geometry (baseline/apex/axis; height) → scaling → physics → solver (toy) → outputs → overlay → validation.
- Inputs: single image or frames, ROI; calibration (`px_per_mm`), densities (`ρ_liquid`, `ρ_air`), gravity.
- Outputs: `h_px` (currently), `R0_mm` (toy).

### 1.3 Algorithms (current)

- Baseline: take max y of the contour within the ROI (assumes horizontal bottom). Apex: min y point on the contour. Axis: median x.
- Height gauge: vertical line from baseline to apex at the median axis; unit is pixels.

### 1.4 Gaps / Inconsistencies

- Unit handling: height reported in pixels; no conversion to mm or link to fluid properties.
- Baseline detection: naive use of `max(y)`; no robust line detection or tilt correction for the tube.
- Meniscus shape: no analysis of curvature, contact angle, or tube radius; `R0` from toy solver is not physically meaningful here.
- Architecture: logic in `stages.py`; `geometry.py/metrics.py/overlay.py/solver.py` not implemented, reducing testability and reuse.
- Results schema: none defined for capillary rise; ResultsPanel not pipeline-aware.

---

## 2) Improvement Opportunities

### 2.1 Performance

- Keep using common edge detection; allow optional downscale for preview with refined measurements at full resolution.
- Vectorize baseline and apex search with robust line-fitting preselection to reduce sensitivity to noise.

### 2.2 Accuracy and Robustness

- Baseline/tube detection: detect the vertical tube walls (Hough lines or RANSAC) to estimate tube center and inner radius; detect horizontal baseline/meniscus reference with tilt correction.
- Meniscus apex: use curvature maxima and subpixel refinement near the apex; consider fitting a circle/Young–Laplace segment locally.
- Height conversion: report `h_mm = h_px / px_per_mm` with uncertainty; validate calibration.
- Physics linkage: use Jurin’s law `h = 2γ cosθ / (ρ g r)` to infer `γ` or `cosθ` when tube radius r and one of (γ, θ) are known; otherwise report `h_mm` only.
- Tube radius: estimate from detected inner walls in pixels and convert to mm.

### 2.3 Scalability and Architecture

- Split logic into `geometry.py` (tube/meniscus detection), `metrics.py` (height, radius, optional θ/γ inference), `overlay.py` (gauges, walls, apex), and `solver.py` (if a Y–L or circle-fit segment is used), keeping `stages.py` as orchestration.
- Define a results schema for capillary rise under `docs/contracts` and update ResultsPanel mapping.

### 2.4 UX and Workflow

- Provide tools to draw/adjust tube walls and baseline; auto-detect with confidence, allow manual corrections.
- Overlay: show tube walls, axis, apex, and height gauge with units; optionally display inferred `γ` or `θ` and tube radius.
- Batch: progress and standardized CSV/JSON export including provenance and method.

### 2.5 Error Handling and Observability

- Validate required inputs: ROI, calibration, tube radius (if not auto-estimated), densities; surface clear messages in GUI.
- Diagnostics: stage timings and detection confidences; plots of vertical profile and curvature near the apex.

---

## 3) Proposed Implementation Plan

### 3.1 Goals

- Robust detection of tube geometry and meniscus apex; accurate height `h_mm` with optional inference of `γ` or `θ` via Jurin’s law; clean overlays and pipeline-aware results display.

### 3.2 Architectural Changes

- Implement `capillary_rise/geometry.py`, `metrics.py`, `overlay.py` and migrate logic out of `stages.py`.
- Add `docs/contracts/capillary_rise_results.md` and update ResultsPanel mapping.

### 3.3 Feature Roadmap (Phased)

Phase 0 — Contracts and wiring (1–2 days)

- Draft results schema: keys (e.g., `h_mm`, `r_tube_mm`, `gamma_mN_m?`, `theta_deg?`) and units; update ResultsPanel plan.
- Ensure discovery and pipeline runner paths are aligned.

Phase 1 — Geometry detection (2–4 days)

- Detect tube walls and estimate inner radius; detect baseline with tilt correction; apex refinement via curvature max.
- Unit tests on sample images; fallback to manual annotations when auto-detect fails.

Phase 2 — Metrics and physics linkage (2–4 days)

- Compute `h_mm`; if `r_tube_mm` and either `γ` or `θ` are provided, infer the other via Jurin’s law. Provide uncertainty estimates.
- Optional: local circle/segment fit on the meniscus for `θ` estimation near wall contact.

Phase 3 — UX, overlays, and exports (2–3 days)

- Overlay enhancements (tube walls, apex, height gauge, annotations with units); diagnostics panel entries (timings, detection confidences).
- ResultsPanel mapping and CSV/JSON exports with provenance.

### 3.4 API Contracts and Data Model

- Inputs: `image_path|frames`, `roi`, `px_per_mm`, densities, `g`, tube inner radius (if not auto-estimated), or wall lines for estimation.
- Intermediates: `contour.xy`, `geometry.baseline_y`, `geometry.axis_x`, `geometry.apex_xy`, `geometry.r_tube_px`.
- Outputs: `{h_mm, r_tube_mm, gamma_mN_m?, theta_deg?}` plus diagnostics (`confidences`, `timings_ms`).

### 3.5 Dependencies

- Imaging: OpenCV/skimage for edges and Hough/RANSAC line detection; numpy/scipy for local fits.

### 3.6 Estimates (rough, engineer-days)

- P0: 1–2; P1: 2–4; P2: 2–4; P3: 2–3 → Total: 7–13 days.

---

## 4) Risks, Dependencies, Prerequisites

- Tube reflections and non-uniform lighting may confuse wall/baseline detection; require manual override tools.
- Calibration and tube radius accuracy dominate physics outputs; enforce validation and clearly communicate assumptions.
- If meniscus deviates from ideal due to contamination or dynamic effects, Jurin-based inference may be biased; report as such.

---

## 5) Physics Background and Research Notes (Capillary Rise)

- Jurin’s law (static capillary rise in a tube)
  - At equilibrium, `h = 2 γ cosθ / (ρ g r)`, with `h` the rise height, `γ` surface tension, `θ` contact angle (against the tube wall), `ρ` the liquid density minus gas density, `g` gravity, and `r` the tube inner radius.
  - Assumptions: circular tube, wetting equilibrium (θ constant), negligible evaporation and flow, isothermal, clean interfaces.
  - Practical: measure `r` in mm from detected tube walls; compute `h_mm` from calibration; infer `γ` or `θ` only if the other is known/reliable.

- Meniscus shape and angle
  - Near the wall, the local contact angle can be estimated by fitting a small arc/curve to the meniscus and evaluating the tangent relative to the wall; this provides an independent `θ` estimate.

- Validation strategy
  - Reference liquids and tubes with known radii; compare `h_mm` against expected values at given `θ`/`γ`.
  - Sensitivity: perturb `px_per_mm`, `r_tube_mm`, and apex localization to estimate uncertainty.

---

## 6) Results Panel Integration (Capillary Rise)

- Goals
  - Display `h_mm` prominently with units; show `r_tube_mm` and either inferred `γ` or `θ` (when available); include diagnostics (confidences, timings, provenance).

- Behavior
  - Core metrics: `h_mm`, `r_tube_mm`; optional: `gamma_mN_m`, `theta_deg`.
  - Diagnostics: `timings_ms`, detection confidences, image/calibration provenance.

- Tasks
  - Add `docs/contracts/capillary_rise_results.md` and update ResultsPanel mapping.
  - Formatting helpers for mm/mN/m and degree fields; handle absent physics outputs gracefully.
  - Cross-link: see `docs/contracts/capillary_rise_results.md` for canonical keys/units and diagnostics fields.
  - Implement pipeline-aware switching: detect active pipeline from context, pick corresponding schema map, and render `h_mm`, `r_tube_mm`, optional `gamma_mN_m`/`theta_deg` with unit-aware formatting; include `schema_version` and `pipeline` in exports.

- Diagnostics tab outline
  - Summary: `h_mm`, `r_tube_mm`, optional `γ`/`θ`; detection confidence and tilt correction.
  - Plots: vertical intensity profile or curvature near apex; overlay thumbnails.
  - Timings and provenance.

---

Comments and prioritization feedback welcome.
