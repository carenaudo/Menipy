# Sessile Drop Pipeline — Current State and Enhancement Plan

This document analyzes the current "sessile" pipeline (processing and GUI), highlights improvement opportunities, and proposes a concrete, phased plan to enhance accuracy, performance, UX, and maintainability. It leverages shared base pipeline functionality and common utilities to minimize duplication.

## Scope

- Pipeline: `sessile` (processing + GUI integration)
- Codebase baseline: repo at time of analysis

---

## 1) Current Implementation Summary

### 1.1 Key Components (by module)

- Core pipeline base (shared)
  - `src/menipy/pipelines/base.py` — `PipelineBase` template with standard stages, orchestration (`run`, `run_with_plan`), timing, error handling.
  - `src/menipy/pipelines/discover.py` — Dynamic discovery used by GUI and CLI.

- Sessile pipeline
  - `src/menipy/pipelines/sessile/stages.py` — `SessilePipeline(PipelineBase)` basic implementation:
    - Acquisition: loads image via OpenCV if not present in context.
    - Ensures contour via `edge_detection.run(...)` (Canny default if settings absent).
    - Geometry: crude baseline at max y, symmetry axis as median x, apex as min y; crude contact angles via local slope at estimated left/right base points; stores angles in `ctx.results`.
    - Scaling: default `px_per_mm=1.0`.
    - Physics: default densities and `g`.
    - Solver: toy Young–Laplace "sphere" integrator to estimate `R0_mm` (for display only).
    - Outputs: merges `R0_mm` with `theta_left_deg`, `theta_right_deg`, residuals.
    - Overlay: draws contour, symmetry axis, baseline, apex cross, and text label with `R0` and contact angles.
    - Validation: checks toy solver success flag.

  - `src/menipy/pipelines/sessile/geometry.py` — Functional analyzer path used by the GUI's direct analysis:
    - `analyze(frame, helpers)` → `extract_external_contour` → `find_apex_index` → `compute_sessile_metrics` → returns `SessileMetrics` with contour, apex, diameter/center/contact line, derived metrics.
    - `HelperBundle` inputs: `px_per_mm`, `substrate_line`, optional `contact_points`, density difference/`g`, contact point tolerance.

  - `src/menipy/pipelines/sessile/metrics.py` — `compute_sessile_metrics(...)` basic implementation:
    - If substrate line provided, finds contact points via `find_contact_points_from_contour`, estimates base diameter, height (perpendicular distance apex→substrate), contact area (base disk), volume by revolution around axis perpendicular to substrate through apex, and spherical-cap-based contact angle.
    - Returns `derived` metrics including `diameter_mm`, `height_mm`, `volume_uL`, `contact_angle_deg`, `contact_surface_mm2`, `drop_surface_mm2`, plus line overlays.

- Common maths/geometry (shared, DRY principle)
  - `src/menipy/common/geometry.py` — contact point finder, curvature estimates, circle fit, intersections.
  - `src/menipy/common/metrics.py` — `find_apex_index` for apex detection.
  - `src/menipy/common/edge_detection.py` — `extract_external_contour` for contour extraction.
  - `src/menipy/models/surface_tension.py` — `volume_from_contour` (revolution integral) and physical relations.
  - `src/menipy/models/drop_extras.py` — `surface_area_mm2` and related utilities.

- GUI integration
  - `src/menipy/gui/controllers/pipeline_controller.py` — Supports two flows:
    - Full pipeline run via `PipelineBase` subclass resolved from `PIPELINE_MAP`.
    - Simple analysis via functional `sessile.geometry.analyze` and `sessile.drawing` for quick overlays.
    - Collects ROI, substrate/contact overlays from preview; pulls calibration from GUI tabs; updates preview/results panels.
  - `src/menipy/gui/services/pipeline_runner.py` — Threaded runner for staged pipelines, with acquisition patching, subset runs, and callback signaling.
  - `src/menipy/gui/overlay.py` — Generic overlay drawing compatible with `SessileMetrics`.
  - `src/menipy/gui/panels/results_panel.py` — Displays metric values with labels (includes contact-angle, contact-surface entries).

### 1.2 Data Flow

- Full pipeline path
  - Setup (sources/SOP) → `PipelineController.run_full/run_all` → `PipelineRunner` job → `PipelineBase.run` stages mutate `Context` → GUI consumes `ctx.preview`/`ctx.results`.

- Simple analysis path
  - GUI extracts ROI + calibration + substrate/contact overlays → `sessile.geometry.analyze(...)` → `SessileMetrics` → overlay composed via `gui/overlay.py` → results panel updated with `metrics.derived`.

- Inputs and overlays
  - Substrate line and optional manual contact points come from `PreviewPanel`; calibration (`px_per_mm`, densities) from calibration tab.

### 1.3 Algorithms (current)

- Contour: Edge detection defaulting to Canny (staged path) or `extract_external_contour` (functional path).
- Baseline: max y of contour (staged path); in metrics path, substrate line is expected from GUI and used directly.
- Contact points: proximity to substrate line with curvature-informed scoring (`find_contact_points_from_contour`).
- Height: perpendicular distance apex→substrate.
- Contact angle: spherical-cap approximation from height and diameter (fast, not tangent-based).
- Volume/surface area: revolution integrals in substrate-normal coordinate system.
- Solver: toy Y–L "sphere" fit is not physically consistent for sessile geometry; used only for a placeholder `R0_mm` value.

### 1.4 Known Gaps / Inconsistencies

- Dual pathways lead to divergent settings and results: staged vs. functional analyzer.
- Baseline/axis/apex estimates in `stages.py` are crude; not robust to noise, tilt, or partial occlusion.
- Contact angle estimation via spherical cap is simplistic; lacks local tangent evaluation and does not use substrate line geometry directly.
- Reliance on user-provided substrate line; no automatic baseline detection or validation that the line intersects/underlies the drop.
- Unit handling split across layers; missing mandatory calibration guards.
- Potential API mismatch for `surface_area_mm2` arguments, similar to pendant pipeline.
- Several sessile stage modules are empty; logic is consolidated in `stages.py` and `metrics.py`.

---

## 2) Improvement Opportunities

### 2.1 Performance

- Unify contour extraction across paths to avoid duplication and make preprocessing/edge settings effective consistently.
- Vectorize diameter/base detection using intersections and robust interpolation rather than scanning per unique y.
- Cache curvature and derivative information for contact-point refinement and area/volume integrals.
- Provide optional downscale/ROI refinement strategy for faster previews.

### 2.2 Accuracy and Robustness

- Baseline estimation: implement automatic baseline detection with RANSAC line fitting below the drop and confidence scoring; verify alignment with user line if provided.
- Apex detection: use curvature maximum approach for sessile apex (min y after tilt correction), with subpixel refinement.
- Contact angles: implement tangent-based angle estimation at detected contact points (fit local polynomial/arc around points), offer multiple methods (tangent, spherical cap, circle fit) with method selection and uncertainty estimates.
- Tilt correction: estimate substrate tilt and transform coordinates accordingly before metric calculations.
- Volume/area: reconcile and standardize revolution integrals, ensure monotonic ordering and deduplication along axis.
- Validation: assert nonzero `px_per_mm`, presence of substrate/contact geometry; graceful degrade with actionable messages.

### 2.3 Scalability and Architecture

- Consolidate the two execution paths: drive the GUI's "Analyze" through the staged pipeline to share configuration and results.
- Separate concerns into stage modules (`geometry`, `edge_detection`, `overlay`, `solver`) and keep `stages.py` as orchestration/thin wiring.
- Standardize results schema and units across pipelines and document in a shared contract document.

### 2.4 UX and Workflow

- Baseline tools: add auto-detect button, nudge/lock controls, and confidence indication; snap to edges.
- Contact-point editing: allow manual adjustments with real-time angle updates; visualize tangent lines.
- Results presentation: show left/right angles with method tags and uncertainties; offer residual plots and quality indicators.
- SOP previews: per-stage preview for sessile geometry and edge detection; progress during batch.
- Results Panel integration: pipeline-aware mapping for sessile schema; clearly present `θL/θR`, method used, uncertainties, and baseline tilt.

### 2.5 Error Handling and Observability

- Input validation at each stage (ROI presence, baseline suitability, scale) with explicit `Context.error` and GUI banners.
- Rich logging for detection/fit steps (e.g., RANSAC inliers, tangent fit RMSE), surfaced in a diagnostics panel.
- Timing metrics per stage to help identify bottlenecks.

### 2.6 Integration and Outputs

- Rely on `PIPELINE_MAP` for pipeline resolution in GUI; avoid ad-hoc imports for the functional path when unified.
- Export standardized CSV/JSON with calibration, baseline/contact geometry, chosen angle method, and metrics.
- Expand tests: contact-point finder, angle methods, baseline detection, overlay rendering, and GUI-controller smoke tests.

---

## 3) Proposed Implementation Plan

### 3.1 Goals

- A single, authoritative staged pipeline for sessile runs (both "Analyze" and SOP-driven), producing robust angles and volumes with consistent units.
- Tooling for baseline/contact management and clear error/diagnostic reporting.

### 3.2 Architectural Changes

- Move heavy logic from `stages.py` and functional analyzer to dedicated stage modules; keep `stages.py` as adapter.
- Replace GUI's direct functional call with a wrapper that invokes the staged pipeline and consumes `Context` (`preview`, `results`).
- Define a sessile results contract (keys/units) and align `ResultsPanel` mapping accordingly.

### 3.3 Feature Roadmap (Phased)

Phase 0 — Contracts and groundwork (1–2 days)

- Define results schema: `{diameter_mm, height_mm, theta_left_deg, theta_right_deg, volume_uL, drop_surface_mm2, baseline_tilt_deg, method}`; document units and methods.
- Fix API mismatches (e.g., `surface_area_mm2` usage) and add missing imports/guards in common geometry if needed.
- Confirm `PIPELINE_MAP` exposure and remove redundant import paths in GUI where applicable.

Phase 1 — Unify execution path (1–2 days)

- Make GUI "Analyze" call into staged pipeline; populate preview from `ctx.preview/overlay` and results from `ctx.results`.
- Factor current metrics logic into `do_geometry` and friends; ensure edge detection settings are respected.

Phase 2 — Detection and geometry accuracy (3–5 days)

- Auto baseline detection via RANSAC with heuristics (candidate region below drop, edge map filtering); expose confidence and allow override.
- Apex refinement using curvature maxima with subpixel fit.
- Interpolated base diameter via intersections along substrate; robust to pixelation and tilt.

Phase 3 — Contact angle methods (3–5 days)

- Implement tangent-based contact angle estimation near contact points (poly/arc fit within local window) with uncertainty.
- Provide method selection: `tangent`, `spherical_cap`, `circle_fit`; expose method tag and quality indicators.
- Update results schema and UI to reflect method and confidence.

Phase 4 — UX, diagnostics, and exports (2–4 days)

- Baseline/contact tools in UI: auto-detect, lock, snap, manual adjust with immediate recompute and overlays for tangent lines.
- Diagnostics: residual/error plots and logs; progress indicators for batch runs.
- Results panel: pipeline-aware mapping (sessile schema), include units, method tag, and uncertainty display; exports: CSV/JSON with provenance.

### 3.4 Detailed Tasks per Phase

- P0
  - Draft `docs/contracts/sessile_results.md` and align `ResultsPanel` labels.
  - Reconcile `surface_area_mm2` usage; add unit tests for volume/surface area integrals with simple shapes.

- P1
  - Introduce a unified controller method to run a staged sessile analysis for the current view; update GUI wiring.
  - Move diameter/height/contact-line computations into `do_geometry`; keep functional analyzer for tests/CLI if desired.

- P2
  - Implement and unit-test baseline detection (RANSAC) on samples; provide fallback to user line.
  - Replace base and height computations with interpolation in tilt-corrected coordinates; add robustness tests.

- P3
  - Implement tangent-based contact angle; validate on synthetic profiles and sample images; provide error bars.
  - Expose angle method selection in settings; update overlay to show tangent lines.

- P4
  - Extend overlay and dialogs for sessile-specific tools; add export with provenance.
  - Add progress and summary for batch; smoke tests for controller flows.

### 3.5 API Contracts and Data Model

- Context inputs: `image_path|frames`, `roi`, `substrate_line`, optional `contact_points`, `px_per_mm`, densities.
- Intermediates: `contour.xy`, `geometry.baseline_y|tilt_deg`, `geometry.apex_xy`.
- Outputs (`results`): `{diameter_mm, height_mm, theta_left_deg, theta_right_deg, volume_uL, drop_surface_mm2, baseline_tilt_deg, method, residuals?}`.

### 3.6 Dependencies and Tooling

- Imaging: OpenCV and/or scikit-image for edge maps and line detection; RANSAC implementation can be internal or from `skimage.measure`.
- Numerical: `numpy`; optionally `scipy` for robust fitting.
- Tests: pytest fixtures from `data/samples`; golden metrics for reference.

### 3.7 Estimates (rough, engineer-days)

- Phase 0: 1–2
- Phase 1: 1–2
- Phase 2: 3–5
- Phase 3: 3–5
- Phase 4: 2–4

Total: 10–18 days depending on method breadth and testing depth.

---

## 5) Physics Background and Research Notes (Sessile)

- Contact angle definitions and conventions
  - Young’s equilibrium angle is defined at the three-phase line on an ideal, smooth, homogeneous surface; in practice, we measure an apparent angle that can differ due to roughness, heterogeneity, and hysteresis.
  - Report left/right angles on asymmetric drops; when tilt is small and surface homogeneous, also report the mean `θ = (θL + θR)/2`.

- Measurement methods
  - Tangent method (recommended): fit a local polynomial/arc near each contact point and evaluate the tangent angle relative to the substrate; robust and widely used when windows are chosen carefully.
  - Spherical-cap method: infer angle from base diameter and height; fast but sensitive to gravity (Bond number) and non-spherical shapes.
  - Circle/arc fit: fit a circle to a larger portion of the profile; can bias angles on flattened (high-Bond) drops.
  - Young–Laplace fit (optional): fit a full Y–L axisymmetric profile to estimate `γ` or angle under known `γ`; heavier but physically grounded.

- Substrate and tilt
  - Baseline must reflect the physical substrate; detect automatically (e.g., RANSAC on edge map) and allow user override. Estimate tilt and transform coordinates so angle computation occurs in the substrate frame.
  - Validate that the baseline underlies and intersects the drop footprint; report confidence/inliers.

- Dimensionless groups and regimes
  - Bond number `Bo = Δρ g R^2 / γ` governs gravity-induced flattening; high `Bo` reduces the validity of spherical-cap assumptions and favors tangent/Y–L methods.
  - Capillary length `ℓ_c = sqrt(γ/(Δρ g))` contextualizes expected curvature; use to choose window sizes for local fits.

- Assumptions and imaging considerations
  - Static, equilibrium drops; negligible evaporation and vibrations; constant `Δρ`, `γ`, and `g`.
  - Clean substrate and liquid; minimal contamination/surfactants.
  - Imaging: orthogonal view, minimal lens distortion, correct pixel aspect ratio, reliable `px_per_mm` calibration; adequate resolution near the contact region.

- Method choices and uncertainties
  - Choose tangent-fit window size adaptively (based on `ℓ_c` or pixel scale); avoid including the footprint kink or far-field curvature.
  - Use robust losses (e.g., `soft_l1`) in local fits; report per-side fit RMSE and translate to angle uncertainty.
  - When both tangent and spherical-cap angles are available, report both and flag large discrepancies (> a threshold) for user review.

- Validation strategy
  - Reference surfaces/liquids (e.g., PTFE, glass) with literature contact angles as sanity checks.
  - Synthetic profiles with known baseline tilt and added noise to test robustness and uncertainty estimation.
  - Sensitivity analyses: perturb `px_per_mm`, baseline tilt, and window sizes; document expected angle variability.

- In-repo references to consult
  - `docs/guides/physics_models.md`
  - `docs/guides/numerical_methods.md`
  - `src/menipy/models/surface_tension.py`
  - `src/menipy/common/geometry.py`

---

## 6) Results Panel Integration (Sessile)

- Goals
  - ResultsPanel displays sessile-specific metrics and units when the sessile pipeline is active.
  - Present angle method and uncertainties prominently; include baseline tilt if estimated.

- Behavior
  - Detect active pipeline from execution context and pick the sessile results schema for labels/units.
  - Core metrics: `diameter_mm`, `height_mm`, `theta_left_deg`, `theta_right_deg`, mean angle (optional), `volume_uL`, `drop_surface_mm2`, `baseline_tilt_deg`.
  - Method/uncertainty: display `method` tag (`tangent`, `spherical_cap`, etc.) and per-side fit RMSE or confidence.

- Tasks
  - Define sessile results schema in `docs/contracts/sessile_results.md` with key names, units, and descriptions; include optional diagnostics keys.
  - Update ResultsPanel to select labels based on pipeline key (sessile) and format angles/uncertainties consistently.
  - Add formatting helpers and unit display aligned with schema; handle absent fields gracefully.
  - Ensure batch export includes schema version, pipeline name, and method metadata.
  - Cross-link: see `docs/contracts/sessile_results.md` for canonical keys/units and angle method tags.
  - Implement pipeline-aware switching: detect active pipeline from context, pick corresponding schema map, render values/uncertainties with unit-aware formatting; include `schema_version`/`pipeline`/`method` in exports.

- Acceptance
  - Switching pipelines updates labels/units appropriately; sessile runs show the expected angles with method and uncertainties.
  - Unknown or extra keys do not break layout and are clearly indicated.

### Diagnostics Tab Outline (Sessile)

- Summary: method used, per-side RMSE and uncertainty, baseline tilt, footprint length.
- Plots: local fit windows with tangent overlays; angle vs. window size sensitivity curve (optional).
- Timings: per‑stage `timings_ms` and total runtime.
- Provenance: schema version, pipeline name, image path, calibration; substrate line and contact points.

---

## 4) Risks, Dependencies, Prerequisites

### 4.1 Technical Risks

- Baseline detection sensitivity to reflections/shadows; RANSAC parameter tuning.
- Contact angle variability across methods; communicating uncertainty and method choice.
- Calibration/tilt issues leading to biased metrics; need robust UI guidance.
- Performance impacts from fitting and line detection on large images; mitigated by ROI and downsampling.

### 4.2 Dependencies

- Reliable calibration (`px_per_mm`) and substrate annotation (or auto-detect) are necessary for valid angles.
- Availability of numerical and image-processing libraries; Qt threading considerations for previews.

### 4.3 Prerequisites

- Agree on the sessile results contract and UI presentation of angles/methods.
- Decide on deprecating the direct functional path in GUI in favor of staged runs.
- Approve dependency choices (e.g., skimage or internal RANSAC).

---

## Acceptance Criteria

- Single-path staged pipeline powers "Analyze" and SOP; consistent overlays and results on sample images.
- Contact angles from tangent method validated against references; spherical-cap value available as a secondary method.
- Clear diagnostics and error messages for missing/invalid baseline/scale; batch processing exports consistent CSV/JSON with provenance.
- Tests cover baseline detection, contact points, angle methods, and overlay drawing; CI passes.

---

## Appendix: Reference Files

- Pipeline base/discovery
  - `src/menipy/pipelines/base.py`
  - `src/menipy/pipelines/discover.py`

- Sessile pipeline
  - `src/menipy/pipelines/sessile/stages.py`
  - `src/menipy/pipelines/sessile/geometry.py`
  - `src/menipy/pipelines/sessile/metrics.py`

- GUI integration
  - `src/menipy/gui/controllers/pipeline_controller.py`
  - `src/menipy/gui/services/pipeline_runner.py`
  - `src/menipy/gui/overlay.py`
  - `src/menipy/gui/panels/results_panel.py`

---

Comments, suggestions, and prioritization feedback are welcome.
