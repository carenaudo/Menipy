# Pendant Drop Pipeline — Current State and Enhancement Plan

This document analyzes the current “pendant” pipeline (processing and GUI), identifies improvement opportunities, and proposes an actionable plan to enhance performance, accuracy, UX, and maintainability. It is intended for developers and stakeholders to converge on scope and priorities before implementation.

## Scope

- Pipeline: `pendant` only (processing + GUI integration)
- Codebase baseline: repo at time of analysis

---

## 1) Current Implementation Summary

### 1.1 Key Components (by module)

- Core pipeline base
  - `src/menipy/pipelines/base.py:35` — `PipelineBase` template class and stage orchestration (`DEFAULT_SEQ`, `run`, `run_with_plan`, error/timing handling).
  - Stages supported: acquisition → preprocessing → edge_detection → geometry → scaling → physics → solver → optimization → outputs → overlay → validation.

- Pendant pipeline
  - `src/menipy/pipelines/pendant/stages.py:38` — `PendantPipeline(PipelineBase)` simplified implementation:
    - Ensures contour via common edge detection; geometry extracts vertical axis (median x) and apex (max y).
    - Sets default scaling (`px_per_mm=1.0`) and physics (`rho1, rho2, g`).
    - Solver: calls a toy Young–Laplace integrator plugin to fit apex radius `R0_mm` with `FitConfig`.
    - Outputs: maps fit params to `ctx.results` and adds residuals; overlay draws contour, symmetry axis, apex cross, and a text label of `R0`; validation passes solver success.

  - `src/menipy/pipelines/pendant/geometry.py:30` — Functional analyzer path used by the GUI’s “simple analysis”:
    - `analyze(frame, helpers)` → `extract_external_contour` → `find_pendant_apex` → `compute_pendant_metrics` → returns `PendantMetrics` with contour, apex, diameter line, and derived metrics.
    - `HelperBundle` inputs: `px_per_mm`, `needle_diam_mm`, `delta_rho`, `g`, `apex_window_px`.

  - `src/menipy/pipelines/pendant/metrics.py:14` — `compute_pendant_metrics(...)` placeholder implementation:
    - Computes: droplet height/diameter in px, converts to mm, apex curvature radius via circle fit near apex window, shape factor s1, Jennings–Pallas `beta`, surface tension `gamma`, volume by revolution, needle/drop surface areas.
    - Notes: logic is a first-pass; some interfaces are inconsistent with helpers in other modules (see gaps below).

- Common maths/geometry
  - `src/menipy/models/surface_tension.py:1` — `jennings_pallas_beta`, `surface_tension`, `volume_from_contour`, `bond_number`.
  - `src/menipy/models/drop_extras.py:1` — extra pendant metrics (Worthington, apex curvature, projected/surface area, apparent weight).
  - `src/menipy/common/geometry.py` — geometric utilities, including `fit_circle`, profile intersections, curvature estimates, and contact-point finding.

- GUI integration (controllers, services, panels)
  - `src/menipy/gui/controllers/pipeline_controller.py:12` — Orchestrates runs from the main window; two flows:
    - Full/step pipeline run via `PipelineBase` subclass (through discovery map).
    - Simple analysis path that imports `menipy.pipelines.<mode>.geometry.analyze` and `...drawing.draw_<mode>_overlay` for direct compute + overlay, bypassing the stage runner.
    - Collects overlays (ROI, needle rect, optional contact line), pulls preprocessing/edge-detection settings from dedicated controllers, updates preview/results panels.
  - `src/menipy/gui/services/pipeline_runner.py:86` — Threaded runner for full/partial stage execution; patches acquisition, runs `run` or `run_with_plan`, emits context for GUI updates.
  - `src/menipy/gui/panels/results_panel.py:23` — Displays `results` mapping into a 2-column table with friendly labels.
  - `src/menipy/gui/panels/setup_panel.py` — Source selection (single, batch, camera), pipeline selection, SOP list, and “Analyze” (simple path) binding.
  - `src/menipy/gui/overlay.py` and `src/menipy/pipelines/pendant/drawing.py` — Compose QPainter-based overlays from metrics (contour/polyline, lines, points, labels).

### 1.2 Data Flow

- Full pipeline path
  - User configures sources and SOP → `PipelineController.run_full/run_all` → `PipelineRunner` starts background job → `PipelineBase.run` sequences stages, mutating a shared `Context`.
  - Results arrive on GUI thread; preview and results panels update from `ctx.preview`, `ctx.results`.

- Simple analysis path (current GUI default for “Analyze”)
  - UI extracts cropped ROI image and calibration/auxiliary inputs → calls `pendant.geometry.analyze(...)` → returns `PendantMetrics` → `pendant.drawing.draw_pendant_overlay(...)` renders overlay, `ResultsPanel.update(...)` shows `metrics.derived`.

- Contour acquisition
  - In `stages.py`, ensured via `menipy.common.edge_detection.run(...)` using the controller-provided settings or default `canny` if none.
  - In `geometry.analyze`, `extract_external_contour(frame)` is called directly (bypassing the edge detection controller).

### 1.3 Algorithms (current)

- Edge detection: Canny (default) via common edge-detection utilities (plugin-capable).
- Axis/apex estimation (pipeline path): symmetry axis as median x of contour; apex as max y index.
- Metric computation (simple path):
  - Height/diameter via per-scanline width; apex radius by local circle fit; s1, beta (Jennings–Pallas), gamma from Young–Laplace analytic relation; volume and surface area by revolution.
- Solver (pipeline path): toy Young–Laplace “sphere” integrator fit for `R0_mm` using `FitConfig` and `common_solver` scaffolding.

### 1.4 GUI Elements Tied to Pendant

- ROI, needle rect, and optional contact line captured from `PreviewPanel` and used for acquisition/analysis.
- Calibration tab provides `px_per_mm`, liquid/air densities, and needle diameter (pendant-specific helpers).
- Overlay styling configurable via dialogs (`GeometryConfigDialog`, `OverlayConfigDialog`); controllers propagate preview frames for interactive tuning.
- SOP support: selective stage runs and per-stage configuration actions (dialog-based).

### 1.5 Known Gaps / Inconsistencies Observed

- Dual pathways: the GUI “Analyze” uses the functional `analyze(...)` path; the “Run full” uses the staged pipeline. The two produce different result schemas and overlays and diverge on responsibilities (edge detection handled differently).
- Placeholders and empties: multiple pendant modules (`acquisition.py`, `preprocessing.py`, `physics.py`, `solver.py`, `overlay.py`, etc.) are empty, while `stages.py` inlines behavior. This complicates testing, documentation, and extension.
- API mismatches:
  - `surface_area_mm2` usage in `pendant/metrics.py` appears inconsistent with the function signature in `models/drop_extras.py` (expects pixel-based contour and `px_per_mm`).
  - `common/geometry.py` utilities have potential import/robustness issues (e.g., missing `lstsq` import for circle fit) and lack validation/typing in places.
- Unit handling: defaults set in `stages.py` and `HelperBundle` may mask missing calibration; unit conversions are spread across modules.
- Error handling: GUI shows warnings for missing overlays, but pipeline-level validation is minimal beyond solver success; missing ROI/scale/invalid inputs can propagate to exceptions.

---

## 2) Improvement Opportunities

### 2.1 Performance

- Unify edge detection and contour extraction across both paths to avoid redundant work and inconsistent settings.
- Vectorize diameter/height computation using `horizontal_intersections` to avoid O(N·unique_y) loops; consolidate to a single pass with robust interpolation.
- Cache/precompute contour derivatives for curvature estimation and revolution integrals; reduce allocations/copies.
- Add optional downscale for initial detection with re-refinement at full resolution near apex/neck.
- Threaded batch processing via existing `PipelineRunner`; enable progress updates for large images.

### 2.2 Accuracy and Robustness

- Apex detection: move from “max y” to curvature-based apex with subpixel refinement; support tilted camera corrections if needed.
- Symmetry axis: replace median x with robust axis estimation (e.g., minimal radius variance axis, or fit symmetric model); optionally use needle region alignment.
- Diameter: compute as maximum of interpolated horizontal intersections, not unique y bins; optionally smooth contour to suppress pixelation.
- Apex radius: use polynomial fit or robust circle/arc fit in an adaptive apex neighborhood; quantify uncertainty.
- Surface tension: add a proper Young–Laplace solver (arc-length parameterization, boundary value problem) and fit to the full profile instead of toy sphere model; keep the Jennings–Pallas method as a rapid estimate + cross-check.
- Volume/area: reconcile API for surface area; validate monotonic ordering and deduplicate y; handle open/partial contours.
- Units/scale: centralize unit conversion; mandatory calibration check with warning/guard rails.

### 2.3 Scalability and Architecture

- Consolidate the two paths: make the functional “simple analysis” call into the same staged pipeline (single entry point), or promote the staged pipeline to be the only path and expose helper convenience wrappers for GUI.
- Fill out pendant module stages (acquisition, preprocessing, edge_detection, geometry, scaling, physics, solver, overlay) to avoid behavior inlining in `stages.py` and support testing and reuse.
- Standardize results schema across pipelines (keys and units) and document in a shared contract.
- Strengthen plugin integration for solvers/edge detectors; move “toy” solver under a consistent plugin strategy and hide from production runs by default.

### 2.4 UX and Workflow

- Provide stage-level preview for pendant geometry and solver residuals; add progress and tooltips on overlays (e.g., show `De`, `R0`, `β`, error bars).
- Add wizards for calibration and ROI/needle/contact guidance; highlight invalid/missing inputs early.
- Make Analyze/Run All buttons consistent (both use staged pipeline); keep an “instant estimate” mode behind an option if needed.
- Results Panel integration: make the results table pipeline-aware. Define a pendant results schema and map keys→labels/units dynamically based on the active pipeline; include a diagnostics subview (residuals, timings) specific to pendant.

### 2.5 Error Handling and Observability

- Validate inputs at stage boundaries; fail fast with helpful messages in `Context.error` and GUI banners.
- Enrich `ctx.timings_ms` and add per-stage counters; surface in GUI diagnostics panel.
- Add structured logs for solver convergence (iterations, loss, constraints); display residual plots in results panel.

### 2.6 Integration and Outputs

- Centralize pipeline discovery and avoid direct `importlib` loads in GUI; rely exclusively on `PIPELINE_MAP`.
- Standard export (CSV/JSON) of results and provenance (image path, calibration, settings, versions) for the pendant pipeline; enable batch export.
- Tighten tests: unit tests for geometry metrics, solver regressions, overlay drawing contracts, GUI-controller smoke tests.

---

## 3) Proposed Implementation Plan

### 3.1 Goals

- Single-source pendant implementation: the staged pipeline is the authoritative path; GUI calls it for both “Analyze” and “Run”.
- Improved accuracy (apex/diameter/axis; full Y–L fit) and consistent units/outputs.
- Better UX and error handling, with predictable performance on single and batch runs.

### 3.2 Architectural Changes

- Refactor `pendant` package to move stage logic out of `stages.py` into their respective modules; keep `stages.py` as orchestration/wiring only.
- Replace GUI’s direct `analyze`/`drawing` path with a thin wrapper that invokes the staged pipeline and adapts context to overlays.
- Define a results contract (schema + units) for pendant and adopt it in `ResultsPanel` label map and docs.

### 3.3 Feature Roadmap (Phased)

Phase 0 — Contracts, wiring, and cleanup (1–2 days)

- Define pendant results schema and document it (keys: `De_mm`, `H_mm`, `R0_mm`, `beta`, `gamma_mN_m`, `V_uL`, `A_mm2`, diagnostics).
- Fix API mismatches and minor issues identified (e.g., `surface_area_mm2` signature use; missing imports in common geometry; empty stage modules structure).
- Ensure `PIPELINE_MAP` discovery returns pendant pipeline; remove ad-hoc module importing in GUI where possible.

Phase 1 — Unify paths (1–2 days)

- Make `SetupPanelController.runAllBtn` drive the staged pipeline (or provide a unified `PipelineController.analyze_current_view()` that uses the staged run under the hood).
- Have `PipelineController._run_analysis_and_update_ui` read from `Context` (`preview`, `overlay`, `results`) populated by pendant stages; retire `geometry.analyze` for GUI.
- Fill pendant stage modules with logic currently in `stages.py` and `geometry.py/metrics.py` where appropriate.

Phase 2 — Geometry and metrics accuracy (3–5 days)

- Edge detection: standardize on a common configuration path; validate ROI/needle rect usage.
- Apex: curvature-based apex with adaptive window; add subpixel refinement.
- Axis: robust axis estimation (minimize asymmetry metric), with fallback to median.
- Diameter: compute via `horizontal_intersections` with interpolation; add optional smoothing.
- Volume/surface area: reconcile function signatures and implement consistent mm-based computation with deduplication and monotonic y.
- Unit checks: require nonzero `px_per_mm`; issue error/warning otherwise.

Phase 3 — Young–Laplace solver integration (5–8 days)

- Integrate a production-ready Y–L solver (arc-length parameterization, boundary conditions from `R0`, density difference, gravity); use `common_solver` infrastructure with `FitConfig`.
- Fit parameters: at minimum `R0_mm`; optionally `Bond` or `gamma` depending on chosen formulation.
- Provide residuals vs. measured profile; expose convergence diagnostics in results.
- Keep Jennings–Pallas as a fast estimate and cross-check in outputs.
- Physics research deliverables: confirm correlation validity ranges (e.g., `s1` domain for Jennings–Pallas), select solver formulation and residual metric, define initialization/bounds, and assemble validation datasets (synthetic and reference liquids).

Phase 4 — UX, diagnostics, and exports (2–4 days)

- Overlay: richer annotations (axis, apex, `De`, `R0`, residual heatmap toggle); configurable via Overlay dialog.
- Results panel: pipeline-aware mapping (pendant schema), add units; enable residual plot tab and solver diagnostics; export CSV/JSON with provenance.
- Batch mode: progress with per-image status; consolidated report.

### 3.4 Detailed Tasks per Phase

Selected high-level tasks (non-exhaustive):

- P0
  - Create `docs/contracts/pendant_results.md` and update `ResultsPanel` labels accordingly.
  - Fix `pendant/metrics.py` surface-area call to match `models/drop_extras.surface_area_mm2` signature; add tests.
  - Patch `common/geometry.fit_circle` to import `lstsq` and add minimal input validation.

- P1
  - Add a unified `PipelineController.analyze_current_view()` that wraps the staged pipeline and populates the preview via `ctx.preview/overlay`.
  - Deprecate the direct `geometry.analyze` path in GUI; keep for testing/CLI only if needed.
  - Move geometry computations from functional path into `do_geometry` stage; ensure edge detection settings are honored consistently.

- P2
  - Implement `find_pendant_apex` refinement and axis estimator; unit-test on sample images under `data/samples`.
  - Replace diameter scan with interpolated intersections; unit-test robustness vs. noise and pixelation.
  - Rework volume/surface area helpers to accept mm inputs; add tests for degenerate contours.

- P3
  - Introduce a `pendant/solver.py` module exporting a proper Y–L integrator; wire into `stages.py` via `common_solver.run`.
  - Add `FitConfig` variants; tune bounds, losses, and distance metrics; unit-test with synthetic profiles.
  - Expose residuals and fit quality metrics in `ctx.results`.

- P4
  - Extend overlay commands to include residual visualization toggle; integrate with `OverlayConfigDialog`.
  - Implement `ResultsPanel` tabs for metrics and diagnostics; add “Export” actions.
  - Add `PipelineRunner` progress signals and GUI progress bar for batch runs.

### 3.5 API Contracts and Data Model

- Context keys expected/produced by pendant:
  - Inputs: `image_path|frames`, `roi`, `needle_rect`, `px_per_mm`, `delta_rho`, `g`.
  - Intermediates: `contour.xy`, `geometry.axis_x`, `geometry.apex_xy`.
  - Outputs: `results` dict with at least `{De_mm, H_mm, R0_mm, beta, gamma_mN_m, V_uL, A_mm2}`; `overlay` commands for GUI.

- Results schema documented and unit-bearing; downstream panels read through stable keys (avoid ad-hoc ones like `s1` unless documented).

### 3.6 Dependencies and Tooling

- Numerical: `scipy` (optimize/integrate) for Y–L solver if not already present; verify in `requirements.txt`.
- Imaging: `opencv-python` and/or `scikit-image` for robust edge detection and subpixel refinement.
- Testing: expand pytest coverage for geometry/solver; provide fixtures from `data/samples`.

### 3.7 Estimates (rough, engineer-days)

- Phase 0: 1–2
- Phase 1: 1–2
- Phase 2: 3–5
- Phase 3: 5–8
- Phase 4: 2–4

Total: 12–21 days depending on solver maturity and test scope.

---

## 4) Risks, Dependencies, Prerequisites

### 4.1 Technical Risks

- Solver complexity: full Y–L BVP fitting can be sensitive to initialization and noise; requires careful parameterization and robust distance metrics.
- Edge cases: small or near-detachment drops (high Wo), partial/occluded contours, specular highlights, tilted needle/camera misalignment.
- Performance: high-resolution images and iterative solvers may increase latency; need caching/downsampling strategies.
- API churn: unifying the two paths may impact existing GUI behaviors; coordinate migration and communication.

### 4.2 Dependencies

- Accurate calibration (`px_per_mm`, densities) is mandatory for physically meaningful results; enforce in UI.
- Consistent ROI and needle/contact annotations; provide guidance tooling.
- Numerical libraries availability (`numpy`, `scipy`); Qt runtime stability for threaded jobs.

### 4.3 Prerequisites

- Decide on the pendant results contract and UI presentation (labels/units).
- Confirm whether “simple analysis” should be deprecated in the GUI or kept as a fast estimate behind an option.
- Approve adding/locking external dependencies (e.g., SciPy) if not already present.

---

## 5) Physics Background and Research Notes

- Governing equations
  - Young–Laplace: the pressure jump across the interface is `Δp = γ (1/R1 + 1/R2)`; along a pendant profile in hydrostatic equilibrium, `Δp(z) = Δρ g z + const` couples curvature to depth.
  - Axisymmetric shape parameterization via arc length with apex boundary conditions (`z=0`, `R0 = 1/κ0`), integrated to generate a theoretical profile for given `γ` (or `R0` given `γ`).

- Dimensionless groups and descriptors
  - Bond number: `Bo = Δρ g R0^2 / γ` governs gravity–capillarity balance; capillary length `ℓ_c = sqrt(γ/(Δρ g))` provides a useful scale.
  - Shape factor `s1 = De/(2 r0)` with empirical correlation `β = f(s1)` (Jennings–Pallas) to estimate `γ` via `γ = Δρ g r0^2 / β`.
  - Validity ranges to respect: typical correlations calibrated for approximately `0.5 ≤ s1 ≤ 2.0`; outside this range, errors grow.

- Methods to infer surface tension
  - Correlation methods: use apex radius and max diameter with a literature `β(s1)` correlation; fast, single-shot, but sensitive to local noise and outside-range use.
  - Full-profile fit: integrate Y–L to produce a synthetic profile and fit to the observed contour (distance metrics: normal, orthogonal, or pointwise); slower but more robust and better use of data.

- Assumptions and experimental conditions
  - Static, axisymmetric, equilibrium drops; negligible inertia/viscosity effects; clean interface (no surfactant dynamics); constant `Δρ`, `γ`, and `g` over the acquisition window; rigid needle; small evaporation.
  - Imaging: adequate resolution near apex and neck; minimal tilt; stable illumination/background; accurate pixel–mm calibration and temperature compensation for densities and `γ`.

- Solver formulation notes
  - Use arc-length ODEs with apex BCs; choose shooting or collocation; scale states to improve conditioning; establish safe parameter bounds for `R0`/`γ`.
  - Residual definition matters: prefer orthogonal distance to the curve or projection on local normals; robust loss (e.g., `soft_l1`) helps outlier handling.
  - Initialization: seed from correlation estimate (`β(s1)`) or geometric heuristics; add step controls and line search for stability.

- Validation strategy
  - Synthetic profiles with known `γ`, `Δρ`, `g` to verify solver recovery across `Bo` and noise levels.
  - Reference liquids (e.g., water at temperature) to check absolute accuracy; cross-check against capillary length inferred from profile.
  - Sensitivity studies: perturb `px_per_mm`, `Δρ`, apex window, and contour noise; include in error budget guidance.

- In-repo references to consult
  - `docs/pendant_drop_methods.md`
  - `docs/guides/physics_models.md`
  - `docs/guides/numerical_methods.md`
  - `src/menipy/models/surface_tension.py`

---

## 6) Results Panel Integration (Pendant)

- Goals
  - ResultsPanel displays pendant-specific metrics and units when the pendant pipeline is active.
  - Provide a diagnostics area for pendant (fit residuals summary, `Bo`, timings).

- Behavior
  - Detect active pipeline from execution context and pick the pendant results schema for labels/units.
  - Core metrics: `De_mm`, `H_mm`, `R0_mm`, `beta`, `gamma_mN_m`, `V_uL`, `A_mm2` (with consistent units and descriptions).
  - Diagnostics: optional section/tab listing `residuals` summary, solver success/iterations, `Bo`, timestamps, and stage timings.

- Tasks
  - Define pendant results schema in `docs/contracts/pendant_results.md` with key names, units, and descriptions; include optional diagnostics keys.
  - Update ResultsPanel to select labels based on pipeline key (pendant) and hide unknown keys or render them in a generic section.
  - Add formatting helpers (e.g., SI units, precision per field) driven by schema.
  - Ensure batch export includes schema version and pipeline name for downstream parsing.
  - Cross-link: see `docs/contracts/pendant_results.md` for canonical keys/units used by the ResultsPanel.
  - Implement pipeline-aware switching: detect active pipeline from context, pick corresponding schema map, and render values with unit-aware formatting; include `schema_version` and `pipeline` in exports.

- Acceptance
  - Switching pipelines updates labels/units appropriately; pendant runs show the expected set with units and diagnostics when available.
  - Unknown or extra keys do not break layout and are clearly indicated.

### Diagnostics Tab Outline (Pendant)

- Summary: solver success, iterations, termination reason, RMS residual, Bond number.
- Plots: residuals vs. arc length; optional overlay of fitted vs. measured profile.
- Timings: per-stage `timings_ms` and total runtime; environment info (image size, px_per_mm).
- Provenance: schema version, pipeline name, image path, calibration and density inputs.


## Acceptance Criteria

- A single pendant pipeline path drives both “Analyze” and “Run All”; results and overlays match within expected tolerance on sample images.
- Geometry metrics (De, H, R0) agree against baselines; surface tension from Y–L solver within specified error bounds vs. references.
- Robust error messages for missing/invalid inputs; improved diagnostics available in the GUI.
- Batch mode produces consistent exports with provenance.
- Unit tests added for geometry metrics, solver fitting, and overlay contracts; CI passes.

---

## Appendix: Reference Files

- Pipeline base and discovery
  - `src/menipy/pipelines/base.py:35`
  - `src/menipy/pipelines/discover.py:1`

- Pendant pipeline
  - `src/menipy/pipelines/pendant/stages.py:38`
  - `src/menipy/pipelines/pendant/geometry.py:30`
  - `src/menipy/pipelines/pendant/metrics.py:14`
  - `src/menipy/pipelines/pendant/drawing.py:1`

- Common models and utilities
  - `src/menipy/models/surface_tension.py:1`
  - `src/menipy/models/drop_extras.py:1`
  - `src/menipy/common/geometry.py:1`

- GUI integration
  - `src/menipy/gui/controllers/pipeline_controller.py:12`
  - `src/menipy/gui/services/pipeline_runner.py:86`
  - `src/menipy/gui/panels/results_panel.py:23`
  - `src/menipy/gui/panels/setup_panel.py:1`
  - `src/menipy/gui/overlay.py:1`

---

Questions, clarifications, or changes in priorities welcome.
