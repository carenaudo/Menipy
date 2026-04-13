# Needle Diameter Detection and Pixel-to‑mm Calibration Plan

## Goal
Build a robust, reusable tool to detect a needle’s outer diameter (OD) in pixels and convert image pixels to millimeters using a known physical needle OD. The tool must work for both sessile and pendant pipelines, support videos and still images, and expose confidence/uncertainty for downstream use.

## Outcomes
- Automatic pixel-to-mm scale: `scale_mm_per_px = OD_mm / diameter_px` with propagated uncertainty.
- Works across sessile and pendant experiments; reusable core in `src/menipy/common`.
- GUI stage for calibration with ROI tools, preview overlays, manual override, and status feedback.
- Batch/video support with robust aggregation across frames.
- Tests with synthetic and real images; clear failure modes and recovery.

## Assumptions and Constraints
- Needle is approximately a rigid cylindrical shaft with two strong, near-parallel edges in the ROI.
- Orientation is often near-vertical (pendant) or near-vertical/tilted (sessile). We will not assume exact axis a priori.
- OD is known by the user in physical units (e.g., 0.51 mm). Integration with Pint for unit safety.
- High-contrast edges are typical but not guaranteed; glare and reflections occur.
- Performance target: < 100 ms/frame on 1080p ROI on a modern CPU.

## User Experience
- Dedicated calibration stage in GUI with:
  - ROI tool to bound the needle region.
  - Live overlay of detected axis and measured width with confidence.
  - Input field for known OD + unit selector; remembers last value per session.
  - Manual alternate mode: user draws a cross-section line across the shaft; tool measures diameter on that line.
  - Result banner with `mm/px`, `px/mm`, and uncertainty; apply to session.
- Pipeline behavior:
  - If calibration exists for a source, reuse it; otherwise prompt/run calibration.
  - Store calibration in results metadata for reproducibility.

## Architecture
- Core module: `src/menipy/common/needle_calibration.py`
  - Stateless detection functions + small dataclasses for results/debug.
- Pipeline adapters:
  - `src/menipy/pipelines/sessile/stages.py`: optional or early stage to run calibration.
  - `src/menipy/pipelines/pendant/stages.py`: same integration point.
- GUI controller:
  - `src/menipy/gui/controllers/pipeline_controller.py` hooks to launch calibration stage.
  - `src/menipy/gui/panels/results_panel.py` displays calibration summary.
- Units: use existing Pint integration (`src/menipy/models/unit_types.py`) to ensure `Quantity` I/O and storage.

## Detection Strategy (Core)
1. Preprocessing
   - Convert to grayscale; denoise (bilateral or fast NLM) to preserve edges.
   - Contrast normalize (CLAHE) within ROI; optional glare suppression by clipping highlights.
2. Edge Extraction
   - Adaptive Canny (thresholds from image median/MAD) or Sobel magnitude.
   - Optional edge-thinning via non-maximum suppression.
3. Axis Hypothesis
   - Probabilistic Hough transform to get dominant line(s); RANSAC line fitting on edge points.
   - Candidate axes = lines with highest inlier support; keep those within plausible orientation (not strictly vertical to allow tilt).
4. Ribbon (Parallel Edge Pair) Detection
   - For each axis candidate, sweep short normals across the line; along each normal profile, locate two strong opposite gradients (left/right edges).
   - Robust pairing: maximize parallelism, minimize width variance along axis, maximize gradient magnitudes and inlier count.
   - Reject spurious pairs by enforcing near-constant width and low curvature of edges.
5. Subpixel Refinement
   - Fit 1D sigmoid/erf to intensity across each edge on profiles to refine edge location to subpixel accuracy.
   - Aggregate refined widths over many profiles to get `diameter_px_mean` and `diameter_px_std`.
6. Multi-frame Aggregation (Video)
   - Detect over N frames spread across the clip; temporal median for robustness; reject outliers via MAD.
   - Optional simple Kalman filter for axis stabilization; width measured in axis-aligned coordinates.
7. Confidence and Uncertainty
   - Confidence from: edge support, parallelism score, Hough/RANSAC residuals, SNR of profiles.
   - Uncertainty combines subpixel fit variance, inter-profile width variance, inter-frame variance.

## Fallbacks and Recovery
- Template matching: correlation with synthetic bar templates (various widths) if edge pairing fails.
- Manual assist: user draws a single cross-section line; tool performs profile fit only on that line.
- Heuristic relaxations: widen Canny thresholds, enlarge ROI, adjust expected needle orientation band.
- Fail closed: return no calibration with actionable reasons and suggested next step.

## Data Structures and API
- Dataclasses
  - `NeedleCalibrationResult`: `scale_mm_per_px: float`, `px_per_mm: float`, `diameter_px: float`, `uncertainty_px: float`, `confidence: float`, `roi: tuple`, `axis: LineModel`, `method: str`, `frames_used: int`.
  - `NeedleDetectionDebug`: sampled profiles, edge points, Hough lines, overlays.
- Core API (proposal)
  - `detect_needle_diameter(image_or_frame, roi=None, expected_orientation=None, debug=False) -> tuple[float, NeedleDetectionDebug]`
  - `estimate_scale_from_known_od(diameter_px, od: Quantity) -> Quantity`  // returns `mm/px` with Pint units
  - `calibrate_from_media(media_source, od: Quantity, roi=None, frame_count=10, seed=None, debug=False) -> NeedleCalibrationResult`

## Integration Points
- Sessile/Pendant pipelines
  - Add optional calibration stage before geometry/metrics to produce scale in pipeline context.
  - Store calibration under results metadata for later use and export (see `docs/contracts/sessile_results.md`).
- GUI
  - New Calibration dialog/panel with ROI selector and live overlay.
  - Persist last OD per user profile; allow unit entry with Pint-backed parser.

## Implementation Steps
1. Core detection prototype in `src/menipy/common/needle_calibration.py` using OpenCV/NumPy.
2. Unit handling utilities leveraging `src/menipy/models/unit_types.py` and existing Pint integration.
3. Synthetic test generators for cylinders/needles with noise, blur, tilt, glare.
4. Unit tests: profile extraction, edge refinement, width statistics, uncertainty composition.
5. Video aggregation and outlier rejection; benchmarks on sample clips.
6. GUI calibration panel + overlays; manual cross-section fallback.
7. Pipeline integration; persist calibration results and display in results panel.
8. Documentation and examples; add sample media and how-to guide.

## Testing Plan
- Synthetic images: varied widths, orientations, blur, noise, contrast, and partial occlusions; ensure <1% width error at SNR ≥ 10.
- Real lab images/videos: at least 20 samples per pipeline; target <2% median error vs. ruler/caliper ground truth.
- Regression tests for failure cases (glare, poor contrast); confirm graceful fallback.
- Property tests: invariance to crop/ROI translation; small rotation invariance.

## Performance Considerations
- Operate on ROI; downscale for Hough then refine at full ROI.
- Vectorized profile sampling; optional numba if needed.
- Avoid heavy dependencies; prefer OpenCV + NumPy; consider scikit-image only if justified.

## Deliverables
- `src/menipy/common/needle_calibration.py` (core)
- GUI calibration panel and controller hooks
- Pipeline stage wiring for sessile and pendant
- Tests under `tests/` with synthetic generators
- Documentation: this plan, developer guide, and user how-to

## Milestones
- M1 Prototype core detection on images (2–3 days)
- M2 Subpixel refinement + uncertainty (1–2 days)
- M3 Video aggregation + confidence scoring (1–2 days)
- M4 GUI calibration panel + manual fallback (3–4 days)
- M5 Pipeline integration + results persistence (1–2 days)
- M6 Test suite + sample datasets + docs (2–3 days)

## Risks and Mitigations
- Strong reflections/glare: profile fitting with robust loss; glare clipping preproc.
- Very low contrast: adaptive CLAHE; prompt user to adjust ROI/lighting.
- Non-cylindrical features nearby: width variance and parallelism checks reject impostors.
- Needle tip only visible: enforce minimum axis length; fallback to manual cross-section.

## Acceptance Criteria
- Pixel-to-mm scale accuracy within ±2% on real datasets; ±1% on synthetic.
- Confidence score correlated with accuracy (R>0.7 on validation set).
- End-to-end pipeline consumes scale seamlessly; results panels display calibration with units and uncertainty.
