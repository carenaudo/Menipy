# Plan: open-source ADSA models and algorithm evaluation

## Objective

Determine which ideas, trained models, and algorithms from
PendantDropMachineLearning, Conan-ML, OpenDrop, Drop-O-Matic, and OpenCapsule
can improve Menipy while preserving scientific validity, maintainability,
runtime size, and license compatibility.

The work ends with an evidence-backed report, not automatic adoption. Every
candidate must be evaluated against the same images, metrics, physical
contracts, and licensing checklist.

## Questions to answer

1. Can the trained models from PendantDropMachineLearning and Conan-ML be
   loaded, converted to ONNX, and reproduced within an acceptable numerical
   tolerance?
2. Do their licenses permit conversion, local experimentation, optional
   download, and redistribution with Menipy?
3. How do OpenDrop, Drop-O-Matic, and OpenCapsule locate or infer the drop,
   needle/capillary, substrate/baseline, contact points, apex, and symmetry
   axis?
4. How does each project calculate contact angle, scale, volume, surface area,
   surface/interfacial tension, Young-Laplace fit parameters, residuals, and
   quality indicators?
5. Which techniques outperform or complement Menipy on accuracy, robustness,
   speed, failure detection, and implementation complexity?
6. Which improvements should be adopted, prototyped, rejected, or deferred?

## Scope and pinned sources

At the start of the research, record the exact commit SHA, retrieval date,
release/tag, model hashes, and license files for:

| Candidate | Official source | Initial license observation |
| --- | --- | --- |
| PendantDropMachineLearning | `github.com/FelixKratz/PendantDropMachineLearning` | GPL-3.0; repository contains Keras `.h5` models |
| Conan-ML | `github.com/jdber1/conan-ml` | GPL-2.0; repository contains training files and datasets |
| OpenDrop | `github.com/jdber1/opendrop` | Verify repository and dependency licenses at pinned commit |
| Drop-O-Matic | `github.com/KrzysztofDorywalski/Drop-O-Matic` | Verify repository and bundled assets at pinned release |
| OpenCapsule | `github.com/jhegemann/opencapsule` | Verify repository and numerical-library licenses at pinned commit |

External repositories must be cloned only into an ignored temporary research
directory. Do not vendor code or weights into Menipy during investigation.
Delete temporary environments and clones after the report is complete.

## Deliverables

1. `docs/research/adsa_open_source_assessment.md`: final detailed report.
2. `docs/research/adsa_algorithm_matrix.csv`: one row per pipeline operation
   and project, including source-file and symbol references.
3. `docs/research/adsa_license_matrix.csv`: code, model, data, documentation,
   redistribution, modification, attribution, and commercial-use findings.
4. `build/research/adsa_benchmark_results.json`: machine-readable experiment
   results; treated as disposable evidence.
5. `build/research/adsa_benchmark_summary.csv`: comparable metrics by sample
   and method.
6. Optional isolated conversion scripts under `scripts/research/` only if they
   are reusable and contain no copied third-party implementation.

## Phase 1: reproducibility and license audit

### 1.1 Repository audit

For every candidate:

- pin a commit or release;
- inventory model, source, dataset, documentation, and example files;
- record language, framework, dependency versions, supported platforms, and
  last meaningful maintenance activity;
- hash every pretrained artifact;
- distinguish repository license from model-weight and dataset licenses;
- identify copied or bundled third-party components with separate terms;
- record citation and attribution requirements;
- flag missing or ambiguous licenses as **not redistributable** until clarified.

### 1.2 License decision categories

Assign each candidate one status:

- **A — redistributable:** compatible code and model terms permit inclusion.
- **B — optional external asset:** usable only through a user download or
  separately installed plugin.
- **C — research reference only:** ideas and measured behavior may be studied,
  but code/weights must not be incorporated.
- **D — unresolved:** written clarification or legal review is required.

The report must discuss GPL-2.0-only versus GPL-3.0 compatibility, whether a
converted ONNX file is considered a derivative of the original weights, and
whether Menipy's distribution model is compatible. This is a compliance
assessment, not legal advice.

## Phase 2: ML conversion feasibility

### 2.1 PendantDropMachineLearning

Models to inspect:

- `model_uniform_rho.h5`;
- `model_uniform_rho_noise.h5`;
- `model_uniform_Wo.h5`.

Tasks:

1. Load each model in a disposable TensorFlow environment using a compatible
   Python/TensorFlow version.
2. Record layers, input/output names, shapes, dtypes, activations, parameter
   counts, file sizes, and unsupported/custom operations.
3. Reproduce the repository examples before conversion.
4. Reimplement only the documented input preparation in an experiment:
   contour orientation, dimensionless scaling, arclength resampling, zero
   padding, flattening, and optional volume feature.
5. Export with `tf2onnx` at a documented opset. Validate with `onnx.checker`.
6. Compare Keras and ONNX Runtime on repository examples, generated
   Young-Laplace contours, and Menipy pendant contours.
7. Map predicted dimensionless parameters back to Menipy's physical and fit
   models. Confirm parameter ordering from code rather than README text.
8. Test the network as both a final predictor and an initializer for Menipy's
   strict Young-Laplace optimizer.
9. Measure accuracy, convergence improvement, runtime, out-of-domain behavior,
   and sensitivity to contour noise, missing apex points, truncation, tilt,
   scale error, and incorrect needle separation.

Conversion acceptance criteria:

- all graph operations supported by CPU ONNX Runtime;
- maximum Keras-versus-ONNX absolute difference documented and no larger than
  `1e-5`, unless a justified precision envelope is established;
- deterministic predictions across repeated CPU runs;
- no TensorFlow import in the proposed Menipy runtime;
- physical-unit conversion independently verified;
- clear out-of-domain and quality gates.

### 2.2 Conan-ML

Tasks:

1. Trace the executable path from image input through edge extraction and ML
   prediction; identify the actual serialized estimator and framework.
2. Determine whether the ML input is raw pixels, filtered edge coordinates,
   geometric descriptors, or another representation.
3. Record training range: contact angles, Bond numbers, resolution, roughness,
   reflection, drop scale, and augmentation assumptions.
4. Reproduce official examples and reported high-angle behavior.
5. Select the converter from the actual artifact type: `tf2onnx`, `skl2onnx`,
   or a small independently specified NumPy/ONNX graph.
6. Compare original and ONNX predictions with the same parity criteria used
   for PendantDropMachineLearning.
7. Test contact angles below and above 110 degrees, Bond numbers near and above
   2, tilted baselines, reflections, low contrast, asymmetric drops, cropped
   contact points, and contaminated contours.
8. Compare Conan-ML against Menipy's circle, ellipse, polynomial/local tangent,
   and Young-Laplace contact-angle methods.
9. Determine whether Conan-ML is useful as a primary estimator, a high-angle
   specialist, an ensemble vote, or a QA discrepancy detector.

Conversion acceptance criteria:

- reproducible reference inference;
- ONNX parity within a justified tolerance;
- explicit valid-domain checks in any proposed integration;
- no silent prediction for unsupported Bond number or contact-angle ranges;
- license status resolved before any model is copied into `models/`.

## Phase 3: algorithm trace of the open-source analyzers

Analyze all three projects using the same trace template. Every statement in
the final report must cite a pinned source file and symbol or a paper equation.

### 3.1 Common trace template

For each project, document:

1. acquisition and image decoding;
2. grayscale/color conversion and intensity normalization;
3. denoising, thresholding, gradients, morphology, or background correction;
4. ROI selection and cropping;
5. substrate/baseline or capillary/needle detection;
6. droplet/capsule edge detection;
7. connected-component or contour candidate selection;
8. contour ordering, splitting, smoothing, interpolation, and outlier removal;
9. apex and symmetry-axis estimation;
10. contact-point and contact-angle calculation where applicable;
11. pixel-to-length calibration and needle diameter usage;
12. Young-Laplace/Bashforth-Adams or capsule-shape equations;
13. parameterization, boundary conditions, integration method, optimizer,
    bounds, initialization, and convergence criteria;
14. volume, area, curvature, surface tension, pressure, and elasticity outputs;
15. residuals, uncertainty, quality scores, and failure handling;
16. batch/video behavior and temporal assumptions;
17. computational complexity and observed runtime;
18. tests, reference data, and documented limitations.

### 3.2 OpenDrop-specific questions

- How are pendant and sessile analysis paths separated?
- How are needle edges, needle width, and drop/capillary junction found?
- Is a substrate explicitly detected or supplied through user-selected points?
- Which edge detector and contour-fitting representation are used?
- How are Young-Laplace parameters initialized and optimized?
- Which variables are fitted versus supplied by the user?
- How are contact angles, surface tension, volume, area, residuals, and
  uncertainty reported?
- Which algorithms are library-quality and separable from GUI/event code?

### 3.3 Drop-O-Matic-specific questions

- How does it detect the baseline in sessile-drop and captive-bubble modes?
- How are left and right contact points tracked over frames?
- Which local fit—line, polynomial, circle, ellipse, spline, or tangent—is used
  for dynamic contact angles?
- How does it handle advancing/receding angles, hysteresis, frame smoothing,
  evaporation, vibration, and transient segmentation failures?
- Does it detect a needle, or is a needle outside its intended workflow?
- Which outputs are direct measurements and which are derived statistics?

### 3.4 OpenCapsule-specific questions

- How are the pendant capsule contour, capillary attachment, apex, and symmetry
  axis obtained?
- Is needle/capillary diameter used for calibration or only as a boundary
  condition?
- Confirm that substrate detection is not applicable unless the code provides
  an analogous boundary detector.
- Which reference-shape, constitutive-law, elastic modulus, Poisson ratio,
  bending, pressure, and surface-tension parameters are fitted?
- How are wrinkles, compression, discontinuities, and non-axisymmetric shapes
  handled or rejected?
- Which numerical methods could strengthen Menipy's oscillating or future
  interfacial-rheology pipelines?

## Phase 4: common benchmark

### 4.1 Dataset

Create a versioned evaluation manifest containing:

- existing Menipy sessile and pendant samples;
- clean synthetic images with known Young-Laplace parameters;
- synthetic perturbations: blur, noise, uneven lighting, reflections, tilt,
  cropping, low contrast, compression artifacts, and partial occlusion;
- representative contact-angle bins: `<30`, `30-60`, `60-90`, `90-110`,
  `110-150`, and `150-180` degrees;
- pendant shapes spanning useful Bond/Worthington-number ranges;
- captive-bubble and capsule examples only where ground truth is defensible;
- explicit train/test separation for any model whose training data are reused.

Do not use output from one candidate as ground truth for another. Ground truth
must come from synthetic parameters, calibrated measurements, or independently
reviewed annotations.

### 4.2 Detection metrics

| Feature | Metrics |
| --- | --- |
| Droplet/capsule mask | IoU, Dice, boundary F-score, Hausdorff distance |
| Ordered contour | symmetric Chamfer distance, normal error, missing fraction |
| Needle/capillary | box IoU, centerline error, width/diameter relative error |
| Substrate/baseline | angle error, vertical RMSE, endpoint/contact-point error |
| Apex/symmetry axis | Euclidean apex error, axis angle and offset error |
| Contact points | left/right pixel and calibrated-distance error |

### 4.3 Scientific-output metrics

- left and right contact-angle MAE and bias;
- surface/interfacial-tension absolute and relative error;
- volume and surface-area relative error;
- fitted curvature, capillary pressure, Bond/Worthington number error;
- Young-Laplace contour residual and parameter uncertainty;
- capsule elastic-parameter error where ground truth exists;
- convergence rate, iteration count, failure rate, and false-success rate;
- CPU initialization, cold inference, warm inference, and peak memory;
- sensitivity curves for each perturbation rather than a single average.

### 4.4 Scientific safety gates

A method cannot be recommended as a primary Menipy path unless it:

- exposes its validity domain;
- rejects or flags invalid/non-axisymmetric inputs;
- preserves units and calibration provenance;
- produces residuals or an independent QA check;
- does not silently substitute an ML prediction for physical validation;
- has focused tests against synthetic and real reference cases.

## Phase 5: Menipy gap analysis

Map each external operation to the current Menipy owner:

- `common/auto_calibrator.py` and detection plugins;
- `common/sessile_detection.py`;
- `pipelines/sessile/geometry.py` and `stages.py`;
- `pipelines/pendant/`;
- `math/young_laplace.py`;
- `models/context.py`, result models, and contracts;
- MobileSAM ONNX runtime and detector registries;
- GUI overlays, diagnostics, and results history.

For every candidate improvement, record:

| Field | Required decision |
| --- | --- |
| Problem | Specific current Menipy failure or limitation |
| External technique | Algorithm, model, and exact source reference |
| Evidence | Benchmark result and uncertainty |
| Proposed role | replacement, fallback, initializer, ensemble, QA, or rejection |
| Integration boundary | module, plugin registry, context fields, and result contract |
| License | A/B/C/D category and obligations |
| Runtime impact | dependencies, model size, CPU time, memory |
| Scientific risk | domain limits and false-success modes |
| Test plan | unit, synthetic, integration, GUI, and regression coverage |
| Recommendation | adopt, prototype, defer, or reject |

Likely hypotheses to test, not assume:

- PendantDropMachineLearning may improve Young-Laplace initialization and reduce
  optimizer iterations without replacing final physical fitting.
- Conan-ML may improve high-angle sessile estimates or serve as an independent
  discrepancy detector.
- OpenDrop may offer stronger contour parameterization, initialization,
  uncertainty, or fitting diagnostics.
- Drop-O-Matic may offer better temporal contact-point tracking and dynamic
  advancing/receding-angle handling.
- OpenCapsule may provide numerical patterns for future elastic-interface or
  interfacial-rheology analysis rather than ordinary pendant tensiometry.

## Phase 6: final report structure

The final `adsa_open_source_assessment.md` should use this order:

1. Executive decision summary.
2. Scope, pinned versions, and reproducibility statement.
3. License and redistribution decision matrix.
4. PendantDropMachineLearning conversion and parity results.
5. Conan-ML conversion and parity results.
6. OpenDrop end-to-end algorithm trace.
7. Drop-O-Matic end-to-end algorithm trace.
8. OpenCapsule end-to-end algorithm trace.
9. Cross-project algorithm and output matrix.
10. Benchmark methodology and data quality.
11. Quantitative results with confidence intervals and failure examples.
12. Menipy gap analysis.
13. Prioritized recommendations.
14. Proposed architecture and plugin boundaries.
15. Implementation roadmap with effort/risk estimates.
16. Rejected ideas and reasons.
17. Limitations, unresolved questions, and required author/legal clarification.
18. Reproduction commands, hashes, and source references.

## Recommendation ranking

Score proposed improvements from 0 to 5 on:

- scientific accuracy;
- robustness;
- explainability and diagnostics;
- compatibility with Menipy's physical contracts;
- implementation effort;
- maintenance burden;
- CPU performance and memory;
- distributable size;
- license compatibility;
- quality of upstream tests and documentation.

Report both the raw dimensions and a weighted score. License incompatibility or
failure to expose validity limits is a blocking gate, regardless of total score.

## Completion criteria

The research is complete only when:

- every repository is pinned and source claims are traceable to symbols;
- both ML candidates have a documented conversion result or a reproducible
  technical reason conversion is impossible;
- license status is separated for code, model weights, and datasets;
- all analyzers are mapped through detection, geometry, physics, outputs, and
  failure handling;
- common benchmarks have been run on the same manifest;
- temporary clones, checkpoints, conversion environments, and caches are
  removed;
- recommendations identify concrete Menipy files and tests;
- the final report distinguishes measured evidence, source-backed facts, and
  engineering inference.

