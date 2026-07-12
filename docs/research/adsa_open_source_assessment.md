# Open-Source ADSA Assessment

Date: 2026-07-10  
Scope: PendantDropMachineLearning, Conan-ML, OpenDrop, Drop-O-Matic, and OpenCapsule  
Menipy baseline: working tree evaluated locally on CPU with Python 3.13; conversion environments used TensorFlow 2.21, tf2onnx 1.17, ONNX 1.22, and ONNX Runtime 1.27.

## 1. Executive decisions

| Candidate | Decision | Intended role | Blocking issue |
| --- | --- | --- | --- |
| PendantDropMachineLearning | Prototype, but do not ship its weights | Optional initializer for strict pendant Young-Laplace fitting | No separate model license; bundled assets must be treated as GPL-3.0 until clarified; no validity-domain gate |
| Conan-ML | Prototype for research only | High-angle QA/discrepancy signal, not a primary estimator | GPL-2.0 model status; preprocessing fragility; strict absolute ONNX parity gate missed |
| OpenDrop | Prototype selected ideas by clean-room implementation | Pendant initialization, fit residuals, needle QA, sessile line/circle selection | GPL-3.0 source cannot be copied into the MIT-distributed project |
| Drop-O-Matic | Reject as an algorithm source | Simple comparison baseline only | Manual ROI/baseline, no temporal tracking despite video mode, weak validity and diagnostics |
| OpenCapsule | Defer | Future elastic-interface plugin | GPL-3.0-or-later, native legacy stack, and a materially different physical problem |

The strongest near-term improvement is not another segmentation model. Menipy first needs explicit validity-domain rejection, residual/uncertainty fields, and a detector benchmark that converts apparent successes into measured geometric errors. OpenDrop provides the clearest design reference for these changes. MobileSAM and YOLO segmentation remain optional contour proposals; they cannot replace calibration, ordered-contour construction, or physical validation.

This is an engineering and research assessment, not legal advice. Redistribution decisions require confirmation from the rights holders or legal counsel.

## 2. Pinned versions and reproducibility

All repositories were cloned only under the ignored `.tmp/adsa-research/repos/` workspace. No repository was added as a submodule and no third-party source was vendored.

| Project | Official repository | Commit evaluated | License observed |
| --- | --- | --- | --- |
| PendantDropMachineLearning | [FelixKratz/PendantDropMachineLearning](https://github.com/FelixKratz/PendantDropMachineLearning) | `9616d05be28315be53f07e4b6b2e2e07528bd10c` | GPL-3.0-only |
| Conan-ML | [jdber1/conan-ml](https://github.com/jdber1/conan-ml) | `db056cbd2958087070d0623b0e46e183d70baa3c` | GPL-2.0-only |
| OpenDrop | [jdber1/opendrop](https://github.com/jdber1/opendrop) | `61b227b70a92cf2f6e97d9ae20ba7a14fdd41bfd` | GPL-3.0-only |
| Drop-O-Matic | [KrzysztofDorywalski/Drop-O-Matic](https://github.com/KrzysztofDorywalski/Drop-O-Matic) | `093ce43024db613174ce4b2b0fd12ba6e683b86f` | MIT |
| OpenCapsule | [jhegemann/opencapsule](https://github.com/jhegemann/opencapsule) | `7af21a2f120ae29c1c40746f6e18d344190d804e` | GPL-3.0-or-later in source headers |

The durable license classification is in [`adsa_license_matrix.csv`](adsa_license_matrix.csv). The algorithm trace is in [`adsa_algorithm_matrix.csv`](adsa_algorithm_matrix.csv). Converted graphs and original assets were deliberately not retained.

Reproduction outline:

```powershell
git check-ignore -q .tmp/adsa-research/probe.txt
# Clone each official repository and check out the exact commit above.
# Create one conversion venv per ML framework under .tmp/adsa-research/venvs/.
# Keep checkpoints and converted graphs under .tmp/adsa-research/ only.
```

Exact model SHA-256 values are reported below as experiment identifiers, not as downloadable artifacts.

## 3. License audit

The audit separated source, weights, datasets, documentation, and dependency licenses.

- PendantDropMachineLearning's code is GPL-3.0. Its three `.h5` files and example dataset have no separately stated license. Conversion does not erase the learned parameters, so the safe classification is unresolved (`D`) and do-not-ship.
- Conan-ML's code is GPL-2.0-only. Its SavedModel and image collections also lack a separate asset grant. They are unresolved (`D`) for redistribution.
- OpenDrop is GPL-3.0. Menipy may study published equations and behavior, but implementation must be independent and must not copy its source.
- Drop-O-Matic is MIT and therefore eligible (`A`) with attribution, but its example media do not have clearly separate terms. Technical quality, not license, blocks adoption.
- OpenCapsule is GPL-3.0-or-later and links GSL. Boost and OpenCV are more permissive, but they do not alter the project's own GPL obligations.

An optional category-B downloader was not proposed because neither ML repository provides a sufficiently explicit model license or stable release manifest. A downloader would move acquisition outside the wheel, but it would not resolve the user's redistribution and use rights.

## 4. PendantDropMachineLearning conversion

### Source trace

`dataHandler.py:DataHandler.zeroPadData` pads coordinate sequences; the README specifies arclength spacing of `1e-2`, 226 coordinate pairs for the rho models, flattening to 452 features, and 512 pairs plus volume for the Worthington-number model. The upstream description is pinned in [README.md at the evaluated commit](https://github.com/FelixKratz/PendantDropMachineLearning/blob/9616d05be28315be53f07e4b6b2e2e07528bd10c/README.md).

The supplied models accept tensors shaped `(batch, 1, features)`, not ordinary two-dimensional feature matrices. The rho models contain 1,023,794 parameters; the Worthington model contains 3,678,514.

### Measured conversion results

| Model | Input features | ONNX size | Max absolute Keras/ORT difference | Mean absolute difference | ORT batch time/sample | Deterministic |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `model_uniform_rho.h5` | 452 | 4,104,882 B | `2.1011e-6` | `2.6796e-7` | 0.0284 ms | Yes |
| `model_uniform_rho_noise.h5` | 452 | 4,104,899 B | `9.5367e-7` | `1.7558e-7` | 0.0259 ms | Yes |
| `model_uniform_Wo.h5` | 1,025 | 14,723,745 B | `2.3842e-6` | `3.3623e-7` | 0.0398 ms | Yes |

All three graphs passed `onnx.checker` and deterministic repeated CPU inference. Their temporary SHA-256 values were:

- rho: `d94784762ef61a8dfe6679d65b2b194eb689c100bfd9d61a0ab0a989be6e8701`
- rho-noise: `fe38cd9aa9867dbc325498d0939f98004e61208ee02536ad5bd89061922233dd`
- Worthington: `c32cdd45664b1b0a8a489d0afe9af6dad0364eb1245b3e92ff09696c40c50d4e`

On the 140 bundled example contours, mean absolute reference-model error against bundled labels ranged from `3.43e-4` to `7.09e-4`. This is a reproduction check, not independent accuracy evidence: the provenance and training overlap of those examples are not documented.

### Decision

The technical conversion gate passed, but the distribution and scientific gates did not. The model has no explicit input rejection for wrong orientation, arclength, truncation, scale, tilt, padding, or volume convention. It should only be revisited after a rights clarification and a held-out synthetic/experimental test. If that happens, use it only to propose initial parameters; Menipy's strict optimizer must remain authoritative and must reject a proposal when residuals or physical bounds worsen.

## 5. Conan-ML conversion

### Source trace

The SavedModel contains three same-padded Conv1D layers (32, 16, and 8 filters), flattening, a 128-unit dense layer, and a scalar output: 1,254,777 parameters with input `(batch, 1223, 2)`. `prepare4model_v03` normalizes and pads half-drop contours; image preprocessing combines Canny/Hough cropping, Otsu contour extraction, OPTICS clustering, path ordering, and tilt correction. The relevant pinned implementation is [prepare_experimental.py](https://github.com/jdber1/conan-ml/blob/db056cbd2958087070d0623b0e46e183d70baa3c/modules/ML_model/prepare_experimental.py).

TensorFlow 2.21 could not object-load the TensorFlow 2.8 SavedModel because the serialized optimizer expected the removed `add_slot` behavior. The serving graph did load through the TensorFlow v1 compatibility loader. Reconstructing the documented inference architecture and assigning the five checkpointed layer pairs reproduced the legacy serving output exactly (`0.0` maximum difference).

### Measured conversion results

The reconstructed inference graph converted to a 5,023,550-byte ONNX graph and passed `onnx.checker`. Its temporary SHA-256 was `bb67f6e19af5346fd1bcea6becba286020b8b5a25a947026dab02c25cdef7ef2`.

- Repeated ORT output was deterministic.
- On 32 intentionally out-of-domain random tensors, maximum absolute difference was `0.046875`, but maximum relative difference was `3.61e-7`. Those huge raw outputs demonstrate why unvalidated tensors are unsafe.
- On 200 normalized smooth synthetic half-contours, maximum absolute difference was `1.8311e-4`, mean difference was `4.6844e-5`, and ORT batch time was 0.1652 ms/sample.
- The plan's absolute `1e-5` parity requirement therefore failed, even though relative numerical agreement was strong.

The model clips final predictions to `[0, 180]` in surrounding Python, not inside the SavedModel. It does not report residuals, uncertainty, side consistency, Bond-number validity, or contamination. The upstream README and training scripts emphasize high contact angles, but that does not establish safe operation outside a verified manifest.

### Decision

Do not use Conan-ML as Menipy's primary contact-angle estimator. If licensing is clarified, a later research plugin may compare its left/right predictions against geometric estimates and emit a discrepancy flag. The ONNX adapter must own normalization, exactly 1,223 points per half contour, range checks, symmetry checks, and rejection rather than silent clipping.

## 6. OpenDrop algorithm trace

OpenDrop has the most reusable design ideas, despite its incompatible source license.

### Pendant path

`extract_pendant_features` converts to grayscale, blurs, calculates Scharr gradients, normalizes squared gradient magnitude, thresholds adaptively, selects connected components, and uses Canny for non-maximum thinning. Needle edges are reduced to left/right extrema per row and fitted; points with residual at least one pixel are excluded and severe side imbalance is rejected. See [pendant.py](https://github.com/jdber1/opendrop/blob/61b227b70a92cf2f6e97d9ae20ba7a14fdd41bfd/opendrop/features/pendant.py).

`find_pendant_apex` uses a robust circle fit around the bowl, median residual selection, and an inertia-derived symmetry direction. The Young-Laplace fitter combines a geometric initial guess with an analytic-Jacobian Levenberg-Marquardt least-squares solve, tight tolerances, and a 50-evaluation cap. It returns the objective, point residuals, closest profile points, arclength locations, volume, and surface area; see [younglaplace/__init__.py](https://github.com/jdber1/opendrop/blob/61b227b70a92cf2f6e97d9ae20ba7a14fdd41bfd/opendrop/fit/younglaplace/__init__.py) and [model.py](https://github.com/jdber1/opendrop/blob/61b227b70a92cf2f6e97d9ae20ba7a14fdd41bfd/opendrop/fit/younglaplace/model.py).

Physical outputs preserve the calibration chain: pixel scale, density difference, gravity, apex radius, and Bond number determine interfacial tension. Needle diameter can provide scale. These provenance fields and residual vectors are the main Menipy opportunity.

### Sessile path

The baseline is supplied; OpenDrop does not provide a general automatic substrate detector here. After the same gradient/threshold family, points near or below the baseline are excluded. Near-contact branches are fitted independently. A line is accepted only when all local residuals are below one pixel; otherwise a circle is used. Baseline intersections and tangents yield left/right angles, curvature, and residuals. See [features/conan.py](https://github.com/jdber1/opendrop/blob/61b227b70a92cf2f6e97d9ae20ba7a14fdd41bfd/opendrop/features/conan.py) and [fit/conan.py](https://github.com/jdber1/opendrop/blob/61b227b70a92cf2f6e97d9ae20ba7a14fdd41bfd/opendrop/fit/conan.py).

## 7. Drop-O-Matic algorithm trace

Drop-O-Matic is a single interactive Python application. The user draws an ROI and baseline. `auto_detect_contour` applies grayscale conversion, Gaussian blur, inverse Otsu thresholding, chooses the largest external contour, samples every tenth point, and removes points within three pixels of or below the baseline. Circle fitting is linear least squares; ellipse fitting delegates to OpenCV. The implementation is pinned in [drop_o_matic.py](https://github.com/KrzysztofDorywalski/Drop-O-Matic/blob/093ce43024db613174ce4b2b0fd12ba6e683b86f/drop_o_matic.py).

The drawn baseline may be tilted visually, but the intersection calculation uses a single average `y`, discarding that tilt. Video mode processes frames independently and reuses manual state; it has no object tracking, temporal smoother, advancing/receding state machine, or hysteresis model. It also has no needle calibration, surface tension, pressure, area, volume, fit residual thresholds, uncertainty, or scientific validity gate.

The MIT license permits reuse with attribution, but there is no technical reason to replace Menipy code with it. Its most valuable contribution is a negative regression case: a tilted user baseline must not silently become horizontal.

## 8. OpenCapsule algorithm trace

OpenCapsule analyzes pendant elastic capsules, not sessile contact angles. Image processing uses a custom multi-scale combined-hysteresis edge detector, connected-edge validation, capillary removal, inner capillary-diameter measurement, left/right exterior extraction, symmetry handling, convex ordering, and sparsification. See [image.cpp](https://github.com/jhegemann/opencapsule/blob/7af21a2f120ae29c1c40746f6e18d344190d804e/source/image.cpp).

The reference shape solves the axisymmetric Laplace equation with Runge-Kutta integration. Deformed shapes support linear Hooke, nonlinear Hooke, and Mooney-Rivlin constitutive laws, with explicit or implicit integration, single/parallel shooting, finite-difference sensitivities, and Nelder-Mead alternatives. Fit failure can discard problematic parameters. Wrinkling is identified from compressive hoop-stress regions and paired with image-derived wavelength. See [capsule.cpp](https://github.com/jhegemann/opencapsule/blob/7af21a2f120ae29c1c40746f6e18d344190d804e/source/capsule.cpp).

Outputs include surface tension, pressure, area, volume, density, elastic moduli, Poisson ratio, area compression, wrinkle length/tension/wavelength, Foppl-von Karman number, and inferred thickness. These capabilities require a new result contract and validation program. They should not be folded into the existing rigid-drop pipeline.

## 9. Common benchmark

Sanitized outputs are written to `build/research/adsa_benchmark_results.json` and `build/research/adsa_benchmark_summary.csv`. The manifest seed is `20260710`.

The executable portion contains 36 locally generated sessile images across clean, Gaussian blur, Gaussian noise, and JPEG perturbations. Geometry was fixed so substrate and needle errors could be measured. Menipy found the baseline with a median absolute error of 1 pixel and ran in a median 12.78 ms, but returned no drop contour in all 36 full-calibration cases; the measured success rate was therefore `0.0`. This is a real regression signal, not a favorable benchmark result. Needle-width median error was 36 pixels, indicating that the top-connected component heuristic selected only a narrow region under these images.

The ML conversion rows measure graph equivalence only. Pendant's upstream examples are not held-out. Conan's repository image/training overlap is unresolved. OpenDrop and OpenCapsule were not executed because doing so would require building legacy GPL/native stacks; Drop-O-Matic lacks a stable headless adapter and requires manual geometry. Reporting fabricated cross-method accuracy would be less useful than recording these coverage gaps.

Consequently, the broad Phase-4 metric set remains a production-research prerequisite, not a completed scientific comparison. Mask Dice, boundary F-score, Hausdorff distance, full contact-angle bins, Bond/Worthington stratification, and independent experimental ground truth require a licensed benchmark dataset.

## 10. Menipy gap analysis

| Improvement | Current limitation | Evidence | Role | Score / 50 | Decision |
| --- | --- | --- | --- | ---: | --- |
| Fit residual and validity contract | Some methods return a plausible scalar without comparable residual/rejection metadata | OpenDrop exposes objective and residual vectors | QA/rejection | 43 | Adopt design, implement independently |
| Residual-driven sessile line/circle selection | Method selection is user/config driven more than evidence driven | OpenDrop uses a one-pixel local residual gate | Ensemble/selection | 38 | Prototype |
| Needle side-balance and fit residuals | Calibration confidence is heuristic and not physically traceable | OpenDrop fits both needle sides and rejects imbalance | QA/rejection | 40 | Prototype |
| Pendant ML initializer | Strict solve can benefit from diverse initial guesses | PDM ONNX parity passed | Initializer | 27 | Prototype only after rights and validity work |
| Conan high-angle discrepancy signal | High-angle geometric methods can disagree | Conan provides an independent learned scalar | QA | 21 | Research only |
| Temporal contact-point tracking | Video results lack a formal tracking/advancing-receding state model | Drop-O-Matic does not actually solve this | New subsystem | 31 | Design independently; do not adopt Drop-O-Matic |
| Elastic capsule analysis | Menipy has no constitutive elastic-interface solver | OpenCapsule demonstrates full scope | Separate plugin | 24 | Defer |

Scores combine accuracy potential, robustness, diagnostics, physical compatibility, effort, maintenance, performance, size, licensing, and upstream quality. Licensing conflicts and absent validity gates override totals.

The current MobileSAM ONNX integration is relevant only at the proposal boundary. A segmentation mask must be converted into a calibrated, ordered, topology-checked contour and must pass separate needle/substrate logic. The benchmark's zero drop-detection success also shows that adding another model without a common manifest would hide rather than solve detector regressions.

## 11. Prioritized recommendations

1. **Adopt a shared `FitDiagnostics` concept** in the existing result contracts: residual norm and distribution, convergence reason, iterations/evaluations, validity-domain checks, calibration provenance, side discrepancy, and explicit rejection reason. Preserve compatibility by making fields optional initially.
2. **Build a detector conformance harness** around existing registries. Each detector should consume the same manifest and return masks/contours plus confidence and rejection metadata. Start with the 36 synthetic cases that exposed the current sessile failure, then add tilted, cropped, reflected, low-contrast, and occluded cases.
3. **Prototype OpenDrop-inspired logic cleanly**: needle bilateral fit QA; robust apex/axis initialization; and residual-driven local line/circle selection. Do not translate or copy GPL source.
4. **Keep ML estimators behind an optional research plugin boundary**. The runtime may be ONNX-only, but model acquisition, license acceptance, hash pinning, preprocessing version, and validity domain must be explicit.
5. **Design temporal tracking independently** using contact-point association, confidence-aware smoothing, advancing/receding state, lost-track recovery, and per-frame calibration checks. Drop-O-Matic is not an implementation reference for this feature.
6. **Treat elastic capsules as a future analysis mode** with its own contract, solver, datasets, and dependency review.

## 12. Proposed architecture and roadmap

### Phase A — diagnostics and benchmark

- Extend result contracts with optional diagnostics and rejection data.
- Add a versioned manifest schema under tests or a future `benchmarks/` package.
- Add synthetic ground-truth generators and perturbation transforms.
- Fail tests on false success, silent clipping, unit loss, or baseline-angle loss.

### Phase B — clean-room geometric prototypes

- Implement bilateral needle fitting behind the existing needle detector registry.
- Add a robust pendant apex/axis initializer behind the solver boundary.
- Add an automatic sessile local-model selector without changing existing explicit method names.
- Compare every prototype with current algorithms before making it a default.

### Phase C — optional ONNX research providers

- Define a model manifest with source URL, license acknowledgement, SHA-256, input contract, preprocessing revision, supported domain, and output semantics.
- Keep TensorFlow, Keras, scikit-learn, and PyTorch out of runtime dependencies.
- Return proposals and QA flags, never silently authoritative physical values.
- Do not add PDM or Conan graphs until model redistribution/use rights are resolved.

### Phase D — video and capsules

- Implemented `sessile_dynamic` as a separately discoverable clean-room
  pipeline after stabilizing Phase-A single-frame diagnostics. It accepts
  videos and image sequences, fixes calibration, tracks substrate and contacts,
  preserves raw angles, records reacquisition segments, and derives
  advancing/receding/hysteresis only behind sequence validity gates.
- The versioned 48-case temporal manifest and deterministic benchmark cover
  state classification, transition timing, recoverable gaps and invalid
  sequences. MobileSAM remains non-authoritative and static `sessile` defaults
  are unchanged. The completed Phase-D run passed all 48 manifest gates with
  zero false-success/rejection failures; sanitized JSON/CSV evidence is written
  under `build/research/`.
- Elastic-capsule/OpenCapsule-like functionality remains deferred as a separate
  future pipeline with its own constitutive solver, datasets and contract.

## 13. Rejected approaches and unresolved questions

Rejected:

- Shipping converted PDM or Conan weights merely because ONNX conversion works.
- Treating a neural mask as a contact angle or surface-tension result.
- Using Drop-O-Matic as a temporal tracker.
- Porting GPL implementation details into Menipy.
- Claiming complete cross-project accuracy from upstream examples or overlapping datasets.

Unresolved:

- Are the PDM and Conan weights licensed separately by all rights holders?
- What exact training cases overlap Conan's `experimental data set` and `sensitivity data set`?
- Which independently licensed experimental dataset can cover contact-angle and Bond/Worthington domains with traceable calibration uncertainty?
- What residual and calibration thresholds should be normative for each Menipy pipeline?
- Why does the current full sessile AutoCalibrator fail to return a drop on the 36 fixed synthetic cases while finding the substrate? This should be diagnosed before detector replacement.

## 14. Limitations and evidence semantics

Statements labeled by numeric measurements came from the isolated local runs. Source-trace statements came from the pinned files and symbols linked above. Recommendations and role assignments are engineering inferences. No claim of legal clearance, clinical/safety suitability, or metrological certification is made.

The evaluation did not build OpenDrop or OpenCapsule, did not establish independent dataset rights, and did not run a blinded physical benchmark. These limitations prevent adopting any external estimator as a production default.

## 15. Phase-B clean-room prototypes

Phase B adds opt-in, clean-room geometry prototypes without importing external
implementations. `sessile_bilateral` and `pendant_bilateral` fit paired shaft
edges with robust linear models; `robust_axis` estimates a pendant symmetry
axis and rotates the strict-fit coordinate frame; `auto_residual` evaluates
tangent and circle candidates independently for each sessile side. Defaults
remain legacy. `experimental_geometry_mode=shadow` records comparisons under
diagnostics while preserving promoted measurements; explicit experimental
methods are authoritative and reject invalid geometry rather than silently
falling back.

The deterministic Phase-B manifest is
`tests/data/adsa_geometry_manifest.json` (seed `20260710`, 60 cases). Runtime
and memory evidence is produced by `scripts/benchmark_adsa_phase_b.py` and is
non-blocking until promotion criteria are approved.

## 16. Phase-C ONNX proposal boundary

Phase C introduces a separate `SEGMENTATION_PROVIDERS` registry, verified model
manifests, topology-checked proposals, and an `off`/`shadow` execution mode.
MobileSAM reuses its TinyViT embedding for droplet and needle box prompts and
never replaces calibrated pipeline state. The substrate remains a classical
line-detection problem. A batch annotation command emits review-required COCO
polygons, overlays, HTML, readiness evidence, and an approved-only YOLO
converter. No YOLO training or PDM/Conan graph is included.

The official MobileSAM repository is Apache-2.0 and distributes the checkpoint
inside its weights directory. No separate checkpoint-specific license was
located, so release packaging retains an explicit notice-review gate. Current
Python package-data does not include the ONNX files.

## 17. Cleanup record

The temporary repositories, framework virtual environments, downloads, converted ONNX graphs, generated inputs, raw predictions, and logs were removed after the sanitized matrices, report, and benchmark summaries were produced. The resolved deletion target was verified as `<repository>/.tmp/adsa-research`; post-cleanup checks confirmed the directory no longer existed and no external code, checkpoints, or unapproved graphs appeared in `git status`.
