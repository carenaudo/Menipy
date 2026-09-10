# Numerical optimization checks

## Audit follow-ups

The implemented audit work still has explicit limits: temporal output contours
and bootstrap allocations grow with sequence length; needle-hysteresis acquisition
still uses eager loading; initial large-history table creation is costly; exact
native 150% requested window sizes could not be tested on the available monitor.
Camera/hardware flows and a full repository-wide scientific release run have not
been certified by the focused checks. Optional archive browsing/restoration,
themes and broader preference unification remain separate product work.

These are not reasons to change numerical tolerances or scientific methodology.

## First optimization: fit-local pendant shape reuse

`fit_pendant_young_laplace_strict` optimizes radius, Bond parameter and X/Z
translation. Only radius and Bond parameter affect its ODE solution; observed
target height and integration options are fixed during each fit. Previously,
offset-only finite differences repeatedly integrated an identical shape.

An eight-entry LRU cache now lives inside each fit call. Keys are the exact
radius/Bond floats, without quantization. Translation creates a separate array;
the cached physical profile is not mutated. No cache survives into a subsequent
fit or stores a rejected/failed integration exception. The cancellation check
runs before every model request, including cache hits. The final right-branch
integration still obtains its own termination metadata.

Equations, solver bounds, optimizer options, ODE tolerances/events, refraction
handling, offset regularization and rejection criteria are unchanged.

`tests/test_pendant_fit_cache.py` compares entire output payloads, including
parameters, residual vectors, all model profiles, optimizer termination and
acceptance/warnings, with the cache disabled. Fixtures include a shifted clean
contour, noisy contour, refraction and severe contamination. The cancellation
test cancels before an offset-only request that would otherwise reuse a shape.
An early-termination test also verifies identical optimizer-failure rejection.

## Reproducible measurement

```powershell
$env:UV_CACHE_DIR='D:\uv_cache_dir'
$env:PYTHONUTF8='1'
uv run --offline --extra test python tools/profile_pendant_cache.py
uv run --offline --extra test python tools/check_gui_execution.py tests/test_pendant_fit_cache.py tests/test_young_laplace_ode.py tests/test_common_solver.py tests/test_pendant_pipeline.py tests/test_execution_lifecycle.py
```

The benchmark alternates cached/uncached order within one process, using four
pairs per deterministic input. It asserts exact full-output equality on every
run and records integration counts separately from wall time. Synthetic inputs
do not establish physical validity of real measurements. A separate full-pipeline
baseline is under `.cache/numerical-baseline`; its 1.43–2.66 s warm spread shows
why historical single-run comparisons should not be used to claim a speedup.

September 9, 2026 paired measurements on Windows 11 build 26200/Python 3.14.7:

| Fixture | Uncached median | Cached median | Time reduction | Integrations before → after |
| --- | --- | --- | --- | --- |
| Shifted, clean | 540 ms | 333 ms | 38.3% | 45 → 27 |
| 0.2 px noise, refraction 1.33 | 547 ms | 337 ms | 38.3% | 45 → 27 |
| 10 px noise | 664 ms | 403 ms | 39.4% | 54 → 32 |

All 24 paired outputs matched exactly, including rejection/acceptance and full
profiles. These three particular fixtures were accepted by the existing gates;
separate tests cover optimizer failure. Timing includes a lightweight integration
call spy on both paths. Raw evidence:
`.cache/pendant-cache/160a44179d1740bbb0c3c227052f25bf/report.json`.
The speedup applies to the strict fit, not overall application startup or
approximation-table generation.

Validation: 55 focused pendant/ODE/solver/execution regressions passed; the final
six cache tests passed after adding optimizer-failure coverage. Lint passed.
The full pendant reference pipeline matched baseline acceptance and every
promoted numeric metric exactly on all four post-change runs. Its three warm
runs were 0.851–0.903 s (median 0.864 s); use the paired table above for speedup
claims because baseline whole-pipeline timings were noisy. Evidence is under
`.cache/numerical-cached/pendant/result.json`.

## Remaining numerical candidates

### Completed follow-up: persistent selected-plane table cache

The deterministic selected-plane table now persists under
`~/.menipy/cache/selected_plane/`. Cache identity includes the table/ODE/cache
source hashes, loaded builder/helper/integrator bytecode, plane grid, Python,
NumPy/SciPy versions and platform. Changing methodology source invalidates the
cache automatically. Runtime replacement of the built-in integrator bypasses
the disk cache. Existing process-local caching remains available.

Files are disposable JSON, with a payload checksum, identity, bounded dimensions
and finite-value validation. Missing, corrupt or incompatible entries rebuild
using the original numerical builder. Write failures return the freshly computed
table. Writes are atomic and cancellation prevents partial publication. No
physical fit/result is persisted in this cache, and no numerical tolerances,
grids or interpolation rules change. Old cache versions may be deleted manually;
there is no automatic cleanup policy yet.

Fresh-process measurements on September 9, 2026: initial table generation/write
took **5.420 s**; two separate launches loaded it in **0.051 s** and **0.046 s**.
Every table byte matched the initial calculation, and the file timestamps
confirmed that neither restart rebuilt the table. This saves about 5.37 s of
table work on these later launches, not the first-ever calculation. Import,
calibration and other approximators still have separate costs.

Evidence: `.cache/lookup-profile/4026715f9e75490c9396c02718bed11e/`.
Reproduce with `uv run --offline --extra test python tools/profile_lookup_cache.py`.
Integrity/cancellation tests live in `tests/test_lookup_cache.py`; 31 cache and
pendant regressions passed before adding additional fault-injection checks.

### Completed follow-up: fixed observations and temporal neighbourhoods

Pointwise fitting now prepares the observed contour's arc-length grid and
resampling once per fit. Model resampling still runs for every evaluation with
the same operations/order and 400-point cap. Prepared data owns the resampled
observations. Direct `_residuals_pointwise` callers keep their original interface;
normal-projection fitting is unchanged.

Temporal velocity estimation previously scanned all valid frames for every
seven-index regression window. Unique increasing frame indices imply at most
three eligible records on either side of the current position. The new helper
checks only those candidates, retaining the original distance and segment
filters. Unsorted or duplicate indices use the original full search. Robust
regression, noise/deadband, state transition and minimum-run rules are unchanged.
Cancellation remains checked while preparing and processing windows.

Paired medians on the same runtime (September 9, 2026):

| Operation | Original | Optimized | Reduction |
| --- | --- | --- | --- |
| 500 residual evaluations, 5,000 observed / 300 model points | 113.7 ms | 20.6 ms | 82% |
| Full classification, 1,000 temporal frames | 686 ms | 534 ms | 22% |
| Full classification, 3,000 temporal frames | 3,481 ms | 1,578 ms | 55% |

These are residual-loop and classification timings, not end-to-end acquisition,
tracking or fitting speedups. Temporal results include all regression and state
work; the seven-window search alone changes from quadratic to linear work on
normal ordered sequences. Memory for full result series is still linear.
The residual microbenchmark excludes one-time preparation and uses a fixed model;
ODE integration remains the dominant cost in real pointwise Young–Laplace fits.

Exact regression checks cover residual arrays, complete weighted fit outputs
(excluding measured solve duration), temporal velocities/deadbands/states,
segments, gaps, duplicate indices and reverse ordering. 51 focused tests passed.
Evidence: `.cache/repeated-work/1e95343ba8884e61a055e35ebd24c30a/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_repeated_work.py`.

### Completed follow-up: bounded bootstrap sampling

Temporal confidence intervals now draw the same 2,000 bootstrap replicates in
batches of at most 262,144 sample elements (or one complete row for longer
inputs). Median partitioning reuses each disposable sample block. The seed,
draw order, median/MAD and percentile calculations are unchanged. Cancellation
is checked between batches; input arrays remain untouched.

Fresh-process measurements on September 9, 2026, with three repetitions per
case and timing under the same `tracemalloc` instrumentation:

| Frames | Original peak allocation | Batched peak allocation | Original median time | Batched median time |
| --- | --- | --- | --- | --- |
| 1,000 | 32.0 MB | 4.2 MB | 39.4 ms | 40.3 ms |
| 10,000 | 320.0 MB | 4.2 MB | 408.4 ms | 381.1 ms |

These are traced peak allocations for bootstrap work, not process RSS or total
analysis memory. The main gain is temporary memory (98.7% lower at 10,000 frames);
small-input timing is essentially unchanged. Input/MAD arrays, a minimum single
sample row and retained temporal results still scale with sequence length.

Exact draw-stream hashes and statistics match the original for odd/even lengths
and repeated values. Tests also check bounded sample allocations, input ownership
and cancellation between batches. The final 62 bootstrap, temporal, execution and
sequence/export regressions passed; targeted lint passed.
Evidence: `.cache/bootstrap-profile/d3ed611220eb4b228776beb82da1d00c/`.
Reproduce with `uv run --offline --extra test python tools/profile_bootstrap_batches.py`.

### Completed follow-up: pendant envelope grouping

Pendant envelope construction now stably sorts height-bin indices once instead
of scanning the entire contour twice for each bin. Each bin retains the original
point order and mean/min/max operations. The final equal-height merge also uses
contiguous groups instead of repeated full-array masks. Cancellation is checked
before sorting and for each group; native sorting itself remains cooperative
only at its boundaries. No bin widths, rejection gates or fit tolerances change.

Six alternating paired runs on Windows 11, Python 3.14.7 and NumPy 2.5.2
(September 9, 2026), using two-sided synthetic contours:

| Contour points / height bins | Original median | Grouped median | Reduction |
| --- | --- | --- | --- |
| 1,000 / 500 | 16.7 ms | 8.1 ms | 51% |
| 5,000 / 2,500 | 148.1 ms | 40.4 ms | 73% |
| 20,000 / 10,000 | 1,499.6 ms | 160.1 ms | 89% |

These measure envelope construction only. Grouping uses additional linear
temporary storage and replaces quadratic scans with sorting plus linear group
work. Exact comparisons cover random contours, tilted/degenerate axes, multiple
bin widths, empty/small inputs, duplicate points, clamped heights and complete
clean/noisy strict fits. 55 focused envelope, pendant, fit-cache and ODE tests
passed; targeted lint passed.

Evidence: `.cache/envelope-profile/42f138e07fe04907858beb8a9f69d6a0/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_pendant_envelope.py`.

### Completed follow-up: strict ODE sine reuse

The strict pendant ODE reuses one `np.sin(psi)` value for the radial correction
and height derivative in each callback. NumPy scalar types, division, apex
handling, solver settings and cancellation checkpoints are preserved. This
avoids one redundant sine evaluation away from the apex without substituting
another math library or changing arithmetic expressions.

Eight alternating paired runs on Windows 11 / Python 3.14.7 / NumPy 2.5.2:
50 integrations took 601.3 ms originally and 594.5 ms with reuse; a full noisy
strict fit took 332.4 ms and 327.6 ms respectively (median). The approximately
1.4% full-fit reduction is modest, and integration-only differences are close
to run-to-run variation. This is not an end-to-end pipeline speedup claim.

All 35 focused tests passed, including exact adaptive profile coordinates,
event metadata, zero/high Bond values, invalid radius, height/needle cutoffs,
branch selection, complete clean/noisy fits and existing cancellation checks.
Targeted lint passed. Compatible lookup tables are regenerated after this source
change under the existing cache invalidation policy.

Evidence: `.cache/ode-reuse/6641100b2ece4bd782e9db20c49f7950/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_ode_reuse.py`.

### Completed follow-up: exact temporal regression convergence

Local robust slope regression now exits when an update produces bit-for-bit
identical coefficients. The next iteration would use the same design, residuals
and weights and reproduce that solution. There is no convergence tolerance:
nonconverged windows retain all six updates. Cancellation remains checked during
iteration and is also checked before returning, including an early return.

Six alternating paired runs on the same Windows/Python/NumPy environment as the
ODE benchmark, using 3,000 frames with gaps and segment boundaries:

| Synthetic sequence | Original classification | Optimized | Reduction |
| --- | --- | --- | --- |
| Constant half-width | 1,666.7 ms | 449.3 ms | 73% |
| Linear trend plus sinusoidal variation | 1,621.6 ms | 576.9 ms | 64% |

These are full state-classification timings, excluding acquisition, tracking and
frame measurements. Savings depend on exact convergence; difficult windows can
still need all updates. Slopes, deadbands and complete frame results matched
exactly. Tests also cover random noise, outliers, duplicate timestamps and
cancellation after a converged solve. 67 focused regressions passed; lint passed.

Evidence: `.cache/slope-convergence/18097a66bdb34d74988cf7588cdcab26/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_slope_convergence.py`.

### Completed follow-up: straight-baseline crossing selection

Contact detection selects candidate crossing edges using NumPy before entering
the scalar intersection loop. Edge ordering, interpolation, projection, endpoint
handling and fallback candidate selection remain unchanged. Curved substrates
retain their existing path. This removes Python work on noncrossing edges while
using linear temporary arrays; contours with many crossings benefit less.

Six alternating paired batches of 100 calls on synthetic circular contours,
with a horizontal baseline crossing two edges, on the same Windows runtime:

| Contour points | Original median per call | Optimized | Reduction |
| --- | --- | --- | --- |
| 1,000 | 0.693 ms | 0.080 ms | 88% |
| 10,000 | 6.287 ms | 0.164 ms | 97% |

These measure contact detection, not complete analysis. Exact reference tests
cover arbitrary contours, empty/small inputs, horizontal/tilted/vertical and
degenerate baselines, fallback selection, duplicated vertices and nearly
parallel edges. The first regression run had 47 passes and one apex-refinement
failure (`test_refine_apex_curvature_on_circle`: y=39.86567 versus minimum 40).
That failure reproduced with the original contact routine, independently of this
optimization; no scientific threshold was changed. The final focused contact,
substrate and dynamic-sessile run passed all 54 tests. Targeted lint passed.

Evidence: `.cache/contact-crossings/4892f0e181e34c91b908dc16b22be2e5/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_contact_crossings.py`.

### Completed follow-up: shared ODE invariant calculations

The shared sessile and pendant integrators calculate radius-dependent curvature
and gravity coefficients once per integration and reuse sine within each
callback. The exact original scalar expressions and NumPy trigonometric
functions are retained. This leaves the ODE, adaptive solver tolerances, events
and callback cancellation unchanged.

Six alternating paired runs without concurrent tests, on the existing Windows
runtime, gave these medians:

| Operation | Original | Optimized |
| --- | --- | --- |
| Shared pendant, 50 integrations | 999.9 ms | 990.8 ms |
| Shared sessile, 50 integrations | 248.3 ms | 238.6 ms |
| Synthetic pendant fitting workflow | 832.0 ms | 814.6 ms |
| Synthetic sessile fitting workflow | 100.2 ms | 96.9 ms |

Fit workflows include synthetic observation generation and generic pointwise
fitting, excluding GUI, detection and export. Median reductions are modest
(about 2.1% and 3.3% for those workflows), not guaranteed end-to-end gains.
An earlier run overlapped tests and showed inconsistent sessile timing; the
isolated repeat above is the reported comparison. Exact adaptive coordinates
and fit payloads (excluding measured duration) match. 48 regressions passed,
including cancellation and advanced sessile tests; targeted lint passed.

Evidence: `.cache/ode-invariants/0028f430e8444f6998d93acfc993a79d/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_ode_invariants.py`.

### Completed follow-up: scalar adjacent-velocity medians

State-transition classification uses at most two adjacent slopes. A singleton's
median is itself; two slopes have their arithmetic mean as median. Scalar
arithmetic now avoids constructing and partitioning a NumPy array per frame.
The sum starts at positive zero to preserve NumPy's signed-zero behavior.
The reported seven-frame robust velocity and all state thresholds are unchanged.

Six alternating paired runs on 3,000 synthetic frames produced these full
classification medians:

| Sequence | Original | Scalar | Reduction |
| --- | --- | --- | --- |
| Constant half-width | 486.6 ms | 441.7 ms | 9.2% |
| Linear trend plus sinusoidal variation | 605.6 ms | 548.1 ms | 9.5% |

These timings exclude acquisition, tracking and frame measurement. All 48
focused tests passed, including exact states/deadbands, gaps, duplicate times,
reverse ordering, isolated segments, signed zeros and extreme scalar values.
Targeted lint and whitespace checks passed.

Evidence: `.cache/adjacent-velocity/3dcd08869a6044f18d95367a02ee9750/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_adjacent_velocity.py`.

### Completed follow-up: shared multi-plane profile preparation

Pendant multi-selected-plane estimation now filters, orders and deduplicates
the input profile once, then reuses those arrays for each plane. Single-plane
callers retain their existing wrapper. Per-plane interpolation, lookup scoring,
physics, aggregation and output fields are unchanged. Preparation is local to
one call, and cancellation is checked between planes.

Six alternating paired batches of 100 calls with real warmed lookup tables:

| Profile points | Original median per call | Prepared | Reduction |
| --- | --- | --- | --- |
| 300 | 0.610 ms | 0.347 ms | 43% |
| 10,000 | 3.726 ms | 0.955 ms | 74% |

These measure the multi-plane estimator only, excluding first-use table
generation, strict fitting and complete image analysis. Exact-output tests
cover real lookup queries, reversed/duplicate/invalid samples, insufficient
profiles and successful/unavailable planes. No numerical tolerances change.
The final 33 pendant, exact-fit and preparation regressions passed, following
15 focused preparation/cache checks; targeted lint passed.
The existing cache source fingerprint invalidates older table files after this
source edit; tables are regenerated once before reuse.

Evidence: `.cache/multi-plane/cba62edb854847c5930aa475ec2aa777/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_multi_plane_preparation.py`.

### Completed follow-up: combined bootstrap percentiles

Both confidence-interval endpoints are now computed in one percentile call,
reusing the disposable 2,000-element bootstrap-median array in place. Sampling,
the seed, replicate count and percentile interpolation method remain unchanged.

Six alternating paired benchmark batches gave 0.545 ms originally versus
0.442 ms combined for five observations (19% lower). At 1,000 observations,
both versions took 40.26 ms: sampling dominates, so no meaningful large-input
speedup is claimed. These are complete bootstrap-component timings, not full
analysis timings.

All 34 focused tests passed, including exact statistics for odd/even sizes,
repeated values and extreme magnitudes, unchanged random draws, input ownership,
bounded sampling and temporal regressions. Targeted lint passed.

Evidence: `.cache/bootstrap-percentiles/5e0014a6ac744b969d4c1c90e2723cb5/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_bootstrap_percentiles.py`.

### Completed follow-up: sessile clipping-mask reuse

Fresh profiling of the sessile reference workflow identified repeated scalar
half-plane tests in contour clipping. The straight-substrate path now computes
its mask once with the same cross-product expression and reuses it for segment
endpoints. Tolerance, scalar intersections, output order, deduplication and the
curved-substrate path remain unchanged.

Six alternating paired batches of ten calls on circular contours crossing a
straight baseline gave these per-call medians:

| Contour points | Original | Reused mask | Reduction |
| --- | --- | --- | --- |
| 1,000 | 13.03 ms | 0.90 ms | 93% |
| 10,000 | 119.01 ms | 6.20 ms | 95% |

These are clipping timings. The full reference pipeline's warmed medians were
about 1.44 s before and 1.41 s after, measured in separate runs; do not infer a
precise end-to-end speedup from that small difference. Acceptance, rejection
reasons and all promoted numeric results matched exactly. ODE integration is
still the dominant full-pipeline cost. All 63 focused clipping, substrate and
contact-angle tests passed; targeted lint passed.

Evidence: `.cache/clip-mask/7b9cd723e7fe404f9e3e20fff764c017/report.json`,
`.cache/current-hotspots/sessile/` and `.cache/current-hotspots-after/sessile/`.
Reproduce the paired component benchmark with
`uv run --offline --extra test python tools/profile_clip_mask.py`.

### Follow-up investigation: generic ODE cache would not help this workload

`tools/profile_solver_reuse.py` traces the integrator requests made by the
current sessile reference workflow. All three observed fits (first, warm and
profiled) made 18 integrations with 18 distinct bitwise parameter vectors.
There was zero repeated integration time to recover with a fit-local exact-key
cache. Instrumentation preserved the reference acceptance, rejection reasons
and promoted numerical values, and successive unprofiled runs matched.

No generic solver cache was added. The current cumulative profile attributes
about 1.74 of 1.87 instrumented seconds to ODE integration, mostly adaptive
stepping. Larger gains here require a separately validated integration approach
or scientific stopping criteria; changing tolerances, truncating profiles or
rounding cache keys is not a safe mechanical optimization. Profile overhead is
included in those cumulative figures, which are not normal wall-clock timings.

Evidence: `.cache/solver-reuse/af55cd948eea42d190e753cd81ed9d57/reuse.json`
and `.cache/current-hotspots/sessile/fitting.txt`. Run
`uv run --offline --extra test python tools/profile_solver_reuse.py` to repeat
the audit as methodologies change. The profiling tool passed lint.

### Completed follow-up: active-contour matrix reuse

Temporal tracking repeats the same internal snake matrix across frames. Its
inverse now uses an eight-entry process-local LRU cache for contours up to 300
nodes. Larger standalone snakes bypass the cache. Identity includes node count,
exact hexadecimal alpha/beta/gamma values, boundary condition, matrix-builder
function and inversion function. Cached arrays are read-only and used only for
matrix products; singular calculations are not cached.

The maximum retained inverse-array payload is about 5.5 MiB, excluding transient
matrix construction. There is no image, measured contour or scientific result
cache. New settings still incur their first inversion. Evolution, force fields,
boundary projections and convergence thresholds remain unchanged.

Six alternating paired complete evolutions of a synthetic disk, with 15 maximum
iterations and warmed matrices (September 10, 2026):

| Nodes | Original median | Cached median | Reduction |
| --- | --- | --- | --- |
| 80 | 3.67 ms | 3.08 ms | 16% |
| 200 | 26.81 ms | 3.59 ms | 87% |
| 300 | 45.68 ms | 4.17 ms | 91% |

Uncached timings varied substantially at larger matrix sizes, but every paired
cached run was faster. These are complete snake evolution timings, not full
video analysis or first-frame speedups. All result arrays, iteration counts,
energies and convergence flags matched exactly. All 60 focused tests passed,
covering all four boundary conditions, tracking, coefficient/builder changes,
large-input bypass, bounded retention and singular failures; lint passed.

Evidence: `.cache/snake-matrix/94b5712381194d97b0f7fd867b2cee51/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_snake_matrix.py`.

### Active-contour gradient sample reuse

Line attraction and normal-flux forces now share the same two image-gradient
samples within each force evaluation. This removes two bilinear sampling calls
per iteration when both weights are nonzero. Samples are recomputed for every
new contour; no image-dependent cache, force arithmetic, stopping criterion or
scientific tolerance changed.

Six alternating paired batches of 30 complete evolutions on a 120 × 120 synthetic
circle, 15 iterations, periodic boundary and warmed matrix cache (Windows 11,
Python 3.14.7, September 10, 2026):

| Settings | Nodes | Original median | Shared median | Reduction |
| --- | --- | --- | --- | --- |
| Line 0.2, flux 0.3 | 80 | 4.116 ms | 3.700 ms | 10% |
| Line 0.2, flux 0.3 | 300 | 5.442 ms | 4.790 ms | 12% |
| Default weights | 80 | 3.020 ms | 3.030 ms | No meaningful change |
| Default weights | 300 | 3.937 ms | 3.938 ms | No meaningful change |

These are full snake evolution timings, not whole application timings. All 109
focused force, matrix, active-contour and tracking tests passed. Exact comparisons
cover all force enable/disable combinations, out-of-image coordinates, supplied
and computed gradients, input preservation, and full evolution across all four
boundary conditions. Output arrays, energies, iteration counts and convergence
flags match the independent pre-change reference exactly.

Evidence: `.cache/snake-forces/bfc3e538b5864078bd9a82f6696e1398/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_snake_forces.py`.

### Defer unused iteration curvature

Active-contour evolution uses normals to compute external forces but previously
computed and discarded curvature on every iteration. A private direction helper
now supplies the iteration normals without computing tangent derivatives or
curvature. The public geometry function still returns full tangents, normals and
curvature, including for the final evolved contour. Finite differences, force
calculations, resampling and convergence criteria are unchanged.

Six alternating paired batches of 30 complete evolutions on a 120 × 120 synthetic
circle, 15 iterations, periodic boundary and warmed matrix cache (Windows 11,
Python 3.14.7, September 10, 2026):

| Settings | Nodes | Original median | Deferred median | Reduction |
| --- | --- | --- | --- | --- |
| Default weights | 80 | 3.150 ms | 2.642 ms | 16% |
| Default weights | 300 | 4.078 ms | 3.417 ms | 16% |
| Line 0.2, flux 0.3 | 80 | 3.738 ms | 3.239 ms | 13% |
| Line 0.2, flux 0.3 | 300 | 4.891 ms | 4.383 ms | 10% |

One combined-force timing pair had a slower optimized run; the table reports
medians, not guaranteed individual-run gains. These measurements cover complete
snake evolution, not application startup or total video processing.

All 150 focused geometry, force, matrix, active-contour and tracking tests passed;
lint passed. Exact reference comparisons cover empty/short/degenerate contours,
open and closed geometry, all four evolution boundary conditions, immediate and
iteration-limit termination, and enabled/disabled resampling. Final arrays,
curvature, energy, convergence and iteration counts match exactly. An instrumented
25-iteration run verifies full curvature is computed only once, for the result.

Evidence: `.cache/snake-geometry/473198ae323e43e09e9db1d59095492b/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_snake_geometry.py`.

### Skip normals for forces that do not use them

When flux and balloon weights are both zero, active-contour evolution now skips
iteration direction calculations entirely. Edge and line forces depend only on
image gradients. The final result still contains full tangents, normals and
curvature. Normal-dependent forces retain the existing direction calculation.
The condition is checked each iteration rather than cached across configuration
changes. `compute_external_forces` accepts `normals=None` only for forces that do
not require normals, and raises a specific error otherwise.

Six alternating paired batches of 30 complete evolutions on a 120 × 120 synthetic
circle, 15 iterations, periodic boundary and warmed matrix cache (Windows 11,
Python 3.14.7, September 10, 2026), compared with the preceding deferred-curvature
implementation:

| Default settings | Previous median | Optimized median | Reduction |
| --- | --- | --- | --- |
| 80 nodes | 2.643 ms | 1.940 ms | 27% |
| 300 nodes | 3.478 ms | 2.715 ms | 22% |

Flux-enabled timing controls showed approximately -0.3% and +2% median changes;
no speedup is claimed for configurations that still need normals. Measurements
cover full snake evolution, not total video processing or GUI latency.

All 173 focused tests passed, including exact full-evolution comparisons across
all four boundary conditions and flux/balloon combinations, direction call counts,
missing-normal validation, configuration changes and existing tracking coverage.
Final geometry, energy, iteration counts and convergence match exactly.

Evidence: `.cache/snake-normals/e1f70ff739014b598903cd3f52cee1ed/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_snake_normals.py`.

### Single-image sessile and pendant focus

Fresh full-pipeline profiles of the repository reference images show fitting is
the dominant cost: about 1.68 of 1.75 profiled seconds for sessile, and 1.04 of
1.10 profiled seconds for pendant. Adaptive ODE stepping dominates both. These
cProfile times include instrumentation overhead. Evidence is in
`.cache/single-image-focus/{sessile,pendant}/fitting.txt` and `result.json`.

The strict pendant callback now indexes the three NumPy state values directly,
avoiding an array iterator on each evaluation. Scalar types, arithmetic, events,
adaptive solver settings and cancellation checkpoints are unchanged. The same
candidate was tried in the shared sessile integrators but reverted because the
full-image improvement was only 0.3%, within timing variation.

The retained pendant change was measured with six alternating warmed pairs,
including image decoding and every pipeline stage through validation. Automatic
calibration is performed once outside timing; the scale is synthetic 100 px/mm,
so these samples validate timing and equivalence, not physical accuracy. Windows
11, Python 3.14.7, September 10, 2026:

| Image | Dimensions | Previous median | Final median | Interpretation |
| --- | --- | --- | --- | --- |
| Pendant water reference | 1280 × 1024 | 820 ms | 788 ms | 3.9% faster |
| Sessile needle reference | 1600 × 1200 | 1334 ms | 1330 ms | Unchanged code; timing control |

Persisted outputs match exactly after removing execution-duration metadata,
including metrics, rejection status/reasons and diagnostics. These are warmed
single-image pipeline timings, not GUI startup, calibration-dialog or first-ever
lookup-generation timings. Existing profiler warnings about an unregistered
auto-detect preprocessor and overlay `summary_error` remain outside this change;
this is not a full release certification.

The new regression reference freezes the pre-change ODE implementations and checks
adaptive profiles plus termination metadata. Existing fit, cancellation and
pendant pipeline tests provide wider regression coverage: all 101 focused tests
passed, as did lint and diff checks. Benchmark evidence:
`.cache/single-image-state/6a79096c09614339ae9395c7b9c7228d/report.json`.
Reproduce independently with
`uv run --offline --extra test python tools/profile_single_image_state.py`.

Further single-image work should target the measured integration cost. Changing
integration extent, tolerances or solver methods needs separate scientific
validation; it is not justified by these timings alone.

### Single-image sessile needle-row fast path

Automatic sessile calibration repeatedly selects the foreground run nearest the
image center. For rows containing one contiguous run, sorted foreground indices
now use the exact span/count identity to return the endpoints directly. This
avoids allocating gap and split arrays and invoking the run-selection callback.
Empty rows and fragmented rows retain their original behavior and tie-breaking.
No segmentation thresholds, contour samples or geometric formulas changed.

Six alternating paired batches of ten complete auto-calibrations on the reference
images (Windows 11, Python 3.14.7, September 10, 2026):

| Mode | Image dimensions | Original median | Optimized median |
| --- | --- | --- | --- |
| Sessile | 1600 × 1200 | 109.13 ms | 93.72 ms (14% reduction) |
| Pendant control | 1280 × 1024 | 10.87 ms | 10.90 ms (unchanged) |

These timings include complete `run_auto_calibration` calls with decoded images,
not disk decoding, profile fitting or GUI rendering. Every calibration field,
including measured/display contours, confidence and contact geometry, matches
exactly. All 25 targeted run-selection, calibration and liquid-boundary tests
passed. Exhaustive eight-pixel rows cover empty, contiguous and fragmented input,
centers outside the image and equal-distance ties.

Broader initial coverage yielded 31 passed, one skipped and one apex failure:
`TestRefineApexCurvature.test_synthetic_drop`. Restoring the original row selector
in memory produced the identical apex value (9.999259 rather than the test's
expected approximately 30). This unrelated methodology failure was not changed.

Evidence: `.cache/single-image-calibration/1b22f0864145452aaf02b5f7f53371d8/report.json`.
Reproduce with
`uv run --offline --extra test python tools/profile_single_image_calibration.py`.

### Reuse sessile shaft detection within one calibration

Sessile auto-calibration previously ran the same shaft detector twice on the same
image and substrate: once to locate the needle, and again to choose the fallback
silhouette's expansion cutoff. The first complete result now travels to fallback
segmentation. A failed detection is also reused. Independent contour detection
still computes its own shaft result when none is supplied, and each calibration
run clears and recomputes the result. This is not a cross-image cache. Bilateral
needle selection and its existing legacy expansion-cutoff behavior are unchanged.

Six alternating paired batches of ten complete calibrations (Windows 11,
Python 3.14.7, September 10, 2026), compared with the preceding row fast path:

| Mode | Image dimensions | Previous median | Reuse median |
| --- | --- | --- | --- |
| Sessile | 1600 × 1200 | 99.44 ms | 82.05 ms (17% reduction) |
| Pendant control | 1280 × 1024 | 10.90 ms | 10.76 ms (no claimed change) |

These are decoded-image auto-calibration timings, excluding disk loading,
fitting and rendering. All calibration fields match exactly, including masks,
measured/display contours, contact points and detector diagnostics. All 32 focused
tests passed, including failed-detection reuse, expansion masks, one shaft call
per run, repeat-run recomputation and offscreen calibration wizard coverage.
Lint and diff checks passed.

Evidence: `.cache/calibration-shaft-reuse/5d9e101175a54fca8a17b442354edb46/report.json`.
Reproduce with
`uv run --offline --extra test python tools/profile_calibration_shaft_reuse.py`.

### First-ever pendant lookup generation

Cold selected-plane table construction still performs all 576 integrations on
the same 24 × 24 beta/height grid. Those integrations repeat many exact derivative
states: an instrumented baseline observed 312,570 repeated evaluations out of
339,318 (92%). Table generation now shares a bounded derivative dictionary across
the height sweep for each beta, clearing it before the next beta and releasing
it when construction ends. Entries are keyed by exact beta hex and float64 state
bytes, with FIFO eviction at 16,384 entries. Only the built-in integrator receives
this private optimization; replacement integrators retain independent execution.

Every profile retains its own adaptive integration and terminal-event handling.
No profile prefix is reused, no states are rounded, and no tolerance, grid or
step limit changes. Cancellation is checked on every cache hit and miss. Ordinary
image fitting uses the original callback path without cache lookups.

Four alternating paired complete table builds, bypassing both disk and in-memory
table caches, on Windows 11 / Python 3.14.7:

| Operation | Original median | Optimized median | Reduction |
| --- | --- | --- | --- |
| Generate all selected-plane tables | 5.140 s | 4.788 s | 6.8% |

This reduces first-ever generation itself; it is separate from the previously
implemented fast loading of persisted tables. The remaining integration/solver
overhead still dominates. All arrays in all generated tables match exactly.
All 67 focused tests passed, covering full-grid equality, adaptive profiles and
event metadata, beta identity, eviction, cancellation on a hit, persistence and
existing fit regressions. Lint passed.

Evidence: `.cache/cold-lookup-build/b71647954415417991bd54a8eb10a0ec/report.json`.
Overlap audit: `.cache/lookup-rhs/c8b254d7a4e24de997be4aa87e10fc99/report.json`.
Reproduce with `uv run --offline --extra test python tools/profile_cold_lookup_build.py`.
Fresh-process validation measured 4.792 s for generation/publication and 51.6 ms
and 47.4 ms for subsequent disk loads, with identical table SHA-256 and no cache
rewrite. Evidence: `.cache/lookup-profile/031aee304d3945d5bb6620051cfb6f47/`.

### Remaining candidates

1. **Further pendant lookup generation reductions:** exact derivative reuse now
   reduces cold generation by 6.8%, in addition to persistent-table reuse on later
   launches. Further savings need separate validation; reusing prefixes from
   longer integrations can change adaptive samples and interpolation.
2. **ODE callback overhead:** strict sine reuse is complete; other scalar and invariant
   calculations remain in the hot callbacks. Benchmark isolated changes against
   full profiles and event locations before adopting them; scalar-library
   substitutions can alter floating-point trajectories.
3. **Temporal result retention:** bootstrap sample temporaries are now bounded
   by a batch or single row, but full result arrays still scale with recording
   length. Further reductions require preserving the full-result retention contract.

No relaxed tolerance, subsampling, optimizer replacement, parallel solver switch
or cross-run physical-result cache is introduced in this change.
