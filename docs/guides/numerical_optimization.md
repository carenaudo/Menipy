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

### Remaining candidates

1. **First-ever pendant approximation lookup initialization:** prior first-use profiling
   attributed about 8 s to `_selected_plane_lookup_all`. It is already reused
   within a process and now across compatible launches. Reducing the initial
   generation work further needs independent equivalence tests; reusing prefixes
   from longer integrations can change adaptive samples and interpolation.
2. **ODE callback overhead:** repeated scalar trigonometry and invariant
   calculations remain in the hot callbacks. Benchmark isolated changes against
   full profiles and event locations before adopting them; scalar-library
   substitutions can alter floating-point trajectories.
3. **Temporal result retention:** bootstrap sample temporaries are now bounded
   by a batch or single row, but full result arrays still scale with recording
   length. Further reductions require preserving the full-result retention contract.

No relaxed tolerance, subsampling, optimizer replacement, parallel solver switch
or cross-run physical-result cache is introduced in this change.
