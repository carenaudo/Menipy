# A11–A13: sequence storage, profiling and exports

## A11 — bounded decoded-frame retention

`DynamicSessilePipeline` requests `disk_backed=True` from video/directory
acquisition. `models/frame_store.py` owns a temporary binary file containing
lossless NumPy records and lightweight offsets/timestamps. Reading a frame
returns an independent array. No memory mapping or decoded image cache grows
with clip duration. Existing loader callers keep the eager-list default.

The declared, serialization-excluded `Context.sequence_store` transfers ownership
between acquisition and metrics. `ctx.frames` retains only the first preview
frame for file-backed dynamic runs. Metrics closes the store in `finally`;
the base pipeline also closes it on failure, cancellation between stages,
and acquisition-only completion. Direct loader callers must close their store.
Decoder captures always release in `finally`. Disk-full errors fail execution;
there is no silent fallback to retaining a complete video in memory.

Timing fallback, natural ordering, source digest, first-five scale sampling,
tracker/reacquisition, rejection, and temporal export order are unchanged.
The current detector already processes frames incrementally, so its scientific
implementation was preserved. Full per-frame result/contour records and summary
statistics remain in memory for the existing timeline and export contract.
This bounds decoded pixels, **not total sequence memory**: metadata, output
contours and bootstrap temporary arrays still scale with frame count. Caller
supplied in-memory images remain caller-owned and are not moved to disk.
Other pipelines using the eager loader API (including needle hysteresis) retain
their current behavior; this integration targets the audited dynamic sessile flow.

Temporary storage costs approximately uncompressed image size plus a small
record header. It uses the OS temporary directory and is deleted on close.
This trades disk capacity and I/O for bounded decoded-pixel RAM.

### Measured acquisition evidence

Fresh Windows processes, Python 3.14.7, Windows 11 build 26200, AMD64 Family 23
Model 96; synthetic 640×480 BGR MJPG, 20 FPS. Inputs generated before measurement.
Native working-set counters include libraries and process-lifetime allocations.

| Frames | Store size | Acquisition | Working set after | Process peak |
| --- | --- | --- | --- | --- |
| 30 | 27,651,840 bytes | 0.093 s | 165,834,752 bytes | 171,671,552 bytes |
| 300 | 276,518,400 bytes | 0.516 s | 165,691,392 bytes | 171,536,384 bytes |

Both stores retained zero decoded arrays after loading; timestamps matched and
stores closed. The tenfold clip increased scratch storage tenfold, while peak
working set stayed near 164 MiB. These are acquisition measurements, not a
claim that full temporal result memory is constant. Evidence is under
`.cache/a11-a13-profile/memory30` and `memory300`.

## A12 — measured optimization targets

`tools/profile_analysis.py` creates isolated settings and database paths, uses
fresh uv subprocesses, records input hashes/dimensions, and writes JSON timing
records, import-time logs and cProfile reports. It does not modify algorithms.
The reference inputs use synthetic 100 px/mm for repeatable timing, not physical
accuracy validation. OS filesystem caches are not flushed. cProfile measurements
are kept separate from uninstrumented timing distributions.

September 8, 2026 samples (one first run, three warm runs per mode):

| Mode | First run | Warm median | Warm min–max |
| --- | --- | --- | --- |
| Sessile | 1.595 s | 1.522 s | 1.519–1.566 s |
| Pendant | 9.039 s | 1.504 s | 1.481–1.554 s |

All repeated runs had identical promoted numeric results and acceptance/rejection
status. The profiled warm sessile run spent 1.914 of 2.026 s in fitting, with
18 Young–Laplace integrations. Pendant spent 1.905 of 1.976 s in fitting, with
71 integrations and 69 residual calls. Numerical differentiation accounts for
much of the repeated pendant model evaluation. These are cumulative timings;
nested costs must not be summed.

A separate instrumented pendant first-use run spent 8.031 s in
`_selected_plane_lookup_all`, constructing the selected-plane approximation
lookup; total profiled execution was 11.699 s. It performed 737 integrations
across initialization and fitting. This identifies a concrete first-use target,
distinct from the warm solver cost. The initialization is already reused within
the process; any persistent/precomputed lookup would need a versioned numerical
contract and equivalence checks. Raw evidence is in
`.cache/a11-a13-profile/first-profile/pendant/first-use.txt`.

Three offscreen fresh-process startup-to-first-event-loop samples were 3.258,
3.162 and 3.216 s, including test isolation. The first import trace attributed
2.133 s cumulatively to pipeline discovery; it eagerly imports every pipeline.
NumPy and SciPy optimization contribute to that dependency graph. GUI module
import after discovery took another 0.281 s. Import tracing itself adds overhead.

Next candidates are repeated integration/model evaluations and separating
pipeline metadata discovery from optional scientific imports. Any cache must
include all physical/model parameters and preserve cancellation. Lazy imports
must preserve plugin registration and available modes. Neither change is
accepted as an optimization here: first measure it against these fixed inputs
and require unchanged numerical results, rejection and conformance tolerances.
Your ongoing methodology work is unchanged.

Reproduce from the repository root:

```powershell
$env:UV_CACHE_DIR='D:\uv_cache_dir'
$env:PYTHONUTF8='1'
uv run --offline --extra test python tools/profile_analysis.py --output .cache/profile-new
uv run --offline --extra test python tools/profile_analysis.py --output .cache/profile-first --case pendant --profile-first --repeats 1
```

For additional fresh startup samples, use `--case startup` with a distinct
output directory each time. Native working-set counters are Windows-specific;
other platforms report null memory fields. New profiler runs also record
dependency versions, effective parameters and registered source hashes.

## A13 — explicit export scopes

**Export all history** (toolbar/File menu) exports every retained history record,
ignoring the selected mode, table filters and hidden columns. It exports original
timestamps using `datetime.isoformat()`, original source, job ID, full precision
metrics and mandatory provenance. It does not retrieve records already removed
by the history retention limit.

`ResultsHistory.export_csv()` writes export schema `1.0`. Mandatory columns:

- `export_schema_version`, `id`, `timestamp`, `pipeline`, `schema_version`;
- `file_name`, `file_path`, `status`, `accepted`;
- human-readable `rejection_reasons` and unambiguous `rejection_reasons_json`;
- `calibration`, `px_per_mm`, `calibration_origin`, `units_json`;
- `results_json`, `diagnostics_json`, `run_metadata_json`.

Additional metric columns are sorted and unrounded; nested values use JSON.
JSON columns preserve original keys and values. Unit descriptions are canonical
metric units, independent of display-unit preferences; metric suffixes remain
authoritative. Export schema version is independent of the scientific result
schema. Missing legacy scientific versions/calibration are left unrecorded,
not inferred. Legacy naive timestamps retain their original date/time without
inventing a timezone. New jobs already use UTC submission timestamps.

New GUI jobs record application version and SHA-256 hashes of registered callable
source files in `run_metadata.runtime`, captured on the worker before execution.
This describes the registered environment, not proof that each plugin ran, and
does not certify binary dependencies. Requested settings and effective
calibration retain their existing metadata fields. Old rows remain loadable.

**Export current view** exports filtered/nonhidden rows and displayed columns
at display precision. Existing mandatory validation/calibration/source columns
are retained even when hidden, as explained by the dialog and tooltip. It is a
human-readable export; use all-history export for machine processing. Dynamic
canonical JSON/summary/frame exports retain their existing scientific contract.

## Validation routes

`tests/test_sequence_storage_exports.py` covers independent pixel ownership,
eager/disk timestamp and pixel equivalence, full temporal output equivalence
including initial scale and occlusions, decoder/store cancellation cleanup,
between-stage cleanup, full dates/precision/provenance and separate GUI export
scopes. Existing Phase-D, result history, execution, tracking, static fitting,
calibration and layout tests remain the surrounding regression checks.

Validation on this checkout: 87 execution/export/temporal tests passed, followed
by 92 nearby scientific/layout tests. The final export scope checks passed with
the current-view hidden-row handling. Model typing checked all 20 model files.
