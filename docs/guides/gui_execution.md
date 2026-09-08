# GUI execution and result ownership

The main window owns one `PipelineRunner` and one `AppSettings` instance.
Required startup services fail visibly; they are never replaced by no-op
objects. The results panel shares the window settings instance. Existing
settings and SOP locations remain unchanged.

## Submission and completion

Controllers collect widget values on the GUI thread and create a `RunRequest`.
It contains a UUID, UTC submission time, pipeline, source, operation, stage plan,
setup revision, and an owned copy of input data and settings. The runner takes
another owned copy at submission; a worker cannot observe subsequent edits.
Image decoding and prerequisite detection for analysis happen in the worker.

The private, single-thread pool accepts one operation per window. Analysis,
quick analysis, SOPs, stage tests, calibration, and preprocessing/edge previews
use this service. Submission failures are reported without synchronous retry.
The standalone preprocessing and edge controllers retain their synchronous
API for non-window callers; the main window attaches background preview adapters.

`RunCompletion` carries the original request, terminal state, and full context
or task value. `RunViewModel.completed` forwards that envelope. The controller
passes analytical contexts through `build_persisted_analysis` once per job ID.
Rejected contexts have empty physical results and retain reasons/diagnostics.
Failed and cancelled executions produce no measurement. Stages without results
produce no accepted measurement. Dynamic sequences remain one history summary;
their temporal context and existing scientific export contract are unchanged.

History IDs equal job UUIDs. Optional `MeasurementResult.run_metadata` records
submission identity and settings, plus effective calibration. Old history files
load with this field absent. Images, runtime cancellation objects, and Qt
objects are not persisted. CSV includes source and run metadata and retains
rejection information even when diagnostic columns are hidden in the UI.

The setup stays editable. Source/settings/geometry changes invalidate automatic
result presentation, not the submitted job. Such completions update history
without changing the current preview, cards, or selected measurement. Duplicate
and late terminal notifications cannot create a second row or finish a new job.

## Cancellation and shutdown

Execution states are `queued`, `running`, `stopping`, `completed`, `failed`, and
`cancelled`. Scientific rejection is a completed execution, not a worker failure.
Stop cancels the current job's thread-safe token. A cancellation received before
GUI completion handling wins the race and suppresses the returned result.

`Context.cancellation_token` is declared explicitly and excluded from
serialization. A scoped context variable supplies the same token to nested
algorithm calls without changing plugin signatures. Shared acquisition,
temporal, and solver entry points also accept optional checkpoint callbacks.
With no token/callback, headless numerical behavior is unchanged.

`AnalysisCancelled` derives from `BaseException`, like Python's asynchronous
cancellation exception. Broad scientific `except Exception` handlers therefore
cannot translate Stop into a failed fit, rejected frame, or fallback estimate.
Workers catch it separately and report cancellation after resource cleanup.

Checkpoints run around stages, between acquisition/temporal work units, between
calibration passes, and in Python solver callbacks. Native calls that do not
return to Python cannot be interrupted safely. The UI stays in `stopping` until
they return. Closing requests cancellation and defers window destruction until
the worker has left the pool; the GUI event loop keeps running throughout.

## Verification

Run `uv run --extra test python tools/check_gui_execution.py` for isolated
startup, worker, controller, history, calibration, and scientific regression
tests. It redirects settings, Qt preferences, and database paths into a fresh
`.cache/execution-checks/` directory and writes JUnit evidence there. Additional
arguments select tests or pass pytest options.

The lifecycle tests cover a 500 ms worker with a 20 ms GUI heartbeat,
cancellation races and uninterruptible work, snapshot ownership, duplicate and
out-of-order notifications, and rejected/static/dynamic result routing.
Real-window tests cover shared services, restart persistence, stale result
selection, calibration dialog closure, and asynchronous shutdown.

For the visible Windows check, run `uv run python tools/smoke_gui_execution.py`.
It opens an isolated window, switches pipeline during a synthetic rejected run,
and measures cooperative and uninterruptible cancellation separately. Screenshots
and `report.json` are written under `.cache/native-execution-smoke/`.

The 2026-09-06 native check used Windows 11, Python 3.14.7, PySide6 6.11.2,
and a 320 × 240 synthetic image. During a 500 ms worker, 26 heartbeat callbacks
ran with a maximum gap of 21.0 ms. Cancellation took 5.6 ms with cooperative
checkpoints and 283.1 ms while waiting for a simulated 350 ms native call.
These are execution-control checks, not scientific throughput benchmarks.
Initial screenshot capture took 1.6 seconds and is excluded from the active-run
heartbeat interval; no application-startup timing claim is made.

Preset expansion and streaming memory redesign remain separate work.
History recovery and calibration publication are described in
[A06–A07](history_recovery_calibration.md).

## Dynamic Sessile and independent-image folders (A04–A05)

Dynamic Sessile is available in the primary analysis selector. Choose File for
a video or Folder for a frame sequence, set Sequence FPS for frame folders,
then choose Run sequence. The existing temporal pipeline produces one summary
with its full context for the timeline and scientific exports. Camera input is
disabled for this mode. Static folder actions are hidden in Dynamic Sessile.

For static pipelines, Folder offers Run selected and Run folder (file count).
Run selected preserves the existing selected-image workflow. Run folder snapshots
all supported image paths, pipeline and settings, then analyzes each image
independently with automatic calibration. Selected-preview geometry, markers,
and detected scale are not carried to unrelated images. This does not alter CLI
folder tracking or the evolving scientific methodologies.

`folder_execution.folder_task` uses `pipeline_runner.execute_request`, the same
execution function as individual GUI runs. A parent job owns the window pool
throughout the folder; each file has its own immutable request and UUID.
`FolderEvent` streams running and terminal outcomes to `FolderController` on the
GUI thread. Accepted and rejected contexts pass through the existing idempotent
history builder exactly once. Folder records never replace the selected preview.

The Folder results dialog reports per-file errors and rejection reasons. Closing
it leaves computation running; Folder results reopens it. Stop requests cooperative
cancellation and preserves file outcomes committed before that request. The active
file and remaining queue become cancelled. Window closure follows the shared
deferred shutdown lifecycle. Corrupt files fail individually; later files continue.
Retry failures creates new UUIDs with the original settings and retains prior
attempt rows, marked retried. Scientific rejections are not execution failures.

Export CSV writes the current folder's attempts, source paths, pipeline, statuses,
error/rejection details, diagnostics, run metadata and union of analytical metrics.
It includes failed/cancelled rows without invented metrics. Completed measurements
also remain available in normal history. The per-attempt panel itself is session
state; a new folder run replaces it, and history persistence behavior is unchanged.

Run `uv run --extra test python tools/check_gui_execution.py
tests/test_gui_folder_execution.py` for deterministic GUI tests covering keyboard
access, one-sequence submission, independent folder identities, corrupt-file
continuation, retry, cancellation, heartbeat and consolidated CSV. These tests
exercise execution contracts without asserting numerical methodology results.
