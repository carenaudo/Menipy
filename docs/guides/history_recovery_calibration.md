# History recovery and calibration provenance (A06–A07)

## Failed history saves

History is serialized before opening a temporary file in the destination directory.
The complete UTF-8 file is flushed and fsynced, then atomically replaces
`~/.menipy/measurement_history.json`. Failed serialization, writes, flushes or
replacement leave the previous file intact. Startup reads only the history file,
never abandoned temporary files. Atomic replacement requires filesystem support;
this does not promise survival of every hardware or filesystem failure.

`ResultsHistory.unsaved` and `save_error` report failure. Measurements stay in
memory; history trimming pauses during the failure so later completed jobs cannot
evict unsaved records. The results panel shows an unsaved notice with **Retry save**
and **Save recovery copy**. Retry writes the current in-memory history and clears
the notice only on success. Recovery export writes all records, including hidden
pipelines, to a separate reloadable JSON file; it does not claim the normal history
destination was repaired. Normal CSV export also remains available.

Closing with unsaved changes asks whether to cancel closure or discard those
in-memory changes. Cancel closure to retry or export. To restore a recovery JSON,
close Menipy and replace `~/.menipy/measurement_history.json` with that copy, retaining
a backup of any current history first. Recovery restores the saved snapshot; it
does not merge histories. No application settings or history location changed.

## Calibration publication

Optional `Context.calibration_provenance` describes `manual`, `measured`,
`estimated`, or `missing` scale, its px/mm value, component confidence and warnings.
GUI workers attach it after computation. Manual calibration survives automatic
stage prerequisite detection. Wizard-drawn regions are identified explicitly.
Temporal calibration retains the existing explicit-scale/needle-median origin.
CLI image and camera exports also identify their supplied or fallback scale;
CLI temporal tracking algorithms are unchanged.

Only finite positive manual or measured scale enables physical-value publication.
Estimated or missing calibration allows computation and diagnostic previews but
withholds the analytical metrics from persisted history and CSV. Such a record is
displayed as **Uncalibrated** when scientific QA otherwise passes. Supply valid
calibration and rerun to publish metrics. Scientific rejection remains Rejected;
its thresholds, reasons and empty-results contract are unchanged. No numerical
solver or acceptance threshold was changed. Diagnostic previews are not verified
physical measurements.

`build_persisted_analysis` adds a calibration diagnostics envelope, including
`physical_values_withheld`. GUI results show a calibration summary independent of
the Diagnostics toggle; CSV retains this summary even when its column is hidden.
Stage tests show scale origin and warnings and withhold uncalibrated analytical
values from their output panel. Folder rows and exports retain the same warnings.
Low substrate confidence remains flagged below the existing 0.75 warning level,
regardless of aggregate confidence. Other component scores below 0.5 get a
presentation warning, not a new scientific rejection gate.

Legacy contexts and histories without provenance remain readable and retain their
prior behavior. They display **Not recorded (legacy)** instead of claiming that
calibration was verified. Headless callers may explicitly attach provenance to
participate in the publication policy without changes to pipeline signatures.

## Verification

Use `uv run --extra test python tools/check_gui_execution.py
tests/test_history_recovery_calibration.py` for isolated fault injection, retry,
recovery exports, manual-scale preservation, component warnings and publication.
Existing calibration, rejection, temporal and execution tests remain separate
scientific/regression checks. Native Windows smoke captures under
`.cache/recovery-native-smoke/` use synthetic data and isolated persistence.
