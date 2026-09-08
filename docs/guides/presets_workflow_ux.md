# Analysis presets and workflow readability (A08–A10)

## Reusable presets

Open **Advanced** to access SOP controls. **Add** captures a named analysis
preset; **Update** replaces the selected named preset. **Export** writes a portable
JSON file and **Import** validates one before storing it. Duplicate import names
are rejected so an imported file cannot silently overwrite an existing preset.
Select a preset and choose **Apply** to review its stages, pipeline, units,
algorithm, geometry and version before replacing the current setup.

The version-1 `AnalysisPreset` model contains typed preprocessing and edge settings,
markers, physics/calibration and mode-specific control values, selected stages,
pipeline settings, units, pixel geometry and plugin source fingerprints. Images,
paths and current source selections are excluded. Applying a preset replaces
each captured configuration group; it does not merge in values from the previous
pipeline. Pixel geometry is deliberately reusable, so the review asks users to
check its fit to the current source before analysis.

Presets remain in `~/.menipy/sops.json` under their pipeline, with a versioned
`__preset__` payload inside the existing SOP parameters. Legacy stage-only SOPs
continue to toggle stage inclusion automatically. Complete presets require the
explicit Apply action. The default SOP remains stage-only. SOP writes now replace
the file atomically and restore in-memory state if saving fails.

Unknown schema versions, missing pipelines, unsupported stages/control values,
invalid geometry and missing/changed plugin files are reported before applying.
The plugin manifest fingerprints Python files in configured plugin directories;
it checks source compatibility, not binary dependency reproducibility. Menipy
version differences are shown in the review. A preset does not install plugins
or reproduce an external Python environment.

## Source readiness and accessibility

Run and calibration submission controls are disabled without a source, with a
reason beside Run. Source or analysis-mode changes clear calibration overlays,
markers and cached calibration, and mark prior preview/results as belonging to
an earlier setup. History is preserved. A calibration dialog opened for an earlier
source cannot apply its result after that source changes. Busy execution still
uses the existing single-worker boundary while setup remains editable.

Icon controls have accessible names, primary calibration labels have buddies,
and keyboard focus has a visible border. **Config → Show analysis mode labels**
persists the option to display names on every mode button. Source controls occupy
their own row, and the setup panel scrolls when the window is short.

## Readable calibration and results

Calibration initially fits the whole image using the viewport dimensions and
preserves aspect ratio during resizing. **100%** remains available for inspection;
**Fit to Window** returns to fitting. Region confidence appears on a separate
line from its Draw action, so doubtful detection text remains readable.
File-based dialogs decode the initial preview in the existing background service;
viewing the image does not require running the detectors first.

Result filters and actions occupy separate rows. A populated results table gets
enough minimum height to show the first row, with the existing splitter presets
and saved geometry retained. Smaller saved windows remain usable through the
scrolling setup panel.

## Verification

`uv run --extra test python tools/check_gui_execution.py
tests/test_presets_readiness_layout.py` checks preset round-trips, fresh-process
loading, compatibility rejection, source invalidation and image/status fitting.
The existing setup, calibration and startup tests cover retained behavior.

`uv run python tools/smoke_presets_layout.py` writes isolated native screenshots
and dimensions under `.cache/presets-layout-smoke`. Native 100% and 125% checks
verified exact 1200×800 and 1366×768 logical windows with full result rows.
At 150%, Windows constrained the requested dimensions to this monitor's available
size; readable rows were verified at that constrained size. Exact-size native
150% verification requires a larger display. Qt's `QT_SCALE_FACTOR` multiplies
the system scale; consult the report's `device_pixel_ratio`, not that variable
alone, when interpreting the evidence.
