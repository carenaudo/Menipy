# Menipy improvement audit — 5 September 2026

The highest-value first release is reliable startup, validated job completion and responsive execution. Several capabilities already exist in the code but are disconnected from the native application. Restore those paths before expanding features; restoring the worker alone would expose a separately reproduced result-integrity defect.

Deliverables: 15 ranked recommendations, 9 accepted native captures (8 workflow steps plus export dialog), runtime evidence, reproducible probes and a CSV backlog. Existing focused checks: 69 passed, 1 skipped. No application or public API changes were made by this audit.

## What changed from the initial assessment

| Initial hypothesis | Audit conclusion |
| --- | --- |
| Some GUI routes are synchronous | Stronger: incorrect startup imports disable the normal runner entirely. The main-window fallback is the current native path. |
| Background results lose validation metadata | Reproduced through a real worker/view-model signal route, but latent in the current native startup until A01 is repaired. |
| Missing calibration is silently estimated | Normal acquisition blocks absent ROI/needle. Advanced configuration can estimate scale and returns a warning; inconsistent treatment/provenance is the confirmed issue. |
| Large history is a present scalability bottleneck | Default retention is 100. The 1,000-row experiment is a stress scenario, not default operating behavior. |
| Customization needs new storage | Real JSON/QSettings/SQLite services preserve the tested preferences across restart. Wiring and completeness are the immediate gaps. |
| Dynamic video features need adding | Core and test coverage exist; the Dynamic button is hidden by the main-window composition. |

## Ranked backlog

P1: correctness or core workflow; P2: substantial usability/scaling improvement; P3: follow-on investment. Effort is a rough engineering-day estimate including focused tests, not a delivery commitment. Dependencies identify changes that should be designed or released together.

| ID | Priority | Area | Opportunity | Evidence class | Effort (days) |
| --- | --- | --- | --- | --- | --- |
| A01 | P1 | Functionality / speed / customization | Restore startup services without silent fallback | Confirmed current defect | 1–2 |
| A02 | P1 | Functionality | Bind completed results to the original job and validation context | Confirmed latent defect | 2–3 |
| A03 | P1 | Speed / UX | Keep computation off the UI thread and make Stop meaningful | Confirmed current and latent defects | 3–5 |
| A04 | P1 | Functionality / UX | Expose Dynamic Sessile in the real application | Confirmed current defect | 0.5–1 |
| A05 | P1 | Functionality / UX | Make folder processing explicit | Confirmed current behavior / missing capability | 3–5 |
| A06 | P1 | Functionality / reliability | Report and recover from failed history writes | Confirmed current defect | 1–2 |
| A07 | P2 | Functionality / UX | Carry calibration provenance and warnings through every path | Confirmed path inconsistency | 2–3 |
| A08 | P2 | Customization | Turn SOPs into complete reusable analysis presets | Confirmed latent gap | 3–5 |
| A09 | P2 | UX / accessibility | Name icon controls and guide source readiness | Confirmed UI/API evidence | 1–3 |
| A10 | P2 | UX | Keep calibration status and result rows readable | Confirmed visual issue | 1–2 |
| A11 | P2 | Speed / scale | Bound decoded video and detection memory | Measured scaling cost | 5–10 |
| A12 | P2 | Speed | Profile fitting and startup imports before broad optimization | Measured hotspot / investigation | 2–4 investigation; implementation sized after profile |
| A13 | P2 | Functionality / UX | Clarify export scope and preserve machine-readable provenance | Confirmed source and output evidence | 1–2 |
| A14 | P3 | Speed / customization | Improve history retention deliberately before scaling the table | Measured future scaling cost | 2–3 |
| A15 | P3 | Customization / UX | Offer a coherent appearance and workspace preference surface | Exploratory product idea | 2–4 |

### A01 · Restore startup services without silent fallback

P1 | Confirmed current defect | Confidence: High | Effort: 1–2 days | Dependencies: A02 and A03 must accompany release.

Evidence: Normal import and native launch both yield runner=None, run_vm=None, sops=None. AppSettings comes from main_window. The import targets views.services, which is absent.

User impact: The desktop silently loses background execution, SOP availability, and main-window persistence.

Recommendation: Correct the service imports, narrow optional-feature exception handling, and report unavailable required services. Declare persisted layout fields and avoid overwriting restored geometry at startup.

Acceptance: Unpatched MainWindow has real runner/view model/SOP/settings objects. Across a fresh process, selected pipeline, units, image, overlays and supported layout fields restore. Test startup service identities rather than patching them away.

Source: [src/menipy/gui/views/main_window.py:113](D:/programacion/Menipy/src/menipy/gui/views/main_window.py:113), [src/menipy/gui/services/settings_service.py:17](D:/programacion/Menipy/src/menipy/gui/services/settings_service.py:17), [src/menipy/gui/app.py:135](D:/programacion/Menipy/src/menipy/gui/app.py:135)

### A02 · Bind completed results to the original job and validation context

P1 | Confirmed latent defect | Confidence: High | Effort: 2–3 days | Dependencies: A01; release together.

Evidence: A real worker/view-model signal probe returned qa.ok=False and an original image path. The resulting history row was Accepted, had empty rejection reasons and no filename, and used the newly selected Pendant pipeline.

User impact: Restoring the runner as-is exposes incorrect validation status and misattributed records.

Recommendation: Use one completion envelope with job identity, submitted pipeline/source/settings and full Context. Persist through build_persisted_analysis exactly once for every mode; do not infer identity from current controls.

Acceptance: Rejected static and dynamic jobs retain rejection reasons/diagnostics with empty accepted metrics. Switching pipeline/source mid-run cannot change stored identity. Success, rejection, duplicate signals and out-of-order completion create exactly one correct row each.

Source: [src/menipy/gui/controllers/pipeline_controller.py:875](D:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py:875), [src/menipy/gui/controllers/pipeline_controller.py:898](D:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py:898), [src/menipy/gui/views/results_panel.py:1191](D:/programacion/Menipy/src/menipy/gui/views/results_panel.py:1191), [src/menipy/models/results.py:40](D:/programacion/Menipy/src/menipy/models/results.py:40)

### A03 · Keep computation off the UI thread and make Stop meaningful

P1 | Confirmed current and latent defects | Confidence: High | Effort: 3–5 days | Dependencies: A01, A02.

Evidence: A 20 ms Qt timer fired after 253.2 ms while a 250 ms synthetic direct pipeline ran. The worker probe still completed 352.2 ms after Stop; Stop only clears queued tasks.

User impact: Users cannot reliably interact with or stop long analysis; status can imply cancellation that never happened.

Recommendation: Route full, simple, stage-test and calibration computation through workers. Replace synchronous fallback with a submission error. Add job-scoped cancellation checks between stages/frames and callbacks inside long solvers where supported; report stopping while an uninterruptible operation finishes.

Acceptance: A synthetic 500 ms job leaves a 20 ms GUI heartbeat running; no duplicate submissions. Stop cancels queued jobs and suppresses cancelled result publication, preserves completed history, and reports terminal cancellation once. Measure solver cancellation latency separately; never claim instantaneous interruption of native calls.

Source: [src/menipy/gui/controllers/pipeline_controller.py:1066](D:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py:1066), [src/menipy/gui/controllers/pipeline_controller.py:667](D:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py:667), [src/menipy/gui/controllers/main_controller.py:332](D:/programacion/Menipy/src/menipy/gui/controllers/main_controller.py:332), [src/menipy/gui/dialogs/calibration_wizard_dialog.py:356](D:/programacion/Menipy/src/menipy/gui/dialogs/calibration_wizard_dialog.py:356)

### A04 · Expose Dynamic Sessile in the real application

P1 | Confirmed current defect | Confidence: High | Effort: 0.5–1 days | Dependencies: Independent of A01; A03 for responsive execution.

Evidence: The controller has dynamicSessileBtn, but its parent is hidden. The toolbar reparents only five static mode buttons. Native startup shows those five buttons, although the core discovers and tests sessile_dynamic.

User impact: A supported analysis capability is inaccessible through the primary workflow.

Recommendation: Include Dynamic Sessile in the primary mode selector and expose its sequence/FPS controls. Add a visible-window integration test across every discovered supported mode.

Acceptance: Keyboard and pointer users can select Dynamic Sessile without private APIs, choose video or frame directory, supply FPS where needed, and reach timeline results/export. Static mode controls remain correct.

Source: [src/menipy/gui/views/main_window.py:550](D:/programacion/Menipy/src/menipy/gui/views/main_window.py:550), [src/menipy/gui/views/main_window.py:555](D:/programacion/Menipy/src/menipy/gui/views/main_window.py:555), [src/menipy/gui/views/main_window.py:691](D:/programacion/Menipy/src/menipy/gui/views/main_window.py:691)

### A05 · Make folder processing explicit

P1 | Confirmed current behavior / missing capability | Confidence: High | Effort: 3–5 days | Dependencies: A02, A03.

Evidence: With three images in a selected folder, gather_run_params provides the folder plus a selected image, but run_full submits exactly one image. Folder mode continues to label its action Run Analysis.

User impact: Users may believe a folder was processed when only one file was analyzed.

Recommendation: Separate Run selected and Run folder; display file count, per-file outcomes, progress, retry failures and a consolidated export. Keep a temporal frame directory distinct from an independent-image batch.

Acceptance: A three-file folder produces three identified outcomes; one corrupt file is reported without losing the others. Cancellation retains completed records. Dynamic sequence input remains one temporal result, not three unrelated analyses.

Source: [src/menipy/gui/controllers/setup_panel_controller.py:426](D:/programacion/Menipy/src/menipy/gui/controllers/setup_panel_controller.py:426), [src/menipy/gui/controllers/pipeline_controller.py:571](D:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py:571)

### A06 · Report and recover from failed history writes

P1 | Confirmed current defect | Confidence: High | Effort: 1–2 days | Dependencies: Independent; coordinate with A02.

Evidence: Injecting PermissionError during add_measurement returned normally and left a new in-memory row. _save_history catches every exception without logging or a user-visible error.

User impact: A displayed measurement can disappear after restart without warning.

Recommendation: Use atomic persistence and a visible unsaved state with retry/export recovery. Preserve the last good file and keep measurements recoverable in memory.

Acceptance: Read-only destination, simulated disk-full and interrupted write retain the prior file. Users see unsaved status; successful retry clears it. Restart never reads a partially written JSON file.

Source: [src/menipy/models/results.py:235](D:/programacion/Menipy/src/menipy/models/results.py:235), [src/menipy/models/results.py:90](D:/programacion/Menipy/src/menipy/models/results.py:90)

### A07 · Carry calibration provenance and warnings through every path

P2 | Confirmed path inconsistency | Confidence: High | Effort: 2–3 days | Dependencies: A02; scientific contract review.

Evidence: Normal Run blocks absent ROI/needle regions. The advanced run builder instead supplied 10 px/mm with a fallback warning when no regions were present. The native wizard allowed Apply All with substrate confidence 25% and overall confidence 81%.

User impact: Different entry points can lead to different confidence assumptions; aggregate confidence can hide a weak prerequisite.

Recommendation: Record scale origin (measured/manual/estimated/missing), show component warnings beside results and export them. Keep preview possible, but require explicit handling of estimated scale before publishing calibrated physical values. Do not change scientific rejection thresholds as part of UI cleanup.

Acceptance: Missing or estimated scale is visible in stage tests and exports. Manual valid calibration is honored. A weak substrate remains visibly flagged even when overall confidence is high. Existing numerical conformance gates continue to pass.

Source: [src/menipy/gui/controllers/pipeline_controller.py:206](D:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py:206), [src/menipy/gui/controllers/pipeline_controller.py:55](D:/programacion/Menipy/src/menipy/gui/controllers/pipeline_controller.py:55), [src/menipy/gui/dialogs/calibration_wizard_dialog.py:450](D:/programacion/Menipy/src/menipy/gui/dialogs/calibration_wizard_dialog.py:450)

### A08 · Turn SOPs into complete reusable analysis presets

P2 | Confirmed latent gap | Confidence: High | Effort: 3–5 days | Dependencies: A01, A02.

Evidence: SopController creates presets with params={}; applying them only sets stage inclusion. SopService can persist parameter dictionaries, but normal startup currently disables the service.

User impact: Researchers cannot reliably repeat or share a full configured workflow using SOPs.

Recommendation: Capture typed preprocessing, detection, physics, geometry, calibration choices, plugin selections and stage inclusion. Support named save/update/import/export with versioning and a summary before applying.

Acceptance: After process restart, a preset round-trips all supported values and reproduces run configuration. Missing plugin/version conflicts are explained. Applying a preset does not silently retain incompatible settings from the previous mode.

Source: [src/menipy/gui/controllers/sop_controller.py:96](D:/programacion/Menipy/src/menipy/gui/controllers/sop_controller.py:96), [src/menipy/gui/controllers/sop_controller.py:200](D:/programacion/Menipy/src/menipy/gui/controllers/sop_controller.py:200), [src/menipy/gui/services/sop_service.py:21](D:/programacion/Menipy/src/menipy/gui/services/sop_service.py:21)

### A09 · Name icon controls and guide source readiness

P2 | Confirmed UI/API evidence | Confidence: High | Effort: 1–3 days | Dependencies: Coordinate with A04, A05.

Evidence: The native accessibility tree reports blank names for inactive mode buttons, source toggles and panel toggles. Startup enables Run Analysis with no image; switching to an empty folder leaves the old preview/results visible.

User impact: Controls are hard to discover without hovering, and the current source can be confused with a previous result.

Recommendation: Set explicit accessible names and label associations, offer persistent mode labels, show source readiness and explain prerequisites next to Run. Clearly mark previous results when source selection changes.

Acceptance: Keyboard-only selection, loading, calibration, run and export have visible focus and meaningful names. Empty-source Run is disabled with a reason or routes to source selection. Source changes invalidate stale calibration or visibly associate it with its source.

Source: [src/menipy/gui/views/main_window.py:539](D:/programacion/Menipy/src/menipy/gui/views/main_window.py:539), [src/menipy/gui/controllers/setup_panel_controller.py:691](D:/programacion/Menipy/src/menipy/gui/controllers/setup_panel_controller.py:691)

### A10 · Keep calibration status and result rows readable

P2 | Confirmed visual issue | Confidence: High | Effort: 1–2 days | Dependencies: Independent; retain existing layout presets.

Evidence: At the captured native size, calibration opens zoomed so the drop base is out of view; detected-region status text is clipped beside Draw buttons. After analysis the results table shows a header and only a sliver of the selected row.

User impact: Users must adjust layout before verifying detections or reading the measurement table.

Recommendation: Fit the complete image on initial calibration display, allocate enough width for status text, and reserve usable table height when results exist. Keep splitter controls and saved layouts.

Acceptance: At 1200×800 and 1366×768 logical layouts, and 100/125/150% display scaling, the first result row and all confidence text remain readable. Window resizing preserves image aspect ratio and offers access to every action.

Source: [src/menipy/gui/dialogs/calibration_wizard_dialog.py:38](D:/programacion/Menipy/src/menipy/gui/dialogs/calibration_wizard_dialog.py:38), [src/menipy/gui/views/main_window.py:299](D:/programacion/Menipy/src/menipy/gui/views/main_window.py:299)

### A11 · Bound decoded video and detection memory

P2 | Measured scaling cost | Confidence: High | Effort: 5–10 days | Dependencies: A03; temporal contract and scientific conformance tests.

Evidence: Separate-process 640×480 MJPG probes retained 26.37 MiB for 30 frames and 263.67 MiB for 300 frames. Process peak working set rose from 171.08 to 408.58 MiB. Temporal analysis additionally builds a detection list for all frames.

User impact: Long recordings can exhaust memory before results are available.

Recommendation: Introduce incremental decoding with bounded frame/detection buffers or a disk-backed sequence store, and retain lightweight result series. Preserve timing, ordering, scale initialization, reacquisition and classification semantics.

Acceptance: A 10× longer clip does not retain 10× decoded pixels. Outputs match existing fixtures within established tolerances, including first-five scale samples, occlusions, rejection, timestamps and export order. Cancellation releases decoder resources.

Source: [src/menipy/common/sequence_acquisition.py:82](D:/programacion/Menipy/src/menipy/common/sequence_acquisition.py:82), [src/menipy/common/temporal_sessile.py:195](D:/programacion/Menipy/src/menipy/common/temporal_sessile.py:195)

### A12 · Profile fitting and startup imports before broad optimization

P2 | Measured hotspot / investigation | Confidence: High | Effort: 2–4 investigation; implementation sized after profile days | Dependencies: A03 first; preserve numerical contracts.

Evidence: Sampled warm pipelines took 1.48 s sessile and 1.90 s pendant; fitting dominated their last-run stage time (~93% and ~92%). One pendant first call took 9.02 s. Fresh-process startup samples ranged 3.18–6.64 s and were dominated by import/Qt setup.

User impact: These are the most defensible targets for reducing waiting in the sampled workflows.

Recommendation: Profile solver evaluations, initialization and repeated model work on controlled fixtures. Separate first-use costs from warm execution; inspect eager imports and optional initialization. Keep existing preview caching, which already makes repeat loads sub-millisecond in this probe.

Acceptance: Record distributions on fixed inputs and environment with numerical equivalence. Accept optimizations only with unchanged rejection/fit behavior and demonstrable timing improvement; no claimed speedup based on these baseline samples alone.

Source: [src/menipy/pipelines/base.py:424](D:/programacion/Menipy/src/menipy/pipelines/base.py:424), [src/menipy/pipelines/discover.py:59](D:/programacion/Menipy/src/menipy/pipelines/discover.py:59), [src/menipy/gui/views/preview_panel.py:163](D:/programacion/Menipy/src/menipy/gui/views/preview_panel.py:163)

### A13 · Clarify export scope and preserve machine-readable provenance

P2 | Confirmed source and output evidence | Confidence: High | Effort: 1–2 days | Dependencies: A02; result contracts.

Evidence: Top Export CSV writes history, while the table Export writes visible cells/columns. The actual native export contains a time-of-day timestamp without its date and lacks schema_version and explicit calibration scale columns.

User impact: Users can export a different scope than expected; cross-day analysis and reproducibility lose context.

Recommendation: Label Export all history and Export current view distinctly. Add a canonical machine export with full ISO timestamp, source identity, schema version, calibration and settings/plugin provenance; keep human-readable view export.

Acceptance: Both actions explain scope and preserve filters as labeled. Machine export round-trips dates, units, validation and original sources; hidden view columns cannot silently remove mandatory provenance from the machine export.

Source: [src/menipy/gui/controllers/main_controller.py:448](D:/programacion/Menipy/src/menipy/gui/controllers/main_controller.py:448), [src/menipy/gui/views/results_panel.py:428](D:/programacion/Menipy/src/menipy/gui/views/results_panel.py:428), [src/menipy/models/results.py:175](D:/programacion/Menipy/src/menipy/models/results.py:175)

### A14 · Improve history retention deliberately before scaling the table

P3 | Measured future scaling cost | Confidence: High | Effort: 2–3 days | Dependencies: A06; A05 if batch increases history volume.

Evidence: Default history retains 100 records. Warm table refresh was 76 ms at 100 and 633 ms at an artificially populated 1,000 records; save was ~10 and 49 ms respectively. The 1,000-row case exceeds the default retention cap.

User impact: Larger projects would make full rebuilding noticeable; current users first need to understand the 100-record retention limit.

Recommendation: Expose retention/export behavior, then consider project-backed history and an incremental Qt table model if larger histories are required. Avoid prioritizing a database rewrite solely from the 1,000-row stress case.

Acceptance: Retention is visible and configurable with deliberate archival behavior. Default-size refresh remains responsive; larger-history targets are benchmarked before committing to a storage migration.

Source: [src/menipy/models/results.py:76](D:/programacion/Menipy/src/menipy/models/results.py:76), [src/menipy/gui/views/results_panel.py:482](D:/programacion/Menipy/src/menipy/gui/views/results_panel.py:482)

### A15 · Offer a coherent appearance and workspace preference surface

P3 | Exploratory product idea | Confidence: Medium | Effort: 2–4 days | Dependencies: A01; validate demand before adding theme/GPU controls.

Evidence: Units, overlay/marker dictionaries, result columns, splitters, analysis QSettings and plugin DB settings have existing persistence mechanisms. The unused SettingsDialog advertises GPU/thread/theme options but has no discovered caller; it is not evidence those controls are available in the product.

User impact: A single preference surface could make existing customization understandable, with font/density/theme choices added only when backed by real behavior.

Recommendation: Consolidate existing preferences and add reset/export/import. Consider readable font size, density and light/dark/system appearance. Do not expose GPU or parallel switches until supported and measured.

Acceptance: Every visible preference loads, applies and survives restart, has a reset path and accurately describes scope. Unsupported controls are absent or clearly disabled; no cosmetic setting claims performance gains.

Source: [src/menipy/gui/services/settings_service.py:17](D:/programacion/Menipy/src/menipy/gui/services/settings_service.py:17), [src/menipy/gui/dialogs/settings_dialog.py:27](D:/programacion/Menipy/src/menipy/gui/dialogs/settings_dialog.py:27)

## Measured performance

Host: Windows 11, 12 logical CPUs (AMD64 Family 23 Model 96), Python 3.14.7, Qt 6.11.2, OpenCV 5.0.0, NumPy 2.5.2. Core probes ran offscreen; native screenshots used the Windows desktop. First-call and warm values are separate: warm is the median of calls 2–3. No OS cache flush or cold-boot benchmark was performed. These small samples are observations, not statistically stable p95 targets; other desktop applications were running and the early probe phase overlapped the focused tests.

| Operation | Input / repetitions | First call | Warm / subsequent | Interpretation |
| --- | --- | --- | --- | --- |
| Startup, fresh process (offscreen) | 3 processes | 6639 ms | 3497 ms subsequent-process median | 3.18–6.64 s range; OS caches not flushed |
| Startup, native | 1 launch | 3407 ms | — | Includes isolation and widget creation, excludes uv startup |
| Sessile preview | 1600×1200, n=3 | 40.25 ms | 0.44 ms | Preview warm sample is same-path caching |
| Sessile calibration | 1600×1200, n=3 | 128.84 ms | 111.91 ms | Calibration = core detector, not entire wizard |
| Sessile pipeline | 1600×1200, n=3 | 1703.26 ms | 1481.85 ms | Explicit synthetic scale; numerical accuracy not assessed |
| Pendant preview | 1280×1024, n=3 | 23.89 ms | 0.54 ms | Preview warm sample is same-path caching |
| Pendant calibration | 1280×1024, n=3 | 11.18 ms | 10.16 ms | Calibration = core detector, not entire wizard |
| Pendant pipeline | 1280×1024, n=3 | 9022.36 ms | 1895.59 ms | Explicit synthetic scale; numerical accuracy not assessed |
| History save | 10 rows, n=3 | 2.01 ms | 3.36 ms | Small scalar result fixture; large diagnostic payloads not benchmarked |
| History table_refresh | 10 rows, n=3 | 26.31 ms | 23.17 ms | Small scalar result fixture; large diagnostic payloads not benchmarked |
| History save | 100 rows, n=3 | 7.04 ms | 9.66 ms | Small scalar result fixture; large diagnostic payloads not benchmarked |
| History table_refresh | 100 rows, n=3 | 104.05 ms | 76.33 ms | Small scalar result fixture; large diagnostic payloads not benchmarked |
| History save | 1000 rows, n=3 | 45.19 ms | 48.52 ms | Above default 100-row cap |
| History table_refresh | 1000 rows, n=3 | 628.35 ms | 632.72 ms | Above default 100-row cap |

| Decode input | Time | Retained image pixels | Working set before → after | Process peak |
| --- | --- | --- | --- | --- |
| 30 frames, 640×480 RGB, 30 fps MJPG | 63.6 ms | 26.37 MiB | 139.9 → 168.2 MiB | 171.1 MiB |
| 300 frames, 640×480 RGB, 30 fps MJPG | 276.4 ms | 263.67 MiB | 139.8 → 405.8 MiB | 408.6 MiB |

Memory samples use separate processes and real MJPG decoding. Peak working set includes imports and fixture encoding; retained pixel bytes are summed from decoded arrays. Full-resolution long-video analysis and GPU performance were not measured. Stage-level raw timings and sample arrays are in evidence.json. The cancellation probe measures completion after an ineffective Stop, not successful cancellation latency.

## Native workflow evidence

All displayed images below are current-run captures saved and inspected without editing. The first occluded capture was rejected and replaced. The tool later stopped exposing the audit window after folder-mode capture; no later error screenshot or crash claim is included. Source fixture attribution remains in data/MANIFEST.json and data/ATTRIBUTION.md; no external ground-truth claim was independently verified.

| Step | Flow | Health |
| --- | --- | --- |
| 1 | Startup | Needs improvement |
| 2 | Source selection | Working, with clarity gaps |
| 3 | Calibration entry | Needs layout improvement |
| 4 | Detection and Apply All | Warnings need greater prominence |
| 5 | Analysis and history | Completes; result inspection constrained |
| 6 | CSV export | Working; scope/provenance gaps |
| 7 | Advanced workflow | Visible controls; unavailable SOP backend |
| 8 | Folder mode | Ambiguous processing scope |
| 9 | Dynamic, pendant and hardware coverage | Mixed coverage; explicit limits |

### 1. Startup — Needs improvement

A strong central preview and clear Calibrate/Run actions are present. Run is enabled with no source; unlabeled icon controls and the missing Dynamic option reduce discoverability. Native UIA names were inspected; full screen-reader navigation was not completed.

![Step 1: Startup](D:/programacion/Menipy/audit-output/screenshots/01-start.png)

### 2. Source selection — Working, with clarity gaps

The native file picker loaded the local 1600×1200 sessile reference successfully. The preview fits the image; path display is abbreviated. Returning to an empty folder later leaves the previous analysis visible (step 8).

![Step 2: Source selection](D:/programacion/Menipy/audit-output/screenshots/02-source.png)

### 3. Calibration entry — Needs layout improvement

Manual drawing and Fit to Window are available. Initial zoom hides the bottom of the drop in this dialog, making substrate review harder until the user fits or scrolls.

![Step 3: Calibration entry](D:/programacion/Menipy/audit-output/screenshots/03-calibration.png)

### 4. Detection and Apply All — Warnings need greater prominence

Detection returns regions and an overall 81% confidence; the substrate row is 25%, its text is clipped, and Apply All is enabled. This is a visibility finding, not proof the numerical result must be rejected.

![Step 4: Detection and Apply All](D:/programacion/Menipy/audit-output/screenshots/04-detected.png)

### 5. Analysis and history — Completes; result inspection constrained

The native sessile run completed, added one record, and populated cards. The table row is vertically clipped in the default layout. The displayed 10 mm needle default was retained for workflow exercise only; these values are not a physical accuracy benchmark.

![Step 5: Analysis and history](D:/programacion/Menipy/audit-output/screenshots/05-results.png)

### 6. CSV export — Working; scope/provenance gaps

The top action successfully created native-export.csv in this audit folder and confirmed it in the status bar. That file was parsed, not merely inferred from the dialog. The separate table action has different scope.

![Step 6: CSV export](D:/programacion/Menipy/audit-output/screenshots/06-export-complete.png)

### 7. Advanced workflow — Visible controls; unavailable SOP backend

Advanced separates stage controls from the main workflow, a useful simplification. The SOP controls are shown although the startup probe confirms sops=None. Stage configuration is not a complete reusable preset.

![Step 7: Advanced workflow](D:/programacion/Menipy/audit-output/screenshots/07-advanced.png)

### 8. Folder mode — Ambiguous processing scope

Folder mode exposes folder and item selectors, with the same Run Analysis label. The isolated controller probe confirms one submission for a three-image folder. The previous image remains visible with an empty folder selector.

![Step 8: Folder mode](D:/programacion/Menipy/audit-output/screenshots/08-batch.png)

### 9. Dynamic, pendant and hardware coverage — Mixed coverage; explicit limits

Dynamic Sessile is present in core discovery/tests but its native selector is hidden. Pendant was exercised in direct pipeline benchmarks, not a full captured native journey. Temporal loading, analysis contracts, CLI exports and timeline are covered by existing tests. Camera hardware, full keyboard navigation and multi-DPI visual testing remain untested.

### Export picker detail

![Native CSV export picker before choosing the isolated output path](D:/programacion/Menipy/audit-output/screenshots/06-export.png)

## Customization inventory and restart checks

| Capability | Existing mechanism | Audit result |
| --- | --- | --- |
| Units / selected mode / remembered image | AppSettings JSON; main-window fallback | Real units JSON round-trip passed. Main-window CGS setting reset to SI after save/reload and separate-process restart. Selected source/mode are likewise routed through the fallback; full native restart capture not performed. |
| Overlays and markers | AppSettings dictionaries | Test dictionaries survived a process restart through the real service. Actual GUI application of every marker/overlay property was not exhaustively tested. |
| Result columns / compare / diagnostics | ResultsPanel uses real AppSettings | Existing column-visibility tests pass; separate-process hidden-column dictionary round-trip passed. |
| Layout | Splitters plus geometry/state saved by LayoutManager | Real splitter fields round-trip; main-window fallback prevents persistence. Geometry/state are dynamically assigned but absent from the dataclass fields; startup resize also merits correction. |
| Analysis settings | QSettings by pipeline | An actual AnalysisSettingsDialog persisted preprocessing target width 777; a second process read 777. |
| SOPs | SopService JSON | Service retains supplied params; controller-created preset reloads with params={}; applying a configured SOP only toggles stages. Native service is None. |
| Plugin preferences | SQLite + settings registries | A test plugin DB setting survived a process restart in an isolated copy. Per-plugin UI validation/activation round-trip was not exhaustively exercised. |
| Theme / GPU / threads | Static theme; unused SettingsDialog | No caller for SettingsDialog was discovered. Its advertised GPU/thread/theme controls should not be described as working customization. |

## Verification scenarios

| Scenario | Evidence / result |
| --- | --- |
| Startup service availability | Import probe + native startup: disabled runner/VM/SOP and fallback settings. |
| Rejected completion + pipeline switching | Synthetic worker in real QThreadPool + real RunViewModel: rejected context became accepted Pendant history row, missing original source. |
| Cancellation | 350 ms worker completed after Stop; 20 ms event-loop timer delayed to ~253 ms by a synchronous 250 ms pipeline. |
| Folder processing | Actual setup/controller with intercepted execution: 3 files, 1 submission. Acquisition preconditions supplied explicitly to isolate batch routing. |
| Missing calibration | Normal gate blocks absent regions; advanced builder supplies fallback scale and warning. |
| Preset and settings restart | SOP service reload plus separate-process JSON, QSettings and SQLite reads. |
| Persistence failure | PermissionError injection: no exception or error signal; row remains only in memory. |
| Static analysis / export | Native sessile journey reached history and CSV. Core sessile and pendant timed with explicit synthetic scale. |
| Temporal correctness | Existing Phase-D tests passed, including deterministic states, reacquisition, rejection, exports, CLI and timeline. |
| Accessibility | Native accessibility tree inspected; absent icon names observed. File picker keyboard entry worked. End-to-end keyboard/screen-reader and alternate-DPI audits remain untested. |

Focused tests: test_gui_startup_preview, test_guided_ui_simplification, test_pipeline_runner, test_phase_d_dynamic_sessile, test_results_panel_history, test_phase_a_results_gui, test_smoke_controller_flows. 69 passed, 1 skipped; the skipped font-family check depends on fonts unavailable to the offscreen Qt runtime. Passing existing tests does not disprove the reproduced integration gaps; several startup tests replace settings/history and do not assert actual service availability.

## Reproduce and interpret the evidence

Run from the repository root with uv. isolation.py redirects Path.home, QSettings and default plugin/material database constructors only inside each audit process. Databases are copied before use; the GUI service import/fallback wiring is deliberately untouched. Native launch uses launch.py; all its persistence is under audit-output/state/native. No application source, tests, scientific thresholds or public APIs were edited by the audit.

```powershell
$env:QT_QPA_PLATFORM="offscreen"
$env:UV_CACHE_DIR="D:\programacion\Menipy\.cache\uv"
uv run --no-sync python audit-output/probes.py
uv run --no-sync --extra test python audit-output/run_tests.py
uv run --no-sync python audit-output/memory_probe.py 30
uv run --no-sync python audit-output/memory_probe.py 300
uv run --no-sync python audit-output/supplemental.py write
uv run --no-sync python audit-output/supplemental.py read
uv run --no-sync python audit-output/startup_metrics.py 1
uv run --no-sync python audit-output/startup_metrics.py 2
uv run --no-sync python audit-output/startup_metrics.py 3
uv run --no-sync python audit-output/build_report.py
```

Audited baseline commit: 2755e58310db3ae283402f4218506e8e07695c60. The working tree was clean at the start of this execution. Separate edits to src/menipy/common/geometry.py and tests/test_sessile_contact_angles.py appeared after the main benchmark/test captures; they were preserved and recorded in concurrent-worktree.patch. The 69-pass result and numerical timings describe the earlier snapshot, not validation of those concurrent edits. Timing and screenshot measurements do not establish scientific accuracy. No application crash is inferred from the capture window later becoming unavailable.

Suggested sequence: A01+A02+A03 as one correctness/responsiveness release, with A04 and A06 as focused fixes; then A05/A07/A08/A09/A10/A13 for dependable workflows; then A11/A12 for measured speed/scaling work. A14 and A15 depend on product demand. Scientific contracts and conformance fixtures remain the acceptance baseline; any Context additions must be declared because extra fields are forbidden.

## Evidence files

- [backlog.csv](D:/programacion/Menipy/audit-output/backlog.csv)
- [backlog.json](D:/programacion/Menipy/audit-output/backlog.json)
- [evidence.json](D:/programacion/Menipy/audit-output/evidence.json)
- [tests.log](D:/programacion/Menipy/audit-output/tests.log)
- [tests.xml](D:/programacion/Menipy/audit-output/tests.xml)
- [native-startup.json](D:/programacion/Menipy/audit-output/native-startup.json)
- [export-summary.json](D:/programacion/Menipy/audit-output/export-summary.json)
- [native-export.csv](D:/programacion/Menipy/audit-output/native-export.csv)
- [memory-30.json](D:/programacion/Menipy/audit-output/memory-30.json)
- [memory-300.json](D:/programacion/Menipy/audit-output/memory-300.json)
- [supplemental-write.json](D:/programacion/Menipy/audit-output/supplemental-write.json)
- [supplemental-read.json](D:/programacion/Menipy/audit-output/supplemental-read.json)
- [startup-1.json](D:/programacion/Menipy/audit-output/startup-1.json)
- [startup-2.json](D:/programacion/Menipy/audit-output/startup-2.json)
- [startup-3.json](D:/programacion/Menipy/audit-output/startup-3.json)
- [concurrent-worktree.patch](D:/programacion/Menipy/audit-output/concurrent-worktree.patch)
- [probes.py](D:/programacion/Menipy/audit-output/probes.py)
- [isolation.py](D:/programacion/Menipy/audit-output/isolation.py)
- [run_tests.py](D:/programacion/Menipy/audit-output/run_tests.py)
- [memory_probe.py](D:/programacion/Menipy/audit-output/memory_probe.py)
- [supplemental.py](D:/programacion/Menipy/audit-output/supplemental.py)
- [startup_metrics.py](D:/programacion/Menipy/audit-output/startup_metrics.py)
- [launch.py](D:/programacion/Menipy/audit-output/launch.py)
