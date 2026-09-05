"""Generate the audit report and ranked CSV backlog from captured evidence."""
import base64
import csv
import hashlib
import html
import json
import statistics
from pathlib import Path
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
e=json.loads((OUT/'evidence.json').read_text())
sw=json.loads((OUT/'supplemental-write.json').read_text())
sr=json.loads((OUT/'supplemental-read.json').read_text())
startups=[json.loads((OUT/f'startup-{i}.json').read_text()) for i in (1,2,3)]
mem=[json.loads((OUT/f'memory-{n}.json').read_text()) for n in (30,300)]
native=json.loads((OUT/'native-startup.json').read_text())

def source(path,needle):
    lines=(ROOT/path).read_text(encoding='utf-8').splitlines()
    line=next(i+1 for i,s in enumerate(lines) if needle in s)
    return f'{path}:{line}'

pc='src/menipy/gui/controllers/pipeline_controller.py'
mw='src/menipy/gui/views/main_window.py'
mc='src/menipy/gui/controllers/main_controller.py'
rs='src/menipy/models/results.py'
rp='src/menipy/gui/views/results_panel.py'
sc='src/menipy/gui/controllers/sop_controller.py'
cw='src/menipy/gui/dialogs/calibration_wizard_dialog.py'
items=[]
def item(id,priority,area,title,kind,evidence,impact,effort,dependencies,action,acceptance,refs):
    items.append(dict(id=id,priority=priority,area=area,title=title,classification=kind,
        confidence='High' if kind!='Exploratory product idea' else 'Medium', evidence=evidence,
        user_impact=impact,effort_engineer_days=effort,dependencies=dependencies,
        recommendation=action,acceptance_criteria=acceptance,source_refs='; '.join(refs)))

item('A01','P1','Functionality / speed / customization','Restore startup services without silent fallback','Confirmed current defect',
 'Normal import and native launch both yield runner=None, run_vm=None, sops=None. AppSettings comes from main_window. The import targets views.services, which is absent.',
 'The desktop silently loses background execution, SOP availability, and main-window persistence.', '1–2','A02 and A03 must accompany release',
 'Correct the service imports, narrow optional-feature exception handling, and report unavailable required services. Declare persisted layout fields and avoid overwriting restored geometry at startup.',
 'Unpatched MainWindow has real runner/view model/SOP/settings objects. Across a fresh process, selected pipeline, units, image, overlays and supported layout fields restore. Test startup service identities rather than patching them away.',
 [source(mw,'from .services.settings_service import AppSettings'),source('src/menipy/gui/services/settings_service.py','class AppSettings'),source('src/menipy/gui/app.py','w.resize(1200, 800)')])
item('A02','P1','Functionality','Bind completed results to the original job and validation context','Confirmed latent defect',
 'A real worker/view-model signal probe returned qa.ok=False and an original image path. The resulting history row was Accepted, had empty rejection reasons and no filename, and used the newly selected Pendant pipeline.',
 'Restoring the runner as-is exposes incorrect validation status and misattributed records.', '2–3','A01; release together',
 'Use one completion envelope with job identity, submitted pipeline/source/settings and full Context. Persist through build_persisted_analysis exactly once for every mode; do not infer identity from current controls.',
 'Rejected static and dynamic jobs retain rejection reasons/diagnostics with empty accepted metrics. Switching pipeline/source mid-run cannot change stored identity. Success, rejection, duplicate signals and out-of-order completion create exactly one correct row each.',
 [source(pc,'def on_context_ready'),source(pc,'def on_results_ready'),source(rp,'def update_single_measurement'),source(rs,'def build_persisted_analysis')])
item('A03','P1','Speed / UX','Keep computation off the UI thread and make Stop meaningful','Confirmed current and latent defects',
 f'A 20 ms Qt timer fired after {sw["timer_observed_ms"]:.1f} ms while a 250 ms synthetic direct pipeline ran. The worker probe still completed {e["probes"]["worker_rejection_switch_and_stop"]["completion_after_stop_ms"]:.1f} ms after Stop; Stop only clears queued tasks.',
 'Users cannot reliably interact with or stop long analysis; status can imply cancellation that never happened.', '3–5','A01, A02',
 'Route full, simple, stage-test and calibration computation through workers. Replace synchronous fallback with a submission error. Add job-scoped cancellation checks between stages/frames and callbacks inside long solvers where supported; report stopping while an uninterruptible operation finishes.',
 'A synthetic 500 ms job leaves a 20 ms GUI heartbeat running; no duplicate submissions. Stop cancels queued jobs and suppresses cancelled result publication, preserves completed history, and reports terminal cancellation once. Measure solver cancellation latency separately; never claim instantaneous interruption of native calls.',
 [source(pc,'def _run_pipeline_direct'),source(pc,'def test_stage'),source(mc,'def stop_pipeline'),source(cw,'def run_detection')])
item('A04','P1','Functionality / UX','Expose Dynamic Sessile in the real application','Confirmed current defect',
 'The controller has dynamicSessileBtn, but its parent is hidden. The toolbar reparents only five static mode buttons. Native startup shows those five buttons, although the core discovers and tests sessile_dynamic.',
 'A supported analysis capability is inaccessible through the primary workflow.', '0.5–1','Independent of A01; A03 for responsive execution',
 'Include Dynamic Sessile in the primary mode selector and expose its sequence/FPS controls. Add a visible-window integration test across every discovered supported mode.',
 'Keyboard and pointer users can select Dynamic Sessile without private APIs, choose video or frame directory, supply FPS where needed, and reach timeline results/export. Static mode controls remain correct.',
 [source(mw,'for button in ('),source(mw,'setup_ctrl.captiveBtn,'),source(mw,'for group_name in ("pipelineGroup", "sourceGroup")')])
item('A05','P1','Functionality / UX','Make folder processing explicit','Confirmed current behavior / missing capability',
 'With three images in a selected folder, gather_run_params provides the folder plus a selected image, but run_full submits exactly one image. Folder mode continues to label its action Run Analysis.',
 'Users may believe a folder was processed when only one file was analyzed.', '3–5','A02, A03',
 'Separate Run selected and Run folder; display file count, per-file outcomes, progress, retry failures and a consolidated export. Keep a temporal frame directory distinct from an independent-image batch.',
 'A three-file folder produces three identified outcomes; one corrupt file is reported without losing the others. Cancellation retains completed records. Dynamic sequence input remains one temporal result, not three unrelated analyses.',
 [source('src/menipy/gui/controllers/setup_panel_controller.py','def gather_run_params'),source(pc,'def run_full')])
item('A06','P1','Functionality / reliability','Report and recover from failed history writes','Confirmed current defect',
 'Injecting PermissionError during add_measurement returned normally and left a new in-memory row. _save_history catches every exception without logging or a user-visible error.',
 'A displayed measurement can disappear after restart without warning.', '1–2','Independent; coordinate with A02',
 'Use atomic persistence and a visible unsaved state with retry/export recovery. Preserve the last good file and keep measurements recoverable in memory.',
 'Read-only destination, simulated disk-full and interrupted write retain the prior file. Users see unsaved status; successful retry clears it. Restart never reads a partially written JSON file.',
 [source(rs,'def _save_history'),source(rs,'def add_measurement')])
item('A07','P2','Functionality / UX','Carry calibration provenance and warnings through every path','Confirmed path inconsistency',
 'Normal Run blocks absent ROI/needle regions. The advanced run builder instead supplied 10 px/mm with a fallback warning when no regions were present. The native wizard allowed Apply All with substrate confidence 25% and overall confidence 81%.',
 'Different entry points can lead to different confidence assumptions; aggregate confidence can hide a weak prerequisite.', '2–3','A02; scientific contract review',
 'Record scale origin (measured/manual/estimated/missing), show component warnings beside results and export them. Keep preview possible, but require explicit handling of estimated scale before publishing calibrated physical values. Do not change scientific rejection thresholds as part of UI cleanup.',
 'Missing or estimated scale is visible in stage tests and exports. Manual valid calibration is honored. A weak substrate remains visibly flagged even when overall confidence is high. Existing numerical conformance gates continue to pass.',
 [source(pc,'def _build_pipeline_run_kwargs'),source(pc,'def _collect_acquisition_inputs'),source(cw,'def _run_best_auto_calibration')])
item('A08','P2','Customization','Turn SOPs into complete reusable analysis presets','Confirmed latent gap',
 'SopController creates presets with params={}; applying them only sets stage inclusion. SopService can persist parameter dictionaries, but normal startup currently disables the service.',
 'Researchers cannot reliably repeat or share a full configured workflow using SOPs.', '3–5','A01, A02',
 'Capture typed preprocessing, detection, physics, geometry, calibration choices, plugin selections and stage inclusion. Support named save/update/import/export with versioning and a summary before applying.',
 'After process restart, a preset round-trips all supported values and reproduces run configuration. Missing plugin/version conflicts are explained. Applying a preset does not silently retain incompatible settings from the previous mode.',
 [source(sc,'def on_add_sop'),source(sc,'def _apply_selected_sop'),source('src/menipy/gui/services/sop_service.py','class Sop')])
item('A09','P2','UX / accessibility','Name icon controls and guide source readiness','Confirmed UI/API evidence',
 'The native accessibility tree reports blank names for inactive mode buttons, source toggles and panel toggles. Startup enables Run Analysis with no image; switching to an empty folder leaves the old preview/results visible.',
 'Controls are hard to discover without hovering, and the current source can be confused with a previous result.', '1–3','Coordinate with A04, A05',
 'Set explicit accessible names and label associations, offer persistent mode labels, show source readiness and explain prerequisites next to Run. Clearly mark previous results when source selection changes.',
 'Keyboard-only selection, loading, calibration, run and export have visible focus and meaningful names. Empty-source Run is disabled with a reason or routes to source selection. Source changes invalidate stale calibration or visibly associate it with its source.',
 [source(mw,'def _install_workflow_setup_controls'),source('src/menipy/gui/controllers/setup_panel_controller.py','def _sync_pipeline_button_presentation')])
item('A10','P2','UX','Keep calibration status and result rows readable','Confirmed visual issue',
 'At the captured native size, calibration opens zoomed so the drop base is out of view; detected-region status text is clipped beside Draw buttons. After analysis the results table shows a header and only a sliver of the selected row.',
 'Users must adjust layout before verifying detections or reading the measurement table.', '1–2','Independent; retain existing layout presets',
 'Fit the complete image on initial calibration display, allocate enough width for status text, and reserve usable table height when results exist. Keep splitter controls and saved layouts.',
 'At 1200×800 and 1366×768 logical layouts, and 100/125/150% display scaling, the first result row and all confidence text remain readable. Window resizing preserves image aspect ratio and offers access to every action.',
 [source(cw,'class CalibrationWizardDialog'),source(mw,'self.inspectTabs.setMinimumHeight(190)')])
item('A11','P2','Speed / scale','Bound decoded video and detection memory','Measured scaling cost',
 'Separate-process 640×480 MJPG probes retained 26.37 MiB for 30 frames and 263.67 MiB for 300 frames. Process peak working set rose from 171.08 to 408.58 MiB. Temporal analysis additionally builds a detection list for all frames.',
 'Long recordings can exhaust memory before results are available.', '5–10','A03; temporal contract and scientific conformance tests',
 'Introduce incremental decoding with bounded frame/detection buffers or a disk-backed sequence store, and retain lightweight result series. Preserve timing, ordering, scale initialization, reacquisition and classification semantics.',
 'A 10× longer clip does not retain 10× decoded pixels. Outputs match existing fixtures within established tolerances, including first-five scale samples, occlusions, rejection, timestamps and export order. Cancellation releases decoder resources.',
 [source('src/menipy/common/sequence_acquisition.py','def load_video'),source('src/menipy/common/temporal_sessile.py','detections = [auto_detect_features')])
item('A12','P2','Speed','Profile fitting and startup imports before broad optimization','Measured hotspot / investigation',
 'Sampled warm pipelines took 1.48 s sessile and 1.90 s pendant; fitting dominated their last-run stage time (~93% and ~92%). One pendant first call took 9.02 s. Fresh-process startup samples ranged 3.18–6.64 s and were dominated by import/Qt setup.',
 'These are the most defensible targets for reducing waiting in the sampled workflows.', '2–4 investigation; implementation sized after profile','A03 first; preserve numerical contracts',
 'Profile solver evaluations, initialization and repeated model work on controlled fixtures. Separate first-use costs from warm execution; inspect eager imports and optional initialization. Keep existing preview caching, which already makes repeat loads sub-millisecond in this probe.',
 'Record distributions on fixed inputs and environment with numerical equivalence. Accept optimizations only with unchanged rejection/fit behavior and demonstrable timing improvement; no claimed speedup based on these baseline samples alone.',
 [source('src/menipy/pipelines/base.py','def _call_stage'),source('src/menipy/pipelines/discover.py','PIPELINE_MAP ='),source('src/menipy/gui/views/preview_panel.py','def load_path')])
item('A13','P2','Functionality / UX','Clarify export scope and preserve machine-readable provenance','Confirmed source and output evidence',
 'Top Export CSV writes history, while the table Export writes visible cells/columns. The actual native export contains a time-of-day timestamp without its date and lacks schema_version and explicit calibration scale columns.',
 'Users can export a different scope than expected; cross-day analysis and reproducibility lose context.', '1–2','A02; result contracts',
 'Label Export all history and Export current view distinctly. Add a canonical machine export with full ISO timestamp, source identity, schema version, calibration and settings/plugin provenance; keep human-readable view export.',
 'Both actions explain scope and preserve filters as labeled. Machine export round-trips dates, units, validation and original sources; hidden view columns cannot silently remove mandatory provenance from the machine export.',
 [source(mc,'def export_results_csv'),source(rp,'def _export_csv'),source(rs,'measurement.timestamp.strftime("%H:%M:%S")')])
item('A14','P3','Speed / customization','Improve history retention deliberately before scaling the table','Measured future scaling cost',
 'Default history retains 100 records. Warm table refresh was 76 ms at 100 and 633 ms at an artificially populated 1,000 records; save was ~10 and 49 ms respectively. The 1,000-row case exceeds the default retention cap.',
 'Larger projects would make full rebuilding noticeable; current users first need to understand the 100-record retention limit.', '2–3','A06; A05 if batch increases history volume',
 'Expose retention/export behavior, then consider project-backed history and an incremental Qt table model if larger histories are required. Avoid prioritizing a database rewrite solely from the 1,000-row stress case.',
 'Retention is visible and configurable with deliberate archival behavior. Default-size refresh remains responsive; larger-history targets are benchmarked before committing to a storage migration.',
 [source(rs,'def __init__(self, max_history: int = 100)'),source(rp,'def update_history')])
item('A15','P3','Customization / UX','Offer a coherent appearance and workspace preference surface','Exploratory product idea',
 'Units, overlay/marker dictionaries, result columns, splitters, analysis QSettings and plugin DB settings have existing persistence mechanisms. The unused SettingsDialog advertises GPU/thread/theme options but has no discovered caller; it is not evidence those controls are available in the product.',
 'A single preference surface could make existing customization understandable, with font/density/theme choices added only when backed by real behavior.', '2–4','A01; validate demand before adding theme/GPU controls',
 'Consolidate existing preferences and add reset/export/import. Consider readable font size, density and light/dark/system appearance. Do not expose GPU or parallel switches until supported and measured.',
 'Every visible preference loads, applies and survives restart, has a reset path and accurately describes scope. Unsupported controls are absent or clearly disabled; no cosmetic setting claims performance gains.',
 [source('src/menipy/gui/services/settings_service.py','class AppSettings'),source('src/menipy/gui/dialogs/settings_dialog.py','class SettingsDialog')])

steps=[
 ('1','Startup','Needs improvement','01-start.png','A strong central preview and clear Calibrate/Run actions are present. Run is enabled with no source; unlabeled icon controls and the missing Dynamic option reduce discoverability. Native UIA names were inspected; full screen-reader navigation was not completed.'),
 ('2','Source selection','Working, with clarity gaps','02-source.png','The native file picker loaded the local 1600×1200 sessile reference successfully. The preview fits the image; path display is abbreviated. Returning to an empty folder later leaves the previous analysis visible (step 8).'),
 ('3','Calibration entry','Needs layout improvement','03-calibration.png','Manual drawing and Fit to Window are available. Initial zoom hides the bottom of the drop in this dialog, making substrate review harder until the user fits or scrolls.'),
 ('4','Detection and Apply All','Warnings need greater prominence','04-detected.png','Detection returns regions and an overall 81% confidence; the substrate row is 25%, its text is clipped, and Apply All is enabled. This is a visibility finding, not proof the numerical result must be rejected.'),
 ('5','Analysis and history','Completes; result inspection constrained','05-results.png','The native sessile run completed, added one record, and populated cards. The table row is vertically clipped in the default layout. The displayed 10 mm needle default was retained for workflow exercise only; these values are not a physical accuracy benchmark.'),
 ('6','CSV export','Working; scope/provenance gaps','06-export-complete.png','The top action successfully created native-export.csv in this audit folder and confirmed it in the status bar. That file was parsed, not merely inferred from the dialog. The separate table action has different scope.'),
 ('7','Advanced workflow','Visible controls; unavailable SOP backend','07-advanced.png','Advanced separates stage controls from the main workflow, a useful simplification. The SOP controls are shown although the startup probe confirms sops=None. Stage configuration is not a complete reusable preset.'),
 ('8','Folder mode','Ambiguous processing scope','08-batch.png','Folder mode exposes folder and item selectors, with the same Run Analysis label. The isolated controller probe confirms one submission for a three-image folder. The previous image remains visible with an empty folder selector.'),
 ('9','Dynamic, pendant and hardware coverage','Mixed coverage; explicit limits',None,'Dynamic Sessile is present in core discovery/tests but its native selector is hidden. Pendant was exercised in direct pipeline benchmarks, not a full captured native journey. Temporal loading, analysis contracts, CLI exports and timeline are covered by existing tests. Camera hardware, full keyboard navigation and multi-DPI visual testing remain untested.')]

metrics=[]
metrics.append(['Startup, fresh process (offscreen)','3 processes',f'{startups[0]["startup_to_event_loop_ms"]:.0f} ms',
                f'{statistics.median(x["startup_to_event_loop_ms"] for x in startups[1:]):.0f} ms subsequent-process median','3.18–6.64 s range; OS caches not flushed'])
metrics.append(['Startup, native','1 launch',f'{native["startup_to_event_loop_s"]*1000:.0f} ms','—','Includes isolation and widget creation, excludes uv startup'])
for mode in ('sessile','pendant'):
    b=e['benchmarks'][mode]
    for operation in ('preview','calibration','pipeline'):
        t=b[operation]
        metrics.append([f'{mode.title()} {operation}',f'{b["shape"][1]}×{b["shape"][0]}, n=3',f'{t["first_ms"]:.2f} ms',f'{t["warm_median_ms"]:.2f} ms','Preview warm sample is same-path caching' if operation=='preview' else 'Calibration = core detector, not entire wizard' if operation=='calibration' else 'Explicit synthetic scale; numerical accuracy not assessed'])
for count in (10,100,1000):
    b=e['benchmarks'][f'history_{count}']
    for operation in ('save','table_refresh'):
        t=b[operation]
        metrics.append([f'History {operation}',f'{count} rows, n=3',f'{t["first_ms"]:.2f} ms',f'{t["warm_median_ms"]:.2f} ms','Above default 100-row cap' if count==1000 else 'Small scalar result fixture; large diagnostic payloads not benchmarked'])

with (OUT/'backlog.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
(OUT/'backlog.json').write_text(json.dumps(items,indent=2,ensure_ascii=False),encoding='utf-8')

with (OUT/'native-export.csv').open(newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f); rows=list(reader); export_summary={'row_count':len(rows),'column_count':len(reader.fieldnames),
        'has_schema_version':'schema_version' in reader.fieldnames,'has_explicit_px_per_mm':'px_per_mm' in reader.fieldnames,
        'timestamp':rows[0]['timestamp'],'status':rows[0]['status'],'filename':rows[0]['file_name']}
(OUT/'export-summary.json').write_text(json.dumps(export_summary,indent=2))

md=[]; ht=[]
def heading(text,level=2):
    md.append('#'*level+' '+text+'\n'); ht.append(f'<h{level}>{html.escape(text)}</h{level}>')
def para(text):
    md.append(text+'\n');ht.append('<p>'+html.escape(text)+'</p>')
def table(headers,rows):
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in r)+' |' for r in rows)+'\n')
    ht.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>')
def picture(filename,caption):
    path=OUT/'screenshots'/filename
    md.append(f'![{caption}]({path.as_posix()})\n')
    data=base64.b64encode(path.read_bytes()).decode()
    ht.append(f'<figure><img loading="lazy" src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')

heading('Menipy improvement audit — 5 September 2026',1)
para('The highest-value first release is reliable startup, validated job completion and responsive execution. Several capabilities already exist in the code but are disconnected from the native application. Restore those paths before expanding features; restoring the worker alone would expose a separately reproduced result-integrity defect.')
para('Deliverables: 15 ranked recommendations, 9 accepted native captures (8 workflow steps plus export dialog), runtime evidence, reproducible probes and a CSV backlog. Existing focused checks: 69 passed, 1 skipped. No application or public API changes were made by this audit.')
heading('What changed from the initial assessment')
table(['Initial hypothesis','Audit conclusion'],[
 ['Some GUI routes are synchronous','Stronger: incorrect startup imports disable the normal runner entirely. The main-window fallback is the current native path.'],
 ['Background results lose validation metadata','Reproduced through a real worker/view-model signal route, but latent in the current native startup until A01 is repaired.'],
 ['Missing calibration is silently estimated','Normal acquisition blocks absent ROI/needle. Advanced configuration can estimate scale and returns a warning; inconsistent treatment/provenance is the confirmed issue.'],
 ['Large history is a present scalability bottleneck','Default retention is 100. The 1,000-row experiment is a stress scenario, not default operating behavior.'],
 ['Customization needs new storage','Real JSON/QSettings/SQLite services preserve the tested preferences across restart. Wiring and completeness are the immediate gaps.'],
 ['Dynamic video features need adding','Core and test coverage exist; the Dynamic button is hidden by the main-window composition.']])
heading('Ranked backlog')
para('P1: correctness or core workflow; P2: substantial usability/scaling improvement; P3: follow-on investment. Effort is a rough engineering-day estimate including focused tests, not a delivery commitment. Dependencies identify changes that should be designed or released together.')
table(['ID','Priority','Area','Opportunity','Evidence class','Effort (days)'],[[x['id'],x['priority'],x['area'],x['title'],x['classification'],x['effort_engineer_days']] for x in items])
for x in items:
    heading(f'{x["id"]} · {x["title"]}',3)
    para(f'{x["priority"]} | {x["classification"]} | Confidence: {x["confidence"]} | Effort: {x["effort_engineer_days"]} days | Dependencies: {x["dependencies"]}.')
    para('Evidence: '+x['evidence']);para('User impact: '+x['user_impact']);para('Recommendation: '+x['recommendation']);para('Acceptance: '+x['acceptance_criteria'])
    refs=x['source_refs'].split('; ')
    md.append('Source: '+', '.join(f'[{r}]({(ROOT/r.rsplit(":",1)[0]).as_posix()}:{r.rsplit(":",1)[1]})' for r in refs)+'\n')
    ht.append('<p class="source">Source: '+html.escape(x['source_refs'])+'</p>')
heading('Measured performance')
para('Host: Windows 11, 12 logical CPUs (AMD64 Family 23 Model 96), Python 3.14.7, Qt 6.11.2, OpenCV 5.0.0, NumPy 2.5.2. Core probes ran offscreen; native screenshots used the Windows desktop. First-call and warm values are separate: warm is the median of calls 2–3. No OS cache flush or cold-boot benchmark was performed. These small samples are observations, not statistically stable p95 targets; other desktop applications were running and the early probe phase overlapped the focused tests.')
table(['Operation','Input / repetitions','First call','Warm / subsequent','Interpretation'],metrics)
table(['Decode input','Time','Retained image pixels','Working set before → after','Process peak'],[[f'{m["frames"]} frames, 640×480 RGB, 30 fps MJPG',f'{m["elapsed_ms"]:.1f} ms',f'{m["retained_pixel_mib"]:.2f} MiB',f'{m["before"]["working_set_mib"]:.1f} → {m["after"]["working_set_mib"]:.1f} MiB',f'{m["after"]["process_peak_working_set_mib"]:.1f} MiB'] for m in mem])
para('Memory samples use separate processes and real MJPG decoding. Peak working set includes imports and fixture encoding; retained pixel bytes are summed from decoded arrays. Full-resolution long-video analysis and GPU performance were not measured. Stage-level raw timings and sample arrays are in evidence.json. The cancellation probe measures completion after an ineffective Stop, not successful cancellation latency.')
heading('Native workflow evidence')
para('All displayed images below are current-run captures saved and inspected without editing. The first occluded capture was rejected and replaced. The tool later stopped exposing the audit window after folder-mode capture; no later error screenshot or crash claim is included. Source fixture attribution remains in data/MANIFEST.json and data/ATTRIBUTION.md; no external ground-truth claim was independently verified.')
table(['Step','Flow','Health'],[[a,b,c] for a,b,c,_,_ in steps])
for number,title,health,filename,notes in steps:
    heading(f'{number}. {title} — {health}',3);para(notes)
    if filename: picture(filename,f'Step {number}: {title}')
heading('Export picker detail',3);picture('06-export.png','Native CSV export picker before choosing the isolated output path')
heading('Customization inventory and restart checks')
table(['Capability','Existing mechanism','Audit result'],[
 ['Units / selected mode / remembered image','AppSettings JSON; main-window fallback','Real units JSON round-trip passed. Main-window CGS setting reset to SI after save/reload and separate-process restart. Selected source/mode are likewise routed through the fallback; full native restart capture not performed.'],
 ['Overlays and markers','AppSettings dictionaries','Test dictionaries survived a process restart through the real service. Actual GUI application of every marker/overlay property was not exhaustively tested.'],
 ['Result columns / compare / diagnostics','ResultsPanel uses real AppSettings','Existing column-visibility tests pass; separate-process hidden-column dictionary round-trip passed.'],
 ['Layout','Splitters plus geometry/state saved by LayoutManager','Real splitter fields round-trip; main-window fallback prevents persistence. Geometry/state are dynamically assigned but absent from the dataclass fields; startup resize also merits correction.'],
 ['Analysis settings','QSettings by pipeline','An actual AnalysisSettingsDialog persisted preprocessing target width 777; a second process read 777.'],
 ['SOPs','SopService JSON','Service retains supplied params; controller-created preset reloads with params={}; applying a configured SOP only toggles stages. Native service is None.'],
 ['Plugin preferences','SQLite + settings registries','A test plugin DB setting survived a process restart in an isolated copy. Per-plugin UI validation/activation round-trip was not exhaustively exercised.'],
 ['Theme / GPU / threads','Static theme; unused SettingsDialog','No caller for SettingsDialog was discovered. Its advertised GPU/thread/theme controls should not be described as working customization.']])
heading('Verification scenarios')
table(['Scenario','Evidence / result'],[
 ['Startup service availability','Import probe + native startup: disabled runner/VM/SOP and fallback settings.'],
 ['Rejected completion + pipeline switching','Synthetic worker in real QThreadPool + real RunViewModel: rejected context became accepted Pendant history row, missing original source.'],
 ['Cancellation','350 ms worker completed after Stop; 20 ms event-loop timer delayed to ~253 ms by a synchronous 250 ms pipeline.'],
 ['Folder processing','Actual setup/controller with intercepted execution: 3 files, 1 submission. Acquisition preconditions supplied explicitly to isolate batch routing.'],
 ['Missing calibration','Normal gate blocks absent regions; advanced builder supplies fallback scale and warning.'],
 ['Preset and settings restart','SOP service reload plus separate-process JSON, QSettings and SQLite reads.'],
 ['Persistence failure','PermissionError injection: no exception or error signal; row remains only in memory.'],
 ['Static analysis / export','Native sessile journey reached history and CSV. Core sessile and pendant timed with explicit synthetic scale.'],
 ['Temporal correctness','Existing Phase-D tests passed, including deterministic states, reacquisition, rejection, exports, CLI and timeline.'],
 ['Accessibility','Native accessibility tree inspected; absent icon names observed. File picker keyboard entry worked. End-to-end keyboard/screen-reader and alternate-DPI audits remain untested.']])
testtree=ET.parse(OUT/'tests.xml')
skips=[(tc.attrib.get('name'),tc.find('skipped').attrib.get('message')) for tc in testtree.iter('testcase') if tc.find('skipped') is not None]
para('Focused tests: '+', '.join(['test_gui_startup_preview','test_guided_ui_simplification','test_pipeline_runner','test_phase_d_dynamic_sessile','test_results_panel_history','test_phase_a_results_gui','test_smoke_controller_flows'])+'. 69 passed, 1 skipped; the skipped font-family check depends on fonts unavailable to the offscreen Qt runtime. Passing existing tests does not disprove the reproduced integration gaps; several startup tests replace settings/history and do not assert actual service availability.')
heading('Reproduce and interpret the evidence')
para('Run from the repository root with uv. isolation.py redirects Path.home, QSettings and default plugin/material database constructors only inside each audit process. Databases are copied before use; the GUI service import/fallback wiring is deliberately untouched. Native launch uses launch.py; all its persistence is under audit-output/state/native. No application source, tests, scientific thresholds or public APIs were edited by the audit.')
commands=['$env:QT_QPA_PLATFORM="offscreen"', '$env:UV_CACHE_DIR="D:\\programacion\\Menipy\\.cache\\uv"',
 'uv run --no-sync python audit-output/probes.py', 'uv run --no-sync --extra test python audit-output/run_tests.py',
 'uv run --no-sync python audit-output/memory_probe.py 30', 'uv run --no-sync python audit-output/memory_probe.py 300',
 'uv run --no-sync python audit-output/supplemental.py write','uv run --no-sync python audit-output/supplemental.py read',
 'uv run --no-sync python audit-output/startup_metrics.py 1', 'uv run --no-sync python audit-output/startup_metrics.py 2',
 'uv run --no-sync python audit-output/startup_metrics.py 3','uv run --no-sync python audit-output/build_report.py']
md.append('```powershell\n'+'\n'.join(commands)+'\n```\n');ht.append('<pre>'+html.escape('\n'.join(commands))+'</pre>')
para('Audited baseline commit: 2755e58310db3ae283402f4218506e8e07695c60. The working tree was clean at the start of this execution. Separate edits to src/menipy/common/geometry.py and tests/test_sessile_contact_angles.py appeared after the main benchmark/test captures; they were preserved and recorded in concurrent-worktree.patch. The 69-pass result and numerical timings describe the earlier snapshot, not validation of those concurrent edits. Timing and screenshot measurements do not establish scientific accuracy. No application crash is inferred from the capture window later becoming unavailable.')
para('Suggested sequence: A01+A02+A03 as one correctness/responsiveness release, with A04 and A06 as focused fixes; then A05/A07/A08/A09/A10/A13 for dependable workflows; then A11/A12 for measured speed/scaling work. A14 and A15 depend on product demand. Scientific contracts and conformance fixtures remain the acceptance baseline; any Context additions must be declared because extra fields are forbidden.')
heading('Evidence files')
files=['backlog.csv','backlog.json','evidence.json','tests.log','tests.xml','native-startup.json','export-summary.json',
       'native-export.csv','memory-30.json','memory-300.json','supplemental-write.json','supplemental-read.json',
       'startup-1.json','startup-2.json','startup-3.json','concurrent-worktree.patch','probes.py','isolation.py',
       'run_tests.py','memory_probe.py','supplemental.py','startup_metrics.py','launch.py']
md.extend(f'- [{name}]({(OUT/name).as_posix()})' for name in files)
ht.append('<ul>'+''.join(f'<li>{html.escape(name)}</li>' for name in files)+'</ul>')
(OUT/'REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
style='''body{font:16px/1.55 system-ui,sans-serif;color:#26323e;background:#f4f6f8;margin:0}main{max-width:1150px;margin:32px auto;padding:36px;background:white}h1{font-size:34px;line-height:1.15;color:#153b57}h2{margin-top:40px;border-bottom:2px solid #dce5ec;padding-bottom:8px}h3{margin-top:30px;color:#214d6b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;padding:10px;border:1px solid #dbe3e9}th{background:#eaf0f5}tr:nth-child(even){background:#f8fafb}.table-wrap{overflow-x:auto}figure{margin:24px 0}img{max-width:100%;height:auto;border:1px solid #ddd}figcaption,.source{font-size:13px;color:#556674}pre{overflow:auto;background:#eef2f5;padding:16px;font-size:13px}p{max-width:100ch}@media print{body{background:white}main{margin:0;padding:0}h2,h3{break-after:avoid}figure{break-inside:avoid}table{font-size:10px}}'''
(OUT/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Menipy improvement audit</title><style>'+style+'</style><main>'+''.join(ht)+'</main></html>',encoding='utf-8')
manifest={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('screenshots/*.png')}
manifest['baseline_commit']='2755e58310db3ae283402f4218506e8e07695c60'
(OUT/'capture-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'report':str(OUT/'REPORT.html'),'findings':len(items),'screenshots':len(manifest)-1,'skip_details':skips},indent=2))
