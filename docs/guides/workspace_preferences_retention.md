# A14–A15: active history and workspace preferences

## History retention and archives

Config → Workspace Preferences exposes the active-history limit (10–1,000,
default 100). The result count also displays the limit. A changed limit applies
on the next measurement; changing it does not immediately remove records.

Before records leave active history, `ResultsHistory` writes a complete atomic
JSON snapshot under `~/.menipy/history_archive/`. Files contain the same
`measurements` envelope as history/recovery files, including validation,
diagnostics, full timestamps and run metadata. Content-derived filenames avoid
creating another file for an identical archived group. Recovery/retries can
produce overlapping snapshots; consumers should deduplicate by measurement ID.
No archive is automatically deleted or loaded into active history at startup.

The results-panel **Archive all** action replaces Clear: it archives current
records before clearing the active list. **Open history archives** in preferences
opens the directory. **Export all history** continues to export active records
only; archived snapshots are separate JSON files. There is no archive-browser or
automatic restore feature in this release.

An archive failure prevents removal, keeps records in memory and shows the
existing unsaved-history notice. Retry writes every retained in-memory record
back to active history; the next insertion retries retention. If the active
history write fails after archiving, the archive still preserves removed rows.
Normal recovery export and close protection remain available. Large or
temporarily unsaved histories may exceed the configured limit until a successful
subsequent insertion.

## Supported preferences

The new SettingsDialog is available through **Config → Workspace Preferences**.
It uses the window's shared AppSettings instance. Supported draft preferences:

- SI/CGS display units;
- persistent analysis-mode labels;
- comparison/diagnostic column defaults (explicit per-view column choices take
  precedence);
- active-history retention limit;
- existing result-column visibility, carried through JSON import/export.

Reset draft to defaults resets this scope, including custom result columns.
Import and reset change only the draft. Apply saves atomically before updating
the shared settings and the UI. Closing without Apply discards the draft.
Export draft writes the current draft, including unapplied changes. Imports
validate version, ranges, types and unknown fields before changing the draft.
A failed settings replacement leaves the last settings file and live settings
intact.

Overlay appearance and marker/label editors are linked from the same dialog;
they retain their separate Save behavior and existing persistence. They are not
included in this preferences JSON. Analysis presets, calibration, source paths,
plugin settings and window geometry remain in their existing scopes. The dialog
does not advertise theme, density, GPU or thread controls. Those require working
implementations and separate validation before being exposed.

## Performance evidence

`tools/profile_history.py` uses isolated settings/databases, a real offscreen
window, and 100/1,000 synthetic static records with five scalar metrics. It
measures table refresh plus event processing and atomic save separately. One
first run and three warm runs were captured on Windows 11 build 26200,
Python 3.14.7. These records do not represent large temporal diagnostics.

| Records | Previous warm refresh median | Cell-reuse warm median | Initial build after change | Warm save median |
| --- | --- | --- | --- | --- |
| 100 | 146 ms | 40 ms | 146 ms | 5 ms |
| 1,000 | 1,249 ms | 192 ms | 1,219 ms | 16 ms |

The panel now retains unchanged QTableWidgetItems when the column schema and
pipeline style match. Changed values and unit headers still update; changing
pipeline/schema rebuilds cells. This is a bounded improvement to the existing
table, not a Qt model or storage migration. The initial 1,000-row build remains
noticeable and the preference text warns about larger histories. A model-backed
view/pagination remains future work if real project volume justifies it.

Raw evidence:
`.cache/history-profile/0dbf5a1cdcbf445a9c63c81620f6271b/report.json` (before),
`.cache/history-profile/a89547bccc9140769eb75adc87e3b7df/report.json` (after).

```powershell
$env:UV_CACHE_DIR='D:\uv_cache_dir'
uv run --offline --extra test python tools/profile_history.py
uv run --offline --extra test python tools/check_gui_execution.py tests/test_workspace_preferences.py tests/test_history_recovery_calibration.py tests/test_results_panel_history.py tests/test_gui_execution_integration.py
```

Tests cover archival before eviction/clear, archival failure and recovery,
transactional preference failure, strict import validation, draft reset,
shared settings identity, immediate UI application and fresh-process reload.
