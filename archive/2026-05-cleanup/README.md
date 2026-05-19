# 2026-05 Cleanup Archive

This folder stores files moved during the archive-first cleanup on 2026-05-19.
No files were deleted.

## Why these files were moved

- `build/`: generated analysis artifacts that can be regenerated.
- `doc_audit/`: one-time documentation audit outputs.
- `mypy_report.txt`: point-in-time report output.
- `profile_run1.csv`: point-in-time profiling output.

## Restore

From repository root, run:

```bash
mv archive/2026-05-cleanup/build ./
mv archive/2026-05-cleanup/doc_audit ./
mv archive/2026-05-cleanup/mypy_report.txt ./
mv archive/2026-05-cleanup/profile_run1.csv ./
```
