# Removal recommendations for menipy (conservative)

This report lists files inside `src/menipy` that have no incoming static imports from other `menipy` modules, with runtime importability flags and conservative recommendations.

## src/menipy/analysis/drop.py

- runtime_importable: **True**
- static imports: 3
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/batch.py

- runtime_importable: **True**
- static imports: 2
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/cli.py

- runtime_importable: **True**
- static imports: 0
- imported_by (static): []

**Notes & confidence**
- Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/contact_angle.py

- runtime_importable: **True**
- static imports: 0
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/detection/__init__.py

- runtime_importable: **True**
- static imports: 0
- imported_by (static): []

**Notes & confidence**
- Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/detection/droplet.py

- runtime_importable: **True**
- static imports: 1
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/detectors/__init__.py

- runtime_importable: **True**
- static imports: 1
- imported_by (static): []

**Notes & confidence**
- Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/gui.py

- runtime_importable: **True**
- static imports: 1
- imported_by (static): []

**Notes & confidence**
- Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/gui/__init__.py

- runtime_importable: **True**
- static imports: 6
- imported_by (static): []

**Notes & confidence**
- Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/metrics/__init__.py

- runtime_importable: **True**
- static imports: 0
- imported_by (static): []

**Notes & confidence**
- Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/metrics/metrics.py

- runtime_importable: **True**
- static imports: 0
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/physics/__init__.py

- runtime_importable: **True**
- static imports: 0
- imported_by (static): []

**Notes & confidence**
- Likely an entrypoint or package initializer; keep unless you are refactoring entrypoints.
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/plugins.py

- runtime_importable: **True**
- static imports: 1
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/processing/metrics.py

- runtime_importable: **True**
- static imports: 2
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---

## src/menipy/utils.py

- runtime_importable: **True**
- static imports: 0
- imported_by (static): []

**Notes & confidence**
- Imports at runtime — may be used externally (CLI, scripts, tests).

- Recommendation: KEEP or MANUALLY REVIEW before removal (LOW confidence).

---
