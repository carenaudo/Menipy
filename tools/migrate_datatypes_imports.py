"""
Scan and optionally rewrite imports from `menipy.models.datatypes` to new module locations.

Usage:
    # dry-run (default)
    python tools/migrate_datatypes_imports.py

    # apply changes
    python tools/migrate_datatypes_imports.py --apply

    # provide an explicit mapping file (JSON) for ambiguous names
    python tools/migrate_datatypes_imports.py --mapping mymap.json --apply

Notes:
- The script performs best-effort rewrites using a built-in mapping. For names not in the mapping
  it will report them and refuse to apply changes unless an explicit mapping is supplied.
- The script does not edit files unless --apply is provided. Always review the dry-run output first.

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".venv", "venv", "build", "dist", ".git", "__pycache__"}

# Conservative default mapping based on the refactor plan.
# If a name is missing from this mapping the script will complain and stop (unless a mapping file is provided).
DEFAULT_MAPPING: Dict[str, str] = {
    # frame-related
    "Frame": "menipy.models.frame",
    "CameraMeta": "menipy.models.frame",
    "Calibration": "menipy.models.frame",
    # geometry
    "Contour": "menipy.models.geometry",
    "Geometry": "menipy.models.geometry",
    "Point": "menipy.models.geometry",
    "ROI": "menipy.models.geometry",
    "Needle": "menipy.models.geometry",
    "ContactLine": "menipy.models.geometry",
    # config
    "PhysicsParams": "menipy.models.config",
    "ResizeSettings": "menipy.models.config",
    "FilterSettings": "menipy.models.config",
    "BackgroundSettings": "menipy.models.config",
    "NormalizationSettings": "menipy.models.config",
    "ContactLineSettings": "menipy.models.config",
    "PreprocessingSettings": "menipy.models.config",
    "EdgeDetectionSettings": "menipy.models.config",
    # fit / result
    "FitConfig": "menipy.models.fit",
    "SolverInfo": "menipy.models.fit",
    "Residuals": "menipy.models.fit",
    "Confidence": "menipy.models.fit",
    "YoungLaplaceFit": "menipy.models.result",
    "OscillationFit": "menipy.models.result",
    "CapillaryRiseFit": "menipy.models.result",
    # other datatypes that may remain ambiguous — user must supply mapping to rewrite
    "PreprocessingState": "menipy.models.state",
    "MarkerSet": "menipy.models.state",
    "PreprocessingStageRecord": "menipy.models.state",
    "AnalysisRecord": "menipy.models.result",
}

IMPORT_RE = re.compile(r"from\s+menipy\.models\.datatypes\s+import\s+(.+)")


def collect_multiline_imports(text: str) -> List[Tuple[int, str]]:
    """Return list of (start_line_index, import_block_text) for multiline imports.

    We look for lines like: from menipy.models.datatypes import (
    and gather until the closing ).
    """
    lines = text.splitlines()
    blocks: List[Tuple[int, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("from menipy.models.datatypes import ("):
            start = i
            block_lines = [line]
            i += 1
            # gather until a line containing ')'
            while i < len(lines) and ")" not in lines[i]:
                block_lines.append(lines[i])
                i += 1
            if i < len(lines):
                block_lines.append(lines[i])
            blocks.append((start, "\n".join(block_lines)))
        i += 1
    return blocks


def find_py_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        files.append(p)
    return files


def parse_imports(line: str) -> Optional[List[str]]:
    m = IMPORT_RE.search(line)
    if not m:
        return None
    names = m.group(1)
    # ignore the opening of a multiline import block: "import ("
    if names.strip().startswith("("):
        return None
    # Remove parentheses and split by comma
    names = names.strip()
    if names.startswith("(") and names.endswith(")"):
        names = names[1:-1]
    parts = [n.strip() for n in names.split(",") if n.strip()]
    return parts


def plan_rewrites(
    files: List[Path], mapping: Dict[str, str]
) -> Tuple[Dict[Path, List[Tuple[str, str, str]]], List[Tuple[Path, str]]]:
    """
    Scan files and plan rewrites.
    Returns a dict of file -> list of tuples(original_line, old_import_statement, new_import_statement)
    and a list of (path, unknown_name) for names without mapping.
    """
    planned: Dict[Path, List[Tuple[str, str, str]]] = {}
    unknowns: List[Tuple[Path, str]] = []
    for f in files:
        changed: List[Tuple[str, str, str]] = []
        text = f.read_text(encoding="utf8")

        # First handle single-line imports
        for line in text.splitlines():
            parsed = parse_imports(line)
            if not parsed:
                continue
            dests_by_module: Dict[str, List[str]] = {}
            missing = []
            for name in parsed:
                if name in mapping:
                    dest = mapping[name]
                    dests_by_module.setdefault(dest, []).append(name)
                else:
                    missing.append(name)
            if missing:
                for nm in missing:
                    unknowns.append((f, nm))
                continue
            for dest_mod, names in dests_by_module.items():
                new_import = f"from {dest_mod} import {', '.join(names)}"
                changed.append((line, line.strip(), new_import))

        # Now handle multiline imports like "from menipy.models.datatypes import (\n  A,\n  B,\n)"
        blocks = collect_multiline_imports(text)
        for start_idx, block_text in blocks:
            # extract names between parentheses robustly
            try:
                start = block_text.index("(") + 1
                end = block_text.rindex(")")
                inner = block_text[start:end]
            except ValueError:
                inner = block_text
            # split on commas and newlines
            parts = [
                n.strip().strip(",") for n in re.split(r"[,\n]", inner) if n.strip()
            ]
            dests_by_module: Dict[str, List[str]] = {}
            missing = []
            for name in parts:
                if name in mapping:
                    dests_by_module.setdefault(mapping[name], []).append(name)
                else:
                    missing.append(name)
            if missing:
                for nm in missing:
                    unknowns.append((f, nm))
                continue
            # Build a combined replacement line grouping names by destination module.
            for dest_mod, names in dests_by_module.items():
                new_import = f"from {dest_mod} import {', '.join(names)}"
                changed.append((block_text, block_text.strip(), new_import))

        if changed:
            planned[f] = changed
    # Filter out accidental unknowns from the tool files themselves
    filtered_unknowns: List[Tuple[Path, str]] = []
    for p, nm in unknowns:
        # ignore unknowns originating from the tools/ directory or this script
        try:
            rel = p.resolve().relative_to(ROOT)
        except Exception:
            rel = p
        if isinstance(rel, Path) and rel.parts and rel.parts[0] == "tools":
            continue
        filtered_unknowns.append((p, nm))

    return planned, filtered_unknowns


def apply_rewrites(planned: Dict[Path, List[Tuple[str, str, str]]]) -> None:
    for f, changes in planned.items():
        text = f.read_text(encoding="utf8")
        for original_line, old_stmt, new_stmt in changes:
            # Replace the old import statement (exact substring match of old_stmt)
            text = text.replace(old_stmt, new_stmt)
        f.write_text(text, encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate imports from menipy.models.datatypes to new model modules"
    )
    parser.add_argument("--apply", action="store_true", help="Apply changes in-place")
    parser.add_argument(
        "--mapping", type=Path, help="JSON file containing name->module mapping"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root to scan")
    parser.add_argument(
        "--report", type=Path, help="Write JSON report of planned changes"
    )
    args = parser.parse_args()

    mapping = dict(DEFAULT_MAPPING)
    if args.mapping:
        js = json.loads(args.mapping.read_text(encoding="utf8"))
        mapping.update(js)

    files = find_py_files(args.root)
    planned, unknowns = plan_rewrites(files, mapping)

    if not planned and not unknowns:
        print("No imports from menipy.models.datatypes found.")
        return

    print("Planned rewrites (dry-run):")
    for f, changes in planned.items():
        print(f"\nFile: {f}")
        for orig, old_stmt, new_stmt in changes:
            print(f"  - {old_stmt}  =>  {new_stmt}")

    if unknowns:
        print("\nUnknown symbol usages (no mapping):")
        for f, nm in unknowns:
            print(f"  - {nm} in {f}")
        print(
            "\nProvide a mapping file via --mapping or extend DEFAULT_MAPPING in this script to handle these names."
        )

    if args.report:
        report = {
            "planned": {
                str(k): [(a, b, c) for (a, b, c) in v] for k, v in planned.items()
            },
            "unknowns": [(str(p), n) for p, n in unknowns],
        }
        args.report.write_text(json.dumps(report, indent=2), encoding="utf8")
        print(f"Wrote report to {args.report}")

    if args.apply:
        if unknowns:
            print(
                "Refusing to apply because of unknown symbol mappings. Provide an explicit mapping first."
            )
            return
        print("Applying rewrites...")
        apply_rewrites(planned)
        print("Applied changes.")


if __name__ == "__main__":
    main()
