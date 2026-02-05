from pathlib import Path
import json

"""
Merges static and runtime import analysis data into a combined JSON report.
Combines 'menipy_import_map.json' and 'import_all_report.json'.
"""

repo = Path(__file__).resolve().parents[1]
build = repo / "build"
jm = build / "menipy_import_map.json"
jr = build / "import_all_report.json"

if not jm.exists():
    raise SystemExit(f"Missing {jm}")
if not jr.exists():
    raise SystemExit(f"Missing {jr}")

jmj = json.loads(jm.read_text(encoding="utf8"))
jrj = json.loads(jr.read_text(encoding="utf8"))
imports = jmj.get("imports", {})
imported_by = jmj.get("imported_by", {})

# map runtime import_ok by path (posix style)
runtime_map = {}
for item in jrj:
    p = item["path"].replace("\\", "/")
    runtime_map[p] = item.get("import_ok")

all_files = sorted(imports.keys())
combined = {}
for f in all_files:
    combined[f] = {
        "imports": imports.get(f, []),
        "imported_by": imported_by.get(f, []),
        "runtime_importable": runtime_map.get(f, None),
    }

# orphans: no incoming edges
orphans = [f for f, v in combined.items() if not v["imported_by"]]
orphans_runtime_true = [f for f in orphans if combined[f]["runtime_importable"] is True]
orphans_runtime_false = [
    f for f in orphans if combined[f]["runtime_importable"] is False
]
orphans_runtime_unknown = [
    f for f in orphans if combined[f]["runtime_importable"] is None
]

out = {
    "combined": combined,
    "orphans": orphans,
    "orphans_runtime_true": orphans_runtime_true,
    "orphans_runtime_false": orphans_runtime_false,
    "orphans_runtime_unknown": orphans_runtime_unknown,
}

outp = build / "combined_import_analysis.json"
outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf8")

print("Wrote", outp)
print("\nOrphans (no incoming static imports):")
for f in orphans:
    print("-", f, "runtime_importable=", combined[f]["runtime_importable"])

print("\nSummary:")
print(" total_files=", len(all_files))
print(" orphans=", len(orphans))
print(" orphans_runtime_true=", len(orphans_runtime_true))
print(" orphans_runtime_false=", len(orphans_runtime_false))
print(" orphans_runtime_unknown=", len(orphans_runtime_unknown))
