"""Attempt to import all modules under src/menipy and report which succeed and what they load.

Outputs JSON with entries for each module file:
- module: menipy.foo.bar
- path: src/menipy/...
- import_ok: true/false
- error: traceback if failed
- new_loaded_modules: list of sys.modules keys added by the import

This helps narrow which modules are actually loadable/used at runtime.
"""

import importlib
import sys
import traceback
from pathlib import Path
import json

repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
package_root = src_root / "menipy"

# Ensure src/ is on sys.path so imports use the local source tree
src_dir = str(src_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

py_files = sorted(package_root.rglob("*.py"))


# build module names
def module_name_for(path: Path):
    rel = path.relative_to(src_root)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


results = []
for p in py_files:
    mod = module_name_for(p)
    before = set(sys.modules.keys())
    try:
        m = importlib.import_module(mod)
        ok = True
        err = None
    except Exception:
        ok = False
        err = traceback.format_exc()
    after = set(sys.modules.keys())
    new = sorted([n for n in after - before if n.startswith("menipy")])
    results.append(
        {
            "module": mod,
            "path": str(p.relative_to(repo_root)),
            "import_ok": ok,
            "error": err,
            "new_loaded_modules": new,
        }
    )

print(json.dumps(results, indent=2))
