"""Find Python modules under src/menipy that are not imported anywhere in the repository.

This script builds absolute module names (menipy....) for each .py file under src/menipy,
parses AST imports from all .py files, resolves relative imports, then marks modules
referenced either directly or via package __init__ re-exports.

It prints a JSON list of unreferenced file paths (relative to repo root).
"""

import ast
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
package_root = src_root / "menipy"

# Gather all python files under src/menipy
py_files = [p for p in package_root.rglob("*.py")]


# Map file path -> module name (menipy....)
def module_name_for(path: Path):
    rel = path.relative_to(src_root)
    parts = list(rel.with_suffix("").parts)
    # remove trailing __init__ part
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


file_to_module = {p: module_name_for(p) for p in py_files}
module_to_file = {m: p for p, m in file_to_module.items()}

# Parse imports across all python files in the repo (including tests) to collect referenced modules
referenced = set()
imported_packages = set()

all_py = [
    p for p in Path(repo_root).rglob("*.py") if repo_root in p.parents or p == repo_root
]
for p in all_py:
    try:
        src = p.read_text(encoding="utf-8")
    except Exception:
        continue
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                # record top-level module and full module
                referenced.add(name)
                # also record the top two parts to be conservative
                # but keep full too
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            level = node.level
            if level == 0:
                if mod:
                    referenced.add(mod)
            else:
                # resolve relative import
                # compute module of current file
                # skip files outside src_root
                try:
                    cur_mod = module_name_for(p)  # may be 'menipy.foo.bar'
                except Exception:
                    continue
                cur_parts = cur_mod.split(".") if cur_mod else []
                if len(cur_parts) == 0:
                    continue
                # when file is a package __init__, cur_mod may be 'menipy.pkg'
                # level=1 means one level up from current module
                target_parts = (
                    cur_parts[: len(cur_parts) - level + 1] if level > 0 else cur_parts
                )
                if mod:
                    mod_parts = mod.split(".")
                    target_parts = target_parts + mod_parts
                abs_mod = ".".join(target_parts)
                if abs_mod:
                    referenced.add(abs_mod)

# At this point 'referenced' contains many module names (some not under menipy)
# We'll expand package-level imports by reading __init__.py files: if a package is imported
# and its __init__ re-exports submodules (via relative imports), mark those submodules referenced.

# Build package re-exports map: for any package path under menipy, list modules it imports via relative imports
package_exports = {}
for p, mod in file_to_module.items():
    # only consider __init__.py files
    if p.name != "__init__.py":
        continue
    # parse its AST and record imported submodules
    try:
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        continue
    exports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1:
                # relative import from package __init__ -> submodule
                base_parts = mod.split(".")
                # for level==1, from .sub import -> base_parts + sub
                target_parts = (
                    base_parts[: len(base_parts) - node.level + 1]
                    if node.level > 0
                    else base_parts
                )
                if node.module:
                    target_parts = target_parts + node.module.split(".")
                abs_mod = ".".join([p for p in target_parts if p])
                if abs_mod:
                    exports.add(abs_mod)
            else:
                # from menipy.something import ...
                if node.module and node.module.startswith("menipy"):
                    exports.add(node.module)
    package_exports[mod] = exports

# If referenced contains a package, add its exports as referenced
changed = True
while changed:
    changed = False
    new_refs = set()
    for r in list(referenced):
        # if r corresponds to a package __init__ in our files
        if r in package_exports:
            for ex in package_exports[r]:
                if ex not in referenced:
                    new_refs.add(ex)
    if new_refs:
        referenced.update(new_refs)
        changed = True

# Normalize referenced to only modules under 'menipy' and without trailing attrs
ref_menipy = set()
for r in referenced:
    if not isinstance(r, str):
        continue
    if r.startswith("menipy"):
        # consider only module prefix (full path)
        ref_menipy.add(r)

# Also, some imports reference subpackages (like menipy.models.geometry). Mark parents as referenced too
ref_with_parents = set(ref_menipy)
for m in list(ref_menipy):
    parts = m.split(".")
    for i in range(1, len(parts)):
        ref_with_parents.add(".".join(parts[: i + 1]))

# Now decide which file modules are not referenced
unreferenced = []
for p, mod in file_to_module.items():
    # Ignore private modules (starting with _)
    if Path(p).name.startswith("_") and Path(p).name != "__init__.py":
        # still include them, but mark separately
        pass
    # consider module referenced if mod in ref_with_parents or any parent package imported
    if mod.startswith("menipy"):
        if mod in ref_with_parents:
            continue
        # additionally, if the parent package is imported and the parent's __init__ contains this module in exports, it would have been marked
        unreferenced.append(str(p.relative_to(repo_root)))

# Sort and print
unreferenced.sort()
print(json.dumps(unreferenced, indent=2))
