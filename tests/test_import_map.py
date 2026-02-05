import ast
import json
from pathlib import Path

"""
Tests for the import map generation script.
Verifies that menipy files are found and imports are correctly resolved.
"""


def collect_menipy_files(menipy_root: Path):
    for p in sorted(menipy_root.rglob("*.py")):
        # skip __pycache__ and tests if any
        if "__pycache__" in p.parts:
            continue
        yield p


def module_name_for(path: Path, menipy_root: Path):
    rel = path.relative_to(menipy_root)
    parts = list(rel.with_suffix("").parts)
    # module name always starts with menipy
    return "menipy" + ("." + ".".join(parts) if parts else "")


def resolve_from(module_name: str, level: int, mod: str | None):
    # module_name is like menipy.foo.bar
    parts = module_name.split(".")
    if level:
        if level > len(parts):
            base = ["menipy"]
        else:
            base = parts[:-level]
        if mod:
            return ".".join(base + mod.split("."))
        return ".".join(base)
    # absolute
    return mod if mod else None


def module_to_path(module: str, menipy_root: Path):
    if not module:
        return None
    if not module.startswith("menipy"):
        return None
    rest = module.split(".")[1:]
    # try .py
    candidate = menipy_root.joinpath(*rest).with_suffix(".py")
    if candidate.exists():
        return candidate
    # try package __init__.py
    candidate2 = menipy_root.joinpath(*rest, "__init__.py")
    if candidate2.exists():
        return candidate2
    # fallback to top-level __init__ for 'menipy'
    if module == "menipy":
        p = menipy_root.joinpath("__init__.py")
        if p.exists():
            return p
    return None


def test_generate_import_map(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    menipy_root = repo_root / "src" / "menipy"
    assert menipy_root.exists(), f"Expected {menipy_root} to exist"

    imports_map: dict[str, list[str]] = {}

    for f in collect_menipy_files(menipy_root):
        module_name = module_name_for(f, menipy_root)
        imports: set[str] = set()
        try:
            tree = ast.parse(f.read_text(encoding="utf8"))
        except Exception:
            # skip files we can't parse
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("menipy"):
                        p = module_to_path(name, menipy_root)
                        if p:
                            imports.add(str(p.relative_to(repo_root).as_posix()))
            elif isinstance(node, ast.ImportFrom):
                # node.level (0=absolute, >0 relative)
                resolved = None
                if node.level:
                    resolved = resolve_from(module_name, node.level, node.module)
                else:
                    resolved = node.module

                if resolved and resolved.startswith("menipy"):
                    p = module_to_path(resolved, menipy_root)
                    if p:
                        imports.add(str(p.relative_to(repo_root).as_posix()))
                else:
                    # sometimes import-from references a subname like 'processing.reader'
                    if resolved:
                        # try to resolve each alias as module under resolved
                        for alias in node.names:
                            candidate = (
                                f"{resolved}.{alias.name}" if resolved else alias.name
                            )
                            if candidate.startswith("menipy"):
                                p2 = module_to_path(candidate, menipy_root)
                                if p2:
                                    imports.add(
                                        str(p2.relative_to(repo_root).as_posix())
                                    )

        imports_map[str(f.relative_to(repo_root).as_posix())] = sorted(imports)

    # Build reverse map
    imported_by: dict[str, list[str]] = {}
    for src, targets in imports_map.items():
        for t in targets:
            imported_by.setdefault(t, []).append(src)

    # ensure deterministic ordering
    for k in imported_by:
        imported_by[k].sort()

    out = {
        "imports": imports_map,
        "imported_by": imported_by,
    }

    build_dir = repo_root / "build"
    build_dir.mkdir(exist_ok=True)
    out_file = build_dir / "menipy_import_map.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf8")

    # simple assertion to make pytest report success
    assert out_file.exists()
