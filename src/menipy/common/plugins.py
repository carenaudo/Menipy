from __future__ import annotations
import sys, importlib.util
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional
import logging

from .registry import EDGE_DETECTORS, SOLVERS, register_edge, register_solver
from .plugin_db import PluginDB

def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # dynamic import from file :contentReference[oaicite:5]{index=5}
    return mod

def _register_from_module(mod) -> None:
    """
    Protocols supported:
      1) register(adsa) -> calls registry functions
      2) EDGE_DETECTORS / SOLVERS dicts
      3) get_edge_detectors() / get_solvers() returning name->callable
    """
    if hasattr(mod, "register"):
        mod.register({"register_edge": register_edge, "register_solver": register_solver})
    if hasattr(mod, "EDGE_DETECTORS"):
        for name, fn in mod.EDGE_DETECTORS.items():
            register_edge(name, fn)
    if hasattr(mod, "SOLVERS"):
        for name, fn in mod.SOLVERS.items():
            register_solver(name, fn)
    if hasattr(mod, "get_edge_detectors"):
        for name, fn in mod.get_edge_detectors().items():
            register_edge(name, fn)
    if hasattr(mod, "get_solvers"):
        for name, fn in mod.get_solvers().items():
            register_solver(name, fn)

def discover_into_db(db: PluginDB, plugin_dirs: Iterable[Path]) -> int:
    """
    Scan plugin_dirs for *.py files and upsert metadata into SQLite.
    Does not load modules. Activation is managed in DB.
    """
    n = 0
    for d in map(Path, plugin_dirs):
        if not d.is_dir():
            continue
        for path in d.glob("*.py"):
            # naive introspection: prefer filename as name; detect kind by filename prefix
            name = path.stem
            kind = "edge" if "edge" in name else ("solver" if "solver" in name else "edge")
            db.upsert_plugin(name=name, kind=kind, file_path=path,
                             entry=None, description=f"Plugin {name} from {d}", version=None)
            n += 1
    return n

def load_active_plugins(db: PluginDB) -> int:
    """
    Load only ACTIVE plugins from SQLite and register them in the in-memory registry.
    """
    count = 0

    def _try_load(kind: str):
        nonlocal count
        for name, file_path, entry in db.active_of_kind(kind):
            try:
                if not file_path:
                    raise FileNotFoundError(f"no file_path recorded for plugin '{name}'")
                p = Path(file_path)
                if not p.exists():
                    # try resolving relative to the current working dir
                    p = p.resolve()
                if not p.exists():
                    raise FileNotFoundError(str(p))

                mod_name = f"adsa_plugins.{kind}_{name}"
                mod = _load_module_from_path(p, mod_name)
                _register_from_module(mod)
                count += 1
            except Exception as exc:  # pragma: no cover - report and continue
                # Log a clear, actionable error but don't abort loading other plugins
                logging.error("Failed to load %s plugin '%s' from %s: %s", kind, name, file_path, exc)

    # edges
    _try_load("edge")
    # solvers
    _try_load("solver")

    return count


def list_plugins_status(db: PluginDB) -> list[dict]:
    """Return a list of dicts describing plugins and whether they load successfully.

    Each dict contains: name, kind, file_path, entry, description, version, is_active, loaded (bool), error (optional).
    """
    rows = []
    for name, kind, file_path, entry, description, version, is_active in db.list_plugins():
        status = {"name": name, "kind": kind, "file_path": file_path, "entry": entry,
                  "description": description, "version": version, "is_active": bool(is_active)}
        if not is_active:
            status.update({"loaded": False, "error": "not active"})
            rows.append(status)
            continue
        try:
            p = Path(file_path)
            if not p.exists():
                raise FileNotFoundError(str(p))
            mod_name = f"adsa_plugins.{kind}_{name}"
            mod = _load_module_from_path(p, mod_name)
            # simple sanity: must expose at least one of the expected symbols
            ok = any(hasattr(mod, k) for k in ("register", "EDGE_DETECTORS", "SOLVERS", "get_edge_detectors", "get_solvers"))
            if not ok:
                status.update({"loaded": False, "error": "missing expected plugin registration symbols"})
            else:
                status.update({"loaded": True})
        except Exception as exc:
            status.update({"loaded": False, "error": str(exc)})
        rows.append(status)
    return rows


def cli_list_plugins(argv=None) -> int:
    """Small CLI entry that prints plugin list and load status.

    Intended to be exposed as a console script (menipy-plugins).
    """
    import argparse, json
    ap = argparse.ArgumentParser(prog="menipy-plugins", description="List plugins and show load status")
    ap.add_argument("--db", type=str, default=None, help="Plugin DB path")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args(argv)
    db = PluginDB(Path(args.db) if args.db else PluginDB().db_path)
    rows = list_plugins_status(db)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    # plain table
    fmt = "{name:20} {kind:8} {active:6} {loaded:6} {error}"
    print(fmt.format(name="NAME", kind="KIND", active="ACTIVE", loaded="LOADED", error="ERROR"))
    for r in rows:
        print(fmt.format(name=r["name"][:20], kind=r["kind"], active=str(r["is_active"]), loaded=str(r.get("loaded", False)), error=r.get("error", "")))
    return 0
