from __future__ import annotations
import sys, importlib.util
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

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
    # edges
    for name, file_path, entry in db.active_of_kind("edge"):
        mod = _load_module_from_path(Path(file_path), f"adsa_plugins.edge_{name}")
        _register_from_module(mod)
        count += 1
    # solvers
    for name, file_path, entry in db.active_of_kind("solver"):
        mod = _load_module_from_path(Path(file_path), f"adsa_plugins.solver_{name}")
        _register_from_module(mod)
        count += 1
    return count
