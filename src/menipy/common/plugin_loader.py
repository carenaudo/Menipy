"""Utilities to apply registered plugin functions into pipeline instances.

The main helper `apply_registered_stages(pipeline, merge_strategy='override')`
merges functions from `menipy.common.registry` into the provided pipeline
instance. `merge_strategy` can be:
- 'override': plugin-provided stage replaces existing pipeline stage
- 'prepend': plugin stage is called before the pipeline's stage
- 'append': plugin stage is called after the pipeline's stage

This is intentionally small and conservative: it does not attempt to import
plugins automatically (discovery should be done elsewhere), it simply looks up
registered utilities and wires them onto a PipelineBase instance.
"""

from __future__ import annotations
from typing import Callable

from menipy.common import registry


def _wrap_chain(first: Callable, second: Callable) -> Callable:
    """Return a function that runs first(ctx) then second(ctx).

    Both callables follow the stage hook signature (ctx) -> Optional[ctx]. If
    the first returns a non-None context, that context is passed to the
    second. The wrapper returns the context from the second (or first if second
    returns None).
    """

    def wrapped(ctx):
        c = first(ctx)
        if c is None:
            c = ctx
        return second(c) or c

    return wrapped


def apply_registered_stages(pipeline, merge_strategy: str = "override") -> None:
    """Apply registered stage utilities to a PipelineBase instance.

    This function mutates the pipeline instance in-place. It looks up the
    pipeline name on `pipeline.name` and applies stage callables registered in
    `registry.PIPELINE_STAGES` as well as utilities in the stage-specific
    registries (preprocessors, scalers, etc.) by matching names. The exact
    merge behaviour is controlled via `merge_strategy`.
    """
    name = getattr(pipeline, "name", None)
    if not name:
        return

    # apply per-pipeline stage entries if the registry exposes them
    if hasattr(registry, "PIPELINE_STAGES"):
        stage_map = getattr(registry, "PIPELINE_STAGES").get(name, {})
        for stage_name, fn in stage_map.items():
            attr = f"do_{stage_name}"
            existing = getattr(pipeline, attr, None)
            if existing is None or merge_strategy == "override":
                setattr(pipeline, attr, fn)
            elif merge_strategy == "prepend":
                setattr(pipeline, attr, _wrap_chain(fn, existing))
            elif merge_strategy == "append":
                setattr(pipeline, attr, _wrap_chain(existing, fn))

    # Helper: map registry names to pipeline stage method names
    registry_to_stage = {
        "preprocessors": "preprocessing",
        "acquisitions": "acquisition",
        "geometries": "geometry",
        "scalers": "scaling",
        "physics": "physics",
        "optimizers": "optimization",
        "outputs": "outputs",
        "overlayers": "overlay",
        "validators": "validation",
    }

    snapshot = registry.get_registry_snapshot()
    for reg_name, stage_name in registry_to_stage.items():
        utils = snapshot.get(reg_name, {})
        # if multiple utils exist, apply them in alphabetical order
        for util_name in sorted(utils.keys()):
            fn = utils[util_name]
            attr = f"do_{stage_name}"
            existing = getattr(pipeline, attr, None)
            if existing is None or merge_strategy == "override":
                setattr(pipeline, attr, fn)
            elif merge_strategy == "prepend":
                setattr(pipeline, attr, _wrap_chain(fn, existing))
            elif merge_strategy == "append":
                setattr(pipeline, attr, _wrap_chain(existing, fn))


# ---------------------------------------------------------------------------
# Plugin Discovery and Loading
# ---------------------------------------------------------------------------

import logging
import sys
import importlib.util
from pathlib import Path
from typing import Optional, Any, List

logger = logging.getLogger(__name__)

# Default plugin directory relative to repository root
_DEFAULT_PLUGIN_DIR = Path(__file__).resolve().parents[3] / "plugins"


def _load_module_from_path(path: Path, module_name: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class PluginLoader:
    """Central plugin loader that scans directories and populates registries."""

    _instance: Optional["PluginLoader"] = None
    _loaded: bool = False

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """Initialize the plugin loader."""
        self.plugin_dirs = plugin_dirs or [_DEFAULT_PLUGIN_DIR]
        self._loaded_modules: dict[str, Any] = {}

    @classmethod
    def instance(cls) -> "PluginLoader":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def ensure_loaded(cls) -> int:
        """Ensure plugins are loaded (idempotent). Returns count of loaded plugins."""
        if cls._loaded:
            return len(cls.instance()._loaded_modules)
        return cls.instance().load_all()

    def load_all(self) -> int:
        """Scan all plugin directories and register discovered plugins."""
        count = 0
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.is_dir():
                logger.debug("Plugin directory does not exist: %s", plugin_dir)
                continue

            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    count += self._load_plugin_file(plugin_file)
                except Exception as exc:
                    logger.warning("Failed to load plugin %s: %s", plugin_file, exc)

        PluginLoader._loaded = True
        logger.info(
            "Loaded %d plugins from %d directories", count, len(self.plugin_dirs)
        )
        return count

    def _load_plugin_file(self, path: Path) -> int:
        """Load a single plugin file and register its components."""
        if path.stem in self._loaded_modules:
            return 0

        mod_name = f"menipy_plugins.{path.stem}"
        try:
            mod = _load_module_from_path(path, mod_name)
            self._loaded_modules[path.stem] = mod
            self._register_module_exports(mod, path.stem)
            logger.debug("Loaded plugin: %s", path.stem)
            return 1
        except Exception as exc:
            logger.warning("Error loading plugin %s: %s", path, exc)
            return 0

    def _register_module_exports(self, mod, name: str) -> None:
        """Register individual exported functions from a module."""
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            attr = getattr(mod, attr_name, None)
            if not callable(attr):
                continue

            # Register solvers (match laplace, solver in name)
            if (
                "solver" in name.lower()
                or "laplace" in attr_name.lower()
                or "solver" in attr_name.lower()
            ):
                if attr_name not in registry.SOLVERS:
                    registry.SOLVERS.register(attr_name, attr)
                    logger.debug("Registered solver: %s", attr_name)
            # Register edge detectors
            elif "edge" in name.lower() or "detector" in attr_name.lower():
                if attr_name not in registry.EDGE_DETECTORS:
                    registry.EDGE_DETECTORS.register(attr_name, attr)
                    logger.debug("Registered edge detector: %s", attr_name)

    def get_module(self, name: str) -> Optional[Any]:
        """Get a loaded module by name."""
        self.ensure_loaded()
        return self._loaded_modules.get(name)


from typing import Callable, Any


def get_solver(name: str, fallback: Optional[Callable[..., Any]] = None) -> Optional[Callable[..., Any]]:
    """Get a solver by name from the registry (ensures plugins are loaded first)."""
    PluginLoader.ensure_loaded()
    return registry.SOLVERS.get(name, fallback)


def get_edge_detector(name: str, fallback: Optional[Callable[..., Any]] = None) -> Optional[Callable[..., Any]]:
    """Get an edge detector by name from the registry (ensures plugins are loaded first)."""
    PluginLoader.ensure_loaded()
    return registry.EDGE_DETECTORS.get(name, fallback)
    PluginLoader.ensure_loaded()
    return registry.EDGE_DETECTORS.get(name, fallback)
