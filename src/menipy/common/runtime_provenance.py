"""Describe the loaded plugin environment without importing new plugins."""

import hashlib
import inspect
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from menipy.common import registry


def runtime_provenance():
    """Hash registered callable source files on the worker, not the GUI thread."""
    paths = set()
    for name, value in tuple(vars(registry).items()):
        if not name.startswith("_") and isinstance(value, (dict, registry.Registry)):
            for _, function in tuple(value.items()):
                if inspect.isfunction(function) or inspect.isclass(function):
                    try:
                        path = inspect.getsourcefile(function)
                    except TypeError:
                        path = None
                    if path:
                        paths.add(Path(path))
    plugins = {}
    for path in sorted(paths):
        try:
            plugins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            plugins[str(path)] = None
    try:
        application_version = version("menipy")
    except PackageNotFoundError:
        application_version = "uninstalled"
    return {
        "application_version": application_version,
        "registered_source_sha256": plugins,
        "scope": "registered callable source environment; not proof of invocation",
    }
