"""Plugin discovery helpers.

This module exposes a very small plugin system used primarily for the
refactoring effort.  Plugins are discovered via Python entry points under the
``og.analysis`` group.  Each entry point should resolve to a callable or module
which is stored in :data:`PLUGINS` under the entry point name.
"""

from __future__ import annotations

from importlib import metadata
import warnings

PLUGINS: dict[str, object] = {}


def load_plugins() -> None:
    """Discover installed plugins.

    This function populates :data:`PLUGINS` by querying the ``og.analysis``
    entry point group via :func:`importlib.metadata.entry_points`.  Any failures
    to load a plugin are reported as warnings and do not raise exceptions.
    """

    PLUGINS.clear()

    try:
        entries = metadata.entry_points(group="og.analysis")
    except TypeError:  # pragma: no cover - older importlib_metadata API
        entries = metadata.entry_points().get("og.analysis", [])

    for ep in entries:
        try:
            PLUGINS[ep.name] = ep.load()
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Failed to load plugin {ep.name!r}: {exc}")


__all__ = ["load_plugins", "PLUGINS"]

