# src/menipy/common/registry.py
"""Central registry for pipeline components and plugins."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class Registry:
    """Generic registry for named pipeline components."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a new item by name."""
        self._items[name] = fn

    def get(self, name: str, default: Any = None) -> Callable[..., Any] | None:
        """Get a registered item by name."""
        return self._items.get(name, default)

    # Dict-like interface for backward compatibility
    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self._items[name]

    def __setitem__(self, name: str, fn: Callable[..., Any]) -> None:
        self.register(name, fn)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        """__iter__."""
        return iter(self._items)

    def __len__(self) -> int:
        """__len__."""
        return len(self._items)

    def items(self):
        """Items."""
        return self._items.items()

    def keys(self):
        """Keys."""
        return self._items.keys()

    def values(self):
        """values.

        Returns
        -------
        type
        Description.
        """
        return self._items.values()


# Core plugin registries for small, pluggable utilities used by pipeline stages.
EDGE_DETECTORS = Registry("edge_detectors")
SOLVERS = Registry("solvers")
PENDANT_APPROXIMATORS = Registry("pendant_approximators")
PENDANT_INITIALIZERS = Registry("pendant_initializers")
SEGMENTATION_PROVIDERS = Registry("segmentation_providers")

# Other common stage utilities
ACQUISITIONS = Registry("acquisitions")
PREPROCESSORS = Registry("preprocessors")
GEOMETRIES = Registry("geometries")
SCALERS = Registry("scalers")
PHYSICS = Registry("physics")
OPTIMIZERS = Registry("optimizers")
OUTPUTS = Registry("outputs")
OVERLAYERS = Registry("overlayers")
VALIDATORS = Registry("validators")

# Utilities registry for image testing and analysis tools
UTILITIES = Registry("utilities")

# Detection registries for auto-calibration plugins
NEEDLE_DETECTORS = Registry("needle_detectors")
ROI_DETECTORS = Registry("roi_detectors")
SUBSTRATE_DETECTORS = Registry("substrate_detectors")
DROP_DETECTORS = Registry("drop_detectors")
APEX_DETECTORS = Registry("apex_detectors")


# Legacy wrapper functions for backward compatibility
def register_edge(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    EDGE_DETECTORS.register(name, fn)


def register_solver(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    SOLVERS.register(name, fn)


def register_pendant_approximator(name: str, fn: Callable[..., Any]) -> None:
    """Register a pendant surface-tension approximation plugin."""
    PENDANT_APPROXIMATORS.register(name, fn)


def register_pendant_initializer(name: str, fn: Callable[..., Any]) -> None:
    """Register a pendant geometry initializer."""
    PENDANT_INITIALIZERS.register(name, fn)


def register_segmentation_provider(name: str, fn: Callable[..., Any]) -> None:
    """Register a non-authoritative segmentation proposal provider."""
    SEGMENTATION_PROVIDERS.register(name, fn)


def register_acquisition(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    ACQUISITIONS.register(name, fn)


def register_preprocessor(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    PREPROCESSORS.register(name, fn)


def register_geometry(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    GEOMETRIES.register(name, fn)


def register_scaler(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    SCALERS.register(name, fn)


def register_physics(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    PHYSICS.register(name, fn)


def register_optimizer(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    OPTIMIZERS.register(name, fn)


def register_output(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    OUTPUTS.register(name, fn)


def register_overlayer(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    OVERLAYERS.register(name, fn)


def register_validator(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    VALIDATORS.register(name, fn)


def register_utility(name: str, fn: Callable[..., Any]) -> None:
    """Register a utility function for image testing/analysis."""
    UTILITIES.register(name, fn)


# Detector registration functions
def register_needle_detector(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    NEEDLE_DETECTORS.register(name, fn)


def register_roi_detector(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    ROI_DETECTORS.register(name, fn)


def register_substrate_detector(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    SUBSTRATE_DETECTORS.register(name, fn)


def register_drop_detector(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    DROP_DETECTORS.register(name, fn)


def register_apex_detector(name: str, fn: Callable[..., Any]) -> None:
    """Register plugin."""
    APEX_DETECTORS.register(name, fn)


def get_registry_snapshot() -> dict[str, dict[str, Callable[..., Any]]]:
    """Return a snapshot of all registries (useful for plugin discovery/debug).

    Example:
        from menipy.common import registry
        snapshot = registry.get_registry_snapshot()
    """
    return {
        "edge_detectors": dict(EDGE_DETECTORS),
        "solvers": dict(SOLVERS),
        "pendant_approximators": dict(PENDANT_APPROXIMATORS),
        "pendant_initializers": dict(PENDANT_INITIALIZERS),
        "segmentation_providers": dict(SEGMENTATION_PROVIDERS),
        "acquisitions": dict(ACQUISITIONS),
        "preprocessors": dict(PREPROCESSORS),
        "geometries": dict(GEOMETRIES),
        "scalers": dict(SCALERS),
        "physics": dict(PHYSICS),
        "optimizers": dict(OPTIMIZERS),
        "outputs": dict(OUTPUTS),
        "overlayers": dict(OVERLAYERS),
        "validators": dict(VALIDATORS),
        "needle_detectors": dict(NEEDLE_DETECTORS),
        "roi_detectors": dict(ROI_DETECTORS),
        "substrate_detectors": dict(SUBSTRATE_DETECTORS),
        "drop_detectors": dict(DROP_DETECTORS),
        "apex_detectors": dict(APEX_DETECTORS),
        "utilities": dict(UTILITIES),
    }
