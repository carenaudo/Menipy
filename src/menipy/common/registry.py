# src/menipy/common/registry.py
"""
Central registry for pipeline components and plugins.
"""
from __future__ import annotations
from typing import Callable, Dict, Any, Optional, Iterator


class Registry:
    """Generic registry for named pipeline components."""

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a new item by name."""
        self._items[name] = fn

    def get(self, name: str, default: Any = None) -> Optional[Callable[..., Any]]:
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
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()


# Core plugin registries for small, pluggable utilities used by pipeline stages.
EDGE_DETECTORS = Registry("edge_detectors")
SOLVERS = Registry("solvers")

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

# Detection registries for auto-calibration plugins
NEEDLE_DETECTORS = Registry("needle_detectors")
ROI_DETECTORS = Registry("roi_detectors")
SUBSTRATE_DETECTORS = Registry("substrate_detectors")
DROP_DETECTORS = Registry("drop_detectors")
APEX_DETECTORS = Registry("apex_detectors")


# Legacy wrapper functions for backward compatibility
def register_edge(name: str, fn: Callable[..., Any]) -> None:
    EDGE_DETECTORS.register(name, fn)


def register_solver(name: str, fn: Callable[..., Any]) -> None:
    SOLVERS.register(name, fn)


def register_acquisition(name: str, fn: Callable[..., Any]) -> None:
    ACQUISITIONS.register(name, fn)


def register_preprocessor(name: str, fn: Callable[..., Any]) -> None:
    PREPROCESSORS.register(name, fn)


def register_geometry(name: str, fn: Callable[..., Any]) -> None:
    GEOMETRIES.register(name, fn)


def register_scaler(name: str, fn: Callable[..., Any]) -> None:
    SCALERS.register(name, fn)


def register_physics(name: str, fn: Callable[..., Any]) -> None:
    PHYSICS.register(name, fn)


def register_optimizer(name: str, fn: Callable[..., Any]) -> None:
    OPTIMIZERS.register(name, fn)


def register_output(name: str, fn: Callable[..., Any]) -> None:
    OUTPUTS.register(name, fn)


def register_overlayer(name: str, fn: Callable[..., Any]) -> None:
    OVERLAYERS.register(name, fn)


def register_validator(name: str, fn: Callable[..., Any]) -> None:
    VALIDATORS.register(name, fn)

# Detector registration functions
def register_needle_detector(name: str, fn: Callable[..., Any]) -> None:
    NEEDLE_DETECTORS.register(name, fn)

def register_roi_detector(name: str, fn: Callable[..., Any]) -> None:
    ROI_DETECTORS.register(name, fn)

def register_substrate_detector(name: str, fn: Callable[..., Any]) -> None:
    SUBSTRATE_DETECTORS.register(name, fn)

def register_drop_detector(name: str, fn: Callable[..., Any]) -> None:
    DROP_DETECTORS.register(name, fn)

def register_apex_detector(name: str, fn: Callable[..., Any]) -> None:
    APEX_DETECTORS.register(name, fn)


def get_registry_snapshot() -> Dict[str, Dict[str, Callable[..., Any]]]:
    """Return a snapshot of all registries (useful for plugin discovery/debug).

    Example:
        from menipy.common import registry
        snapshot = registry.get_registry_snapshot()
    """
    return {
        "edge_detectors": dict(EDGE_DETECTORS),
        "solvers": dict(SOLVERS),
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
    }
