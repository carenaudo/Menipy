# src/adsa/common/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any

# Core plugin registries for small, pluggable utilities used by pipeline stages.
# Plugins can import these helpers and register implementations at import time.

EDGE_DETECTORS: Dict[str, Callable[..., Any]] = {}
SOLVERS: Dict[str, Callable[..., Any]] = {}

# Other common stage utilities
ACQUISITIONS: Dict[str, Callable[..., Any]] = {}
PREPROCESSORS: Dict[str, Callable[..., Any]] = {}
GEOMETRIES: Dict[str, Callable[..., Any]] = {}
SCALERS: Dict[str, Callable[..., Any]] = {}
PHYSICS: Dict[str, Callable[..., Any]] = {}
OPTIMIZERS: Dict[str, Callable[..., Any]] = {}
OUTPUTS: Dict[str, Callable[..., Any]] = {}
OVERLAYERS: Dict[str, Callable[..., Any]] = {}
VALIDATORS: Dict[str, Callable[..., Any]] = {}


def register_edge(name: str, fn: Callable[..., Any]) -> None:
    EDGE_DETECTORS[name] = fn


def register_solver(name: str, fn: Callable[..., Any]) -> None:
    SOLVERS[name] = fn


def register_acquisition(name: str, fn: Callable[..., Any]) -> None:
    ACQUISITIONS[name] = fn


def register_preprocessor(name: str, fn: Callable[..., Any]) -> None:
    PREPROCESSORS[name] = fn


def register_geometry(name: str, fn: Callable[..., Any]) -> None:
    GEOMETRIES[name] = fn


def register_scaler(name: str, fn: Callable[..., Any]) -> None:
    SCALERS[name] = fn


def register_physics(name: str, fn: Callable[..., Any]) -> None:
    PHYSICS[name] = fn


def register_optimizer(name: str, fn: Callable[..., Any]) -> None:
    OPTIMIZERS[name] = fn


def register_output(name: str, fn: Callable[..., Any]) -> None:
    OUTPUTS[name] = fn


def register_overlayer(name: str, fn: Callable[..., Any]) -> None:
    OVERLAYERS[name] = fn


def register_validator(name: str, fn: Callable[..., Any]) -> None:
    VALIDATORS[name] = fn


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
    }
