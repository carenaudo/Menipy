# src/adsa/common/registry.py
from __future__ import annotations
from typing import Callable, Dict

EDGE_DETECTORS: Dict[str, Callable] = {}
SOLVERS: Dict[str, Callable] = {}

def register_edge(name: str, fn: Callable) -> None:
    EDGE_DETECTORS[name] = fn

def register_solver(name: str, fn: Callable) -> None:
    SOLVERS[name] = fn
