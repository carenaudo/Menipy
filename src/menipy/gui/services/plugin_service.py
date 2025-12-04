"""
Plugin management service for GUI.
"""
# src/adsa/gui/services/plugin_service.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence
from menipy.common.plugin_db import PluginDB
from menipy.common.plugins import discover_into_db, load_active_plugins

class PluginService:
    def __init__(self, db: PluginDB):
        self.db = db

    def discover(self, dirs: Sequence[str | Path]):
        discover_into_db(self.db, [Path(p) for p in dirs])

    def list(self, only_active=None):
        return self.db.list_plugins(only_active=only_active)

    def set_active(self, name: str, kind: str, active: bool):
        self.db.set_active(name, kind, active)

    def load_active(self) -> int:
        return load_active_plugins(self.db)
