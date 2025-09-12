# src/adsa/gui/viewmodels/plugins_vm.py
from __future__ import annotations
from PySide6.QtCore import QObject, Signal
from menipy.gui.viewmodels import run_vm
from menipy.gui.services.plugin_service import PluginService

class PluginsViewModel(QObject):
    changed = Signal()

    def __init__(self, db):
        super().__init__()
        self.svc = PluginService(db)

    def refresh(self):
        self.changed.emit()

    def discover(self, dirs):
        self.svc.discover(dirs)
        self.changed.emit()

    def toggle(self, name: str, kind: str, active: bool):
        self.svc.set_active(name, kind, active)
        self.svc.load_active()
        self.changed.emit()

    def rows(self):
        return self.svc.list(only_active=None)
