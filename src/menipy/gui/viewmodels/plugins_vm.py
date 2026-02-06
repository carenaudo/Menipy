"""View model for plugin management UI."""

# src/adsa/gui/viewmodels/plugins_vm.py
from __future__ import annotations
from PySide6.QtCore import QObject, Signal
from menipy.gui.viewmodels import run_vm
from menipy.gui.services.plugin_service import PluginService


class PluginsViewModel(QObject):
    changed = Signal()

    def __init__(self, db):
        """__init__."""
        super().__init__()
        self.svc = PluginService(db)

    def refresh(self):
        """Placeholder docstring for refresh.
    
        TODO: Complete docstring with full description.
    
        Returns
        -------
        type
        Description of return value.
        """
        self.changed.emit()

    def discover(self, dirs):
        """Placeholder docstring for discover.
    
        TODO: Complete docstring with full description.
    
        Returns
        -------
        type
        Description of return value.
        """
        self.svc.discover(dirs)
        self.changed.emit()

    def toggle(self, name: str, kind: str, active: bool):
        """Placeholder docstring for toggle.
    
        TODO: Complete docstring with full description.
    
        Parameters
        ----------
        dirs : type
        Description of dirs.
    
        Returns
        -------
        type
        Description of return value.
        """
        self.svc.set_active(name, kind, active)
        self.svc.load_active()
        self.changed.emit()

    def rows(self):
        """Placeholder docstring for rows.
    
        TODO: Complete docstring with full description.
    
        Parameters
        ----------
        dirs : type
        Description of dirs.
    
        Returns
        -------
        type
        Description of return value.
        """
        return self.svc.list(only_active=None)
