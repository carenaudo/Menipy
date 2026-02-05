"""
Dialog for managing and configuring plugins.
"""

from __future__ import annotations
import json
from typing import Sequence
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QDialog, QTableWidgetItem, QMessageBox, QAbstractItemView, 
    QPushButton, QStyle, QWidget, QHBoxLayout, QToolButton,
    QInputDialog
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

from menipy.gui.viewmodels.plugins_vm import PluginsViewModel
from menipy.gui.services.settings_service import AppSettings
from menipy.common.plugin_settings import get_detector_settings_model, DETECTOR_SETTINGS
from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog

_COL_ACTIVE, _COL_NAME, _COL_KIND, _COL_CONFIG, _COL_PATH, _COL_DESC, _COL_VER = range(7)


class PluginManagerDialog(QDialog):
    def __init__(self, vm: PluginsViewModel, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.vm = vm
        self.settings = settings
        self._updating = False  # guard against recursive cellChanged

        # Load UI (resource path first, fallback to file path)
        loader = QUiLoader()
        ui_file = QFile(":/views/plugin_manager.ui")
        if not ui_file.exists():
            ui_file = QFile(
                str(
                    Path(__file__).resolve().parent.parent
                    / "views"
                    / "plugin_manager.ui"
                )
            )
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        self.setLayout(self.ui.layout())

        # Wire controls
        try:
            self.ui.closeButton.clicked.connect(
                self.accept
            )  # optional if present in UI
        except Exception:
            pass
        self.ui.rescanButton.clicked.connect(self._on_rescan)
        self.ui.activateAllButton.clicked.connect(lambda: self._activate_all(True))
        self.ui.deactivateAllButton.clicked.connect(lambda: self._activate_all(False))

        # Init dirs line edit
        self.ui.dirsEdit.setText(self._dirs_as_text(self.settings.plugin_dirs))

        # Table setup
        t = self.ui.pluginsTable
        t.setColumnCount(7)  # Increased for Config column
        t.setHorizontalHeaderLabels(
            ["Active", "Name", "Kind", "Config", "Path", "Description", "Version"]
        )
        # Use enums from QAbstractItemView for selection/edit behavior
        try:
            t.setSelectionBehavior(QAbstractItemView.SelectRows)  # older-style enum
            t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        except Exception:
            # Qt6 namespaced enums fallback
            t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setColumnWidth(_COL_ACTIVE, 60)
        t.setColumnWidth(_COL_NAME, 150)
        t.setColumnWidth(_COL_KIND, 90)
        t.setColumnWidth(_COL_CONFIG, 60) # Small width for button
        t.setColumnWidth(_COL_PATH, 280)
        t.horizontalHeader().setStretchLastSection(True)
        t.cellChanged.connect(self._on_cell_changed)

        # Populate
        self.refresh()

    # --------------------------- helpers -------------------------------------

    def _dirs_as_text(self, dirs: Sequence[str]) -> str:
        # use ':' separator on all platforms; accept ';' too
        return ":".join(dirs) if dirs else "./plugins"

    def _parse_dirs(self, text: str) -> list[str]:
        parts = [p.strip() for p in text.replace(";", ":").split(":") if p.strip()]
        return parts or ["./plugins"]

    # --------------------------- actions -------------------------------------

    def refresh(self):
        self._updating = True
        rows = self.vm.rows()  # list of tuples from DB
        t = self.ui.pluginsTable
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, (name, kind, path, entry, desc, ver, active) in enumerate(rows):
            # Active (checkbox)
            item_active = QTableWidgetItem()
            item_active.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item_active.setCheckState(Qt.Checked if int(active) else Qt.Unchecked)
            # keep name/kind as data to use on toggle
            item_active.setData(Qt.UserRole, (name, kind))
            t.setItem(i, _COL_ACTIVE, item_active)

            # Name / Kind / Path / Desc / Version
            t.setItem(i, _COL_NAME, QTableWidgetItem(str(name)))
            t.setItem(i, _COL_KIND, QTableWidgetItem(str(kind)))
            
            # Config Button
            # Use a container widget to center the button and avoid full-cell stretch
            container = QWidget()
            # Ensure container is transparent so row selection/color shows through
            container.setStyleSheet("background-color: transparent;")
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            
            btn = QToolButton()
            btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
            btn.setToolTip("Configure Plugin")
            btn.setAutoRaise(True) # Make it look cleaner (flat)
            btn.setFixedSize(28, 28) # Slightly larger touch target
            btn.setIconSize(QSize(18, 18)) # Ensure icon is readable
            # Add a style for the button to ensure it doesn't look like a solid block
            btn.setStyleSheet("QToolButton { border: none; background: transparent; } QToolButton:hover { background: rgba(255, 255, 255, 30); border-radius: 4px; }")

            # Connect using closure to capture variables
            btn.clicked.connect(lambda checked=False, n=name, k=kind: self._on_configure(n, k))
            
            layout.addWidget(btn)
            t.setCellWidget(i, _COL_CONFIG, container)

            t.setItem(i, _COL_PATH, QTableWidgetItem(str(path)))
            t.setItem(i, _COL_DESC, QTableWidgetItem(str(desc or "")))
            t.setItem(i, _COL_VER, QTableWidgetItem(str(ver or "")))
        t.blockSignals(False)
        self._updating = False

    def _on_rescan(self):
        # Parse dirs from edit, persist to settings and DB, rescan and reload
        text = self.ui.dirsEdit.text().strip()
        dirs = self._parse_dirs(text)
        # persist to app settings
        try:
            self.settings.plugin_dirs = dirs
            self.settings.save()
        except Exception:
            pass
        # persist to DB settings for CLI/shared use
        try:
            if hasattr(self.vm, "svc") and hasattr(self.vm.svc, "db"):
                self.vm.svc.db.set_setting("plugin_dirs", ":".join(dirs))  # type: ignore[attr-defined]
        except Exception:
            pass
        # Discover plugins and reload active ones
        try:
            self.vm.discover(dirs)
        except Exception as e:
            QMessageBox.critical(self, "Rescan failed", str(e))
            return
        self.refresh()

    def _activate_all(self, make_active: bool) -> None:
        # Toggle all listed plugins to the desired state
        try:
            t = self.ui.pluginsTable
            for row in range(t.rowCount()):
                name = t.item(row, _COL_NAME).text()
                kind = t.item(row, _COL_KIND).text()
                self.vm.toggle(name, kind, make_active)
        except Exception as e:
            QMessageBox.critical(self, "Activate/Deactivate failed", str(e))
        self.refresh()

    def _on_cell_changed(self, row: int, col: int):
        if self._updating or col != _COL_ACTIVE:
            return
        item = self.ui.pluginsTable.item(row, _COL_ACTIVE)
        if not item:
            return
        active = item.checkState() == Qt.Checked
        name = self.ui.pluginsTable.item(row, _COL_NAME).text()
        kind = self.ui.pluginsTable.item(row, _COL_KIND).text()
        try:
            self.vm.toggle(name, kind, active)
        except Exception as e:
            # Show error and revert checkbox state
            QMessageBox.critical(self, "Toggle failed", str(e))
            t = self.ui.pluginsTable
            t.blockSignals(True)
            # revert to previous state
            item.setCheckState(Qt.Unchecked if active else Qt.Checked)
            t.blockSignals(False)

    def _find_settings_models_for_plugin(self, plugin_name: str) -> dict[str, type]:
        """Find all settings models registered by this plugin module."""
        matches = {}
        # Expected module name for the plugin
        module_name = f"menipy_plugins.{plugin_name}"
        
        # Iterate all registered detector settings
        # We access the internal dict of the Registry
        for method_name, model_cls in DETECTOR_SETTINGS.items(): # Registry supports items()
            if model_cls.__module__ == module_name:
                matches[method_name] = model_cls
        return matches

    def _on_configure(self, name: str, kind: str):
        """Open configuration dialog for the plugin."""
        # 1. Try to find settings models associated with this plugin name (module)
        models = self._find_settings_models_for_plugin(name)
        
        target_name = name
        model_cls = None
        
        if not models:
             # Fallback: maybe the name itself IS the method name (unlikely given naming convention, but possible)
             model_cls = get_detector_settings_model(name)
             if model_cls:
                 target_name = name
        elif len(models) == 1:
            # Only one configurable item, auto-select
            target_name, model_cls = list(models.items())[0]
        else:
            # Multiple items, ask user
            items = list(models.keys())
            item, ok = QInputDialog.getItem(
                self, 
                "Select Configuration", 
                "Choose component to configure:", 
                items, 
                0, 
                False
            )
            if ok and item:
                target_name = item
                model_cls = models[item]
            else:
                return

        if not model_cls:
            QMessageBox.information(
                self, 
                "No Configuration", 
                f"No configurable settings found for plugin '{name}'."
            )
            return
 
        # Load existing settings from DB (key uses the *target_name*, i.e., the method name)
        # We use a specific key format for method-specific defaults
        conf_key = f"plugin:{target_name}:config"
        
        current_values = {}
        try:
            db_val = None
            if hasattr(self.vm, "svc") and hasattr(self.vm.svc, "db"):
                 db_val = self.vm.svc.db.get_setting(conf_key)
            if db_val:
                current_values = json.loads(db_val)
        except Exception as e:
            print(f"Error loading settings for {target_name}: {e}")

        # Show dialog
        dlg = PluginConfigDialog(model_cls, current_values, self)
        if dlg.exec():
            new_settings = dlg.get_settings()
            # Save to DB
            try:
                if hasattr(self.vm, "svc") and hasattr(self.vm.svc, "db"):
                    self.vm.svc.db.set_setting(conf_key, json.dumps(new_settings))
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Could not save settings: {e}")

