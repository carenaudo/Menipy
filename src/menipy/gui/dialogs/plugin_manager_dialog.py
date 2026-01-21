"""
Dialog for managing and configuring plugins.
"""

from __future__ import annotations
from typing import Sequence
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QTableWidgetItem, QMessageBox, QAbstractItemView
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

from menipy.gui.viewmodels.plugins_vm import PluginsViewModel
from menipy.gui.services.settings_service import AppSettings

_COL_ACTIVE, _COL_NAME, _COL_KIND, _COL_PATH, _COL_DESC, _COL_VER = range(6)


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
        t.setColumnCount(6)
        t.setHorizontalHeaderLabels(
            ["Active", "Name", "Kind", "Path", "Description", "Version"]
        )
        # Use enums from QAbstractItemView for selection/edit behavior
        try:
            t.setSelectionBehavior(QAbstractItemView.SelectRows)  # older-style enum
            t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        except Exception:
            # Qt6 namespaced enums fallback
            t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setColumnWidth(_COL_ACTIVE, 70)
        t.setColumnWidth(_COL_NAME, 160)
        t.setColumnWidth(_COL_KIND, 90)
        t.setColumnWidth(_COL_PATH, 320)
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
