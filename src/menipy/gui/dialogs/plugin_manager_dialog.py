from __future__ import annotations
from typing import Sequence
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QTableWidgetItem, QMessageBox
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
            ui_file = QFile(str(Path(__file__).resolve().parent.parent / "views" / "plugin_manager.ui"))
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        self.setLayout(self.ui.layout())

        # Wire controls
        self.ui.closeButton.clicked.connect(self.accept)
        self.ui.rescanButton.clicked.connect(self._on_rescan)
        self.ui.activateAllButton.clicked.connect(lambda: self._activate_all(True))
        self.ui.deactivateAllButton.clicked.connect(lambda: self._activate_all(False))

        # Init dirs line edit
        self.ui.dirsEdit.setText(self._dirs_as_text(self.settings.plugin_dirs))

        # Table setup
        t = self.ui.pluginsTable
        t.setColumnCount(6)
        t.setHorizontalHeaderLabels(["Active", "Name", "Kind", "Path", "Description", "Version"])
        t.setSelectionBehavior(t.SelectRows)
        t.setEditTriggers(t.NoEditTriggers)
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
