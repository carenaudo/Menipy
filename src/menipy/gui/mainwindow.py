# src/adsa/gui/mainwindow.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QCloseEvent, QAction
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QDockWidget, QWidget, QFileDialog,
    QMessageBox, QTableWidgetItem, QLabel, QTableWidget,QMenuBar
)
from PySide6.QtUiTools import QUiLoader

# If you have compiled resources, import them before setupUi:
# import resources_rc

#  Generated from: pyside6-uic main_window.ui -o ui_main_window.py
from .views.ui_main_window import Ui_MainWindow

# App services / viewmodels
from ..common.plugin_db import PluginDB
from ..common.plugins import discover_into_db, load_active_plugins
from .services.settings_service import AppSettings
from .services.pipeline_runner import PipelineRunner
from .viewmodels.run_vm import RunViewModel
from .viewmodels.plugins_vm import PluginsViewModel
from .services.image_convert import to_pixmap
from .dialogs.plugin_manager_dialog import PluginManagerDialog

_COL_ACTIVE, _COL_NAME, _COL_KIND, _COL_PATH, _COL_DESC, _COL_VER = range(6)



class MainWindow(QMainWindow, Ui_MainWindow):
    """
    SetupUi-based main window:
      - Calls setupUi(self) to populate THIS QMainWindow from Designer.
      - Embeds run/overlay/results panels into QWidget placeholders.
      - Adds plugin_panel.ui as a dock widget.
      - Wires RunViewModel and PluginsViewModel + persists settings.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menipy GUI")

        # -------- Settings ----------
        self.settings = AppSettings.load()

        # -------- Build main window from Designer ----------
        self.setupUi(self)  # <-- everything from main_window.ui is now on self.*

        # Ensure placeholders have layouts to receive child panels
        self._ensure_placeholder_layout(self.runHost)
        self._ensure_placeholder_layout(self.overlayHost)
        self._ensure_placeholder_layout(self.resultsHost)


        # -------- Loader helpers for sub-panels ----------
        loader = QUiLoader()
        views_dir = Path(__file__).resolve().parent / "views"

        def load_ui(res_path: str, fallback_filename: str) -> QWidget:
            f = QFile(res_path)
            if not f.exists():
                f = QFile(str(views_dir / fallback_filename))
            f.open(QFile.ReadOnly)
            w = loader.load(f, self)
            f.close()
            if w is None:
                raise RuntimeError(f"Failed to load sub-UI: {res_path} / {fallback_filename}")
            return w


        def embed(child: QWidget, host: QWidget):
            lay = host.layout() or QVBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(child)

        # -------- Load/Embed panels ----------
        self.run_panel = load_ui(":/views/run_panel.ui", "run_panel.ui")
        self.overlay_panel = load_ui(":/views/overlay_panel.ui", "overlay_panel.ui")
        self.results_panel = load_ui(":/views/results_panel.ui", "results_panel.ui")

        embed(self.run_panel, self.runHost)
        embed(self.overlay_panel, self.overlayHost)
        embed(self.results_panel, self.resultsHost)

        # -------- Plugin dock ----------
        self.plugin_panel = load_ui(":/views/plugin_panel.ui", "plugin_panel.ui")
        self.plugin_dock = QDockWidget("Plugins", self)
        self.plugin_dock.setObjectName("pluginDock")
        self.plugin_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        )
        self.plugin_dock.setWidget(self.plugin_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plugin_dock)
        self.plugin_dock.hide()  # hidden by default; toggle via View menu

        # -------- Services / ViewModels ----------
        self.db = PluginDB()
        self.db.init_schema()
        # Discover from settings plugin dirs (defaults to ["./plugins"])
        self._plugins_discover(self.settings.plugin_dirs)
        # Load only active ones into registry
        load_active_plugins(self.db)

        self.runner = PipelineRunner()
        self.run_vm = RunViewModel(self.runner)
        self.plugins_vm = PluginsViewModel(self.db)

        # -------- Wire panels ----------
        self._wire_run_panel()
        self._wire_overlay_panel()
        self._wire_results_panel()
        self._wire_plugin_panel()


        # Toggles for embedded panels (run/overlay/results)
        # Initial states reflect current visibility
        self.action_plugins.setChecked(self.plugin_dock.isVisible())
        self.action_preview.setChecked(self.overlayHost.isVisible())
        self.action_case.setChecked(self.runHost.isVisible())
        self.action_results.setChecked(self.resultsHost.isVisible())

        # Wire toggles -> show/hide the corresponding panels
        self.action_plugins.toggled.connect(self.plugin_dock.setVisible)
        self.action_preview.toggled.connect(self.overlayHost.setVisible)
        self.action_case.toggled.connect(self.runHost.setVisible)
        self.action_results.toggled.connect(self.resultsHost.setVisible)

        # Keep Plugins action in sync if user closes the dock via the [x] on the dock
        self.plugin_dock.visibilityChanged.connect(self.action_plugins.setChecked)

        # Initial refresh of plugin table
        self._refresh_plugin_table()

    # ---------------- Helpers ----------------
    def _ensure_placeholder_layout(self, host: QWidget) -> None:
        if host.layout() is None:
            lay = QVBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)

    # ======================================================================
    # Run panel
    # ======================================================================
    def _wire_run_panel(self):
        # Widgets from run_panel.ui
        self.pipelineCombo = self.run_panel.findChild(QWidget, "pipelineCombo")
        self.imagePathEdit = self.run_panel.findChild(QWidget, "imagePathEdit")
        self.browseButton = self.run_panel.findChild(QWidget, "browseButton")
        self.cameraCheck = self.run_panel.findChild(QWidget, "cameraCheck")
        self.framesSpin = self.run_panel.findChild(QWidget, "framesSpin")
        self.runButton = self.run_panel.findChild(QWidget, "runButton")

        # Restore settings
        if self.pipelineCombo and hasattr(self.pipelineCombo, "setCurrentText"):
            self.pipelineCombo.setCurrentText(self.settings.selected_pipeline)
        if self.imagePathEdit and self.settings.last_image_path:
            self.imagePathEdit.setText(self.settings.last_image_path)

        # Browse button
        if self.browseButton and hasattr(self.browseButton, "clicked"):
            self.browseButton.clicked.connect(self._browse_image)

        # Run button
        if self.runButton and hasattr(self.runButton, "clicked"):
            self.runButton.clicked.connect(self._on_run_clicked)

        # Bind VM signals
        self.run_vm.preview_ready.connect(self._on_preview_ready)
        self.run_vm.results_ready.connect(self._on_results_ready)
        self.run_vm.error_occurred.connect(self._on_pipeline_error)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.imagePathEdit.setText(path)

    def _on_run_clicked(self):
        pipeline = (self.pipelineCombo.currentText()
                    if hasattr(self.pipelineCombo, "currentText") else "sessile")
        use_camera = bool(self.cameraCheck.isChecked()) if hasattr(self.cameraCheck, "isChecked") else False
        frames = int(self.framesSpin.value()) if hasattr(self.framesSpin, "value") else 1
        image = None if use_camera else (self.imagePathEdit.text().strip() if self.imagePathEdit else None)
        camera = 0 if use_camera else None

        # Persist settings
        self.settings.selected_pipeline = pipeline
        self.settings.last_image_path = image
        self.settings.save()

        # Execute
        self.statusBar().showMessage("Running pipeline…")
        self.run_vm.run(pipeline=pipeline, image=image, camera=camera, frames=frames)

    # ======================================================================
    # Overlay panel
    # ======================================================================
    def _wire_overlay_panel(self):
        self.previewLabel: QLabel = self.overlay_panel.findChild(QLabel, "previewLabel")  # type: ignorere

    def _on_preview_ready(self, pixmap):
        if self.previewLabel:
            self.previewLabel.setPixmap(pixmap)
        self.statusBar().showMessage("Preview updated", 2000)

    # ======================================================================
    # Results panel
    # ======================================================================
    def _wire_results_panel(self):
        self.paramsTable: QTableWidget = self.results_panel.findChild(QTableWidget, "paramsTable")  # type: ignore
        self.residualsTable: QTableWidget = self.results_panel.findChild(QTableWidget, "residualsTable")  # type: ignore
        self.timingsTable: QTableWidget = self.results_panel.findChild(QTableWidget, "timingsTable")  # type: ignore


        # Minimal table setup
        if self.paramsTable:
            self.paramsTable.setColumnCount(2)
            self.paramsTable.setHorizontalHeaderLabels(["Name", "Value"])
            self.paramsTable.horizontalHeader().setStretchLastSection(True)
        if self.residualsTable:
            self.residualsTable.setColumnCount(2)
            self.residualsTable.setHorizontalHeaderLabels(["Metric", "Value"])
            self.residualsTable.horizontalHeader().setStretchLastSection(True)
        if self.timingsTable:
            self.timingsTable.setColumnCount(2)
            self.timingsTable.setHorizontalHeaderLabels(["Stage", "ms"])
            self.timingsTable.horizontalHeader().setStretchLastSection(True)

    def _on_results_ready(self, results: dict):
        # Params
        if self.paramsTable:
            self._fill_kv_table(self.paramsTable, results)

        # Residuals
        residuals = results.get("residuals") or {}
        if self.residualsTable:
            self._fill_kv_table(self.residualsTable, residuals)

        # Timings: ask VM? In our design, RunViewModel doesn't emit timings;
        # If you want timings, you can extend RunViewModel to emit ctx.timings_ms too.
        # For now, leave as-is or clear.
        if self.timingsTable:
            self.timingsTable.setRowCount(0)

        gamma = results.get("gamma_mN_per_m")
        if gamma is not None:
            self.statusBar().showMessage(f"γ ≈ {gamma} mN/m", 4000)
        else:
            self.statusBar().showMessage("Results updated", 2000)

    def _fill_kv_table(self, table: QTableWidget, mapping: dict):
        table.setRowCount(0)
        for k, v in mapping.items():
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(str(k)))
            table.setItem(r, 1, QTableWidgetItem(str(v)))

    def _on_pipeline_error(self, msg: str):
        QMessageBox.critical(self, "Pipeline Error", msg)
        self.statusBar().showMessage("Error", 3000)
    # ======================================================================
    # Plugin panel (dock)
    # ======================================================================
    def _wire_plugin_panel(self):
        # Controls from plugin_panel.ui
        self.pluginsTable: QTableWidget = self.plugin_panel.findChild(QTableWidget, "pluginsTable")  # type: ignore
        self.dirsEdit: QWidget = self.plugin_panel.findChild(QWidget, "dirsEdit")
        self.rescanButton: QWidget = self.plugin_panel.findChild(QWidget, "rescanButton")
        self.activateAllButton: QWidget = self.plugin_panel.findChild(QWidget, "activateAllButton")
        self.deactivateAllButton: QWidget = self.plugin_panel.findChild(QWidget, "deactivateAllButton")

        # Init dirs from settings
        if self.dirsEdit and hasattr(self.dirsEdit, "setText"):
            self.dirsEdit.setText(self._dirs_as_text(self.settings.plugin_dirs))

        # Table setup
        if self.pluginsTable:
            self.pluginsTable.setColumnCount(6)
            self.pluginsTable.setHorizontalHeaderLabels(["Active", "Name", "Kind", "Path", "Description", "Version"])
            self.pluginsTable.setColumnWidth(_COL_ACTIVE, 70)
            self.pluginsTable.setColumnWidth(_COL_NAME, 160)
            self.pluginsTable.setColumnWidth(_COL_KIND, 90)
            self.pluginsTable.setColumnWidth(_COL_PATH, 320)
            self.pluginsTable.horizontalHeader().setStretchLastSection(True)
            self.pluginsTable.cellChanged.connect(self._on_plugin_cell_changed)

        # Buttons
        if self.rescanButton and hasattr(self.rescanButton, "clicked"):
            self.rescanButton.clicked.connect(self._on_plugins_rescan)
        if self.activateAllButton and hasattr(self.activateAllButton, "clicked"):
            self.activateAllButton.clicked.connect(lambda: self._activate_all_plugins(True))
        if self.deactivateAllButton and hasattr(self.deactivateAllButton, "clicked"):
            self.deactivateAllButton.clicked.connect(lambda: self._activate_all_plugins(False))

    def _dirs_as_text(self, dirs: Sequence[str]) -> str:
        return ":".join(dirs) if dirs else "./plugins"

    def _parse_dirs(self, text: str) -> list[str]:
        return [p.strip() for p in text.replace(";", ":").split(":") if p.strip()] or ["./plugins"]

    def _plugins_discover(self, dirs: Sequence[str | Path]):
        discover_into_db(self.db, [Path(d) for d in dirs])

    def _refresh_plugin_table(self):
        if not hasattr(self, "pluginsTable") or not self.pluginsTable:
            return
        rows = self.plugins_vm.rows()
        t = self.pluginsTable
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, (name, kind, path, entry, desc, ver, active) in enumerate(rows):
            # Active checkbox
            item_active = QTableWidgetItem()
            item_active.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item_active.setCheckState(Qt.Checked if int(active) else Qt.Unchecked)
            # store name/kind for easy toggle retrieval
            item_active.setData(Qt.UserRole, (name, kind))
            t.setItem(i, _COL_ACTIVE, item_active)

            t.setItem(i, _COL_NAME, QTableWidgetItem(str(name)))
            t.setItem(i, _COL_KIND, QTableWidgetItem(str(kind)))
            t.setItem(i, _COL_PATH, QTableWidgetItem(str(path)))
            t.setItem(i, _COL_DESC, QTableWidgetItem(str(desc or "")))
            t.setItem(i, _COL_VER, QTableWidgetItem(str(ver or "")))
        t.blockSignals(False)

    def _on_plugins_rescan(self):
        # Discover from dirs, then reload active into registry
        dirs_text = self.dirsEdit.text().strip() if hasattr(self.dirsEdit, "text") else ""
        dirs = self._parse_dirs(dirs_text)
        try:
            self._plugins_discover(dirs)
            load_active_plugins(self.db) # (re)load active into registry
            # Persist dirs in settings
            self.settings.plugin_dirs = list(dirs)
            self.settings.save()
            self._refresh_plugin_table()
            self.statusBar().showMessage("Plugins rescanned", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Plugin Rescan Failed", str(e))

    def _activate_all_plugins(self, active: bool):
        t = self.pluginsTable
        if not t:
            return
        # Flip UI checkboxes; cellChanged will trigger toggles
        for r in range(t.rowCount()):
            itm = t.item(r, _COL_ACTIVE)
            if itm:
                itm.setCheckState(Qt.Checked if active else Qt.Unchecked)

    def _on_plugin_cell_changed(self, row: int, col: int):
        if col != _COL_ACTIVE:
            return
        t = self.pluginsTable
        item = t.item(row, _COL_ACTIVE)
        if not item:
            return
        active = item.checkState() == Qt.Checked
        name = t.item(row, _COL_NAME).text()
        kind = t.item(row, _COL_KIND).text()
        try:
            self.plugins_vm.toggle(name, kind, active)
            self.statusBar().showMessage(f"{'Activated' if active else 'Deactivated'} {name}:{kind}", 1500)
        except Exception as e:
            QMessageBox.critical(self, "Toggle failed", str(e))
            # revert checkbox
            t.blockSignals(True)
            item.setCheckState(Qt.Unchecked if active else Qt.Checked)
            t.blockSignals(False)


    # ======================================================================
    # Plugin Manager dialog (optional, via button)
    # ======================================================================
    def _open_plugin_manager(self):
        dlg = PluginManagerDialog(self.plugins_vm, self.settings, self)
        dlg.exec()
        # After dialog closes, refresh panel table (in case of changes)
        self._refresh_plugin_table()

    # ======================================================================
    # Lifecycle
    # ======================================================================
    def closeEvent(self, event: QCloseEvent) -> None:
        # Persist last used pipeline and image path on window close
        if hasattr(self, "pipelineCombo") and hasattr(self.pipelineCombo, "currentText"):
            self.settings.selected_pipeline = self.pipelineCombo.currentText()
        if hasattr(self, "imagePathEdit") and hasattr(self.imagePathEdit, "text"):
            txt = self.imagePathEdit.text().strip()
            self.settings.last_image_path = txt or None
        self.settings.save()
        super().closeEvent(event)

