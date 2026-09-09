"""Workspace preferences backed by working application settings."""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from menipy.gui.services.workspace_preferences import WorkspacePreferences


class SettingsDialog(QDialog):
    """Import/reset edit a draft; Apply persists it using the window's settings."""

    def __init__(self, parent):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Workspace Preferences")
        self.resize(570, 440)
        layout = QVBoxLayout(self)
        scope = QLabel(
            "Import/export covers these display/history settings and result-column visibility. Analysis presets, sources, calibration, plugin settings and window layout are managed separately. Appearance buttons below open separate editors with their own Save controls."
        )
        scope.setWordWrap(True)
        layout.addWidget(scope)
        form = QFormLayout()
        self.units = QComboBox()
        self.units.addItems(["SI", "CGS"])
        form.addRow("Display units", self.units)
        self.checks = {}
        for key, label in (
            ("show_mode_labels", "Always show analysis mode labels"),
                ("compare_methods_visible", "Show method comparisons by default"),
                ("diagnostics_visible", "Show result diagnostics by default"),
        ):
            checkbox = QCheckBox(label)
            self.checks[key] = checkbox
            form.addRow(checkbox)
        self.limit = QSpinBox()
        self.limit.setRange(10, 1000)
        form.addRow("Active history limit", self.limit)
        layout.addLayout(form)
        note = QLabel(
            "On the next measurement, older records are archived as complete JSON before leaving active history. Export all history includes active records only. Archives are kept until you remove them. Larger limits can slow table refresh."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.archive_path = parent.results_panel_ctrl.history.archive_directory
        archives = QPushButton("Open history archives")
        archives.setToolTip(str(self.archive_path))
        archives.clicked.connect(self.open_archives)
        layout.addWidget(archives)
        appearance = QHBoxLayout()
        for text, callback in (
            ("Overlay appearance…", parent.main_controller.open_overlay),
            ("Markers and labels…", parent.main_controller.open_marker_config),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            appearance.addWidget(button)
        layout.addLayout(appearance)
        operations = QHBoxLayout()
        for text, callback in (
            ("Import…", self.import_file),
            ("Export draft…", self.export_file),
            ("Reset draft to defaults", lambda: self.load(WorkspacePreferences())),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            operations.addWidget(button)
        layout.addLayout(operations)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.apply
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.load(WorkspacePreferences.capture(parent.settings))

    def load(self, draft):
        self.draft = draft
        self.units.setCurrentText(draft.unit_system)
        self.limit.setValue(draft.history_limit)
        for key, widget in self.checks.items():
            widget.setChecked(getattr(draft, key))

    def values(self):
        return WorkspacePreferences(
            unit_system=self.units.currentText(),
            history_limit=self.limit.value(),
            results_hidden_columns=self.draft.results_hidden_columns,
            **{key: widget.isChecked() for key, widget in self.checks.items()},
        )

    def apply(self):
        try:
            self.values().apply(self.window.settings)
        except OSError as exc:
            QMessageBox.warning(self, "Preferences not saved", str(exc))
            return
        self.window.results_panel_ctrl.history.max_history = (
            self.window.settings.history_limit
        )
        self.window.setup_panel_ctrl._sync_pipeline_button_presentation()
        for action in self.window.menuConfig.actions():
            if action.text() == "Show analysis mode labels":
                action.blockSignals(True)
                action.setChecked(self.window.settings.show_mode_labels)
                action.blockSignals(False)
        self.window.main_controller.refresh_unit_labels()
        for action in self.window.findChildren(QAction):
            if action.isCheckable() and action.data() in ("SI", "CGS"):
                action.setChecked(action.data() == self.window.settings.unit_system)
        panel = self.window.results_panel_ctrl
        panel._sync_helper_buttons()
        panel.update_history(activate=False)
        self.window.statusBar().showMessage("Workspace preferences saved", 3000)

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import workspace preferences", "", "JSON (*.json)"
        )
        if path:
            try:
                self.load(
                    WorkspacePreferences.model_validate_json(
                        Path(path).read_text(encoding="utf-8")
                    )
                )
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Preferences not imported", str(exc))

    def export_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export draft workspace preferences", "", "JSON (*.json)"
        )
        if path:
            try:
                Path(path).write_text(
                    self.values().model_dump_json(indent=2), encoding="utf-8"
                )
            except OSError as exc:
                QMessageBox.warning(self, "Preferences not exported", str(exc))

    def open_archives(self):
        try:
            self.archive_path.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.archive_path)))
        except OSError as exc:
            QMessageBox.warning(self, "Cannot open archives", str(exc))
