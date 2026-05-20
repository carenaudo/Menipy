"""Advanced workflow dialog for SOP and pipeline-stage controls."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QGroupBox, QLabel, QVBoxLayout


class AdvancedWorkflowDialog(QDialog):
    """Modeless dialog that hosts the guided workbench's advanced controls."""

    def __init__(
        self,
        parent=None,
        sop_group: QGroupBox | None = None,
        steps_group: QGroupBox | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("advancedWorkflowDialog")
        self.setWindowTitle("Advanced Workflow")
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(460, 560)
        self.sop_group = sop_group
        self.steps_group = steps_group

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        if self.sop_group is not None:
            self.sop_group.setTitle("SOP Controls")
            layout.addWidget(self.sop_group)
        else:
            layout.addWidget(QLabel("SOP controls are unavailable.", self))

        if self.steps_group is not None:
            self.steps_group.setTitle("Pipeline Stages")
            layout.addWidget(self.steps_group, 1)
        else:
            layout.addWidget(QLabel("Pipeline stages are unavailable.", self), 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def show_controls(self) -> None:
        for widget in (self.sop_group, self.steps_group):
            if widget is not None:
                widget.setVisible(True)

    def showEvent(self, event) -> None:
        self.show_controls()
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        for widget in (self.sop_group, self.steps_group):
            if widget is not None:
                widget.setVisible(False)
        super().closeEvent(event)
