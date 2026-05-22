"""Pipeline step test panel for scientific workflow mode."""

from __future__ import annotations

from typing import Any, Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class PipelineStepTestPanel(QWidget):
    """Compact left-rail panel for testing pipeline stages with sandbox settings."""

    runRequested = Signal(str)
    editRequested = Signal(str)
    applyRequested = Signal()
    discardRequested = Signal()
    stageChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pipelineStepTestPanel")
        self._stage_names: list[str] = []
        self._dirty = False
        self._build_ui()
        self._wire_signals()

    def set_stages(self, stages: Iterable[str]) -> None:
        current = self.current_stage()
        self._stage_names = [stage for stage in stages if stage != "acquisition"]
        self.stageList.blockSignals(True)
        try:
            self.stageList.clear()
            for stage in self._stage_names:
                item = QListWidgetItem(stage.replace("_", " ").title())
                item.setData(Qt.UserRole, stage)
                item.setToolTip(stage)
                self.stageList.addItem(item)
            next_row = 0
            if current in self._stage_names:
                next_row = self._stage_names.index(current)
            if self.stageList.count():
                self.stageList.setCurrentRow(next_row)
        finally:
            self.stageList.blockSignals(False)
        self._on_stage_selection_changed()

    def current_stage(self) -> str | None:
        item = self.stageList.currentItem()
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        return str(data) if data else None

    def set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        self.applyBtn.setEnabled(self._dirty)
        self.discardBtn.setEnabled(self._dirty)
        suffix = "Unsaved test settings" if self._dirty else "Sandbox is clean"
        self.sandboxStatusLabel.setText(suffix)

    def set_stage_help(self, text: str) -> None:
        self.configInfo.setText(text)

    def set_status(self, text: str) -> None:
        self.statusLabel.setText(text)

    def set_output(self, text: str) -> None:
        self.outputText.setPlainText(text)

    def append_output(self, text: str) -> None:
        current = self.outputText.toPlainText()
        self.outputText.setPlainText(f"{current}\n{text}".strip())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("Step Test")
        title.setObjectName("pipelineStepTestTitle")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        root.addWidget(title)

        self.statusLabel = QLabel("Select a stage to test.")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setObjectName("pipelineStepTestStatus")
        root.addWidget(self.statusLabel)

        self.stageList = QListWidget(self)
        self.stageList.setObjectName("pipelineStepTestStageList")
        self.stageList.setSelectionMode(QListWidget.SingleSelection)
        self.stageList.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stageList.setMinimumHeight(170)
        self.stageList.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        root.addWidget(self.stageList)

        config_frame = QFrame(self)
        config_frame.setObjectName("pipelineStepTestConfigFrame")
        config_layout = QVBoxLayout(config_frame)
        config_layout.setContentsMargins(8, 8, 8, 8)
        config_layout.setSpacing(6)
        self.configInfo = QLabel("")
        self.configInfo.setWordWrap(True)
        config_layout.addWidget(self.configInfo)
        self.editConfigBtn = QPushButton("Edit Test Config")
        self.editConfigBtn.setObjectName("pipelineStepTestEditConfigBtn")
        config_layout.addWidget(self.editConfigBtn)
        root.addWidget(config_frame)

        action_row = QHBoxLayout()
        self.runBtn = QPushButton("Run Stage")
        self.runBtn.setObjectName("pipelineStepTestRunBtn")
        self.applyBtn = QPushButton("Apply")
        self.applyBtn.setObjectName("pipelineStepTestApplyBtn")
        self.discardBtn = QPushButton("Discard")
        self.discardBtn.setObjectName("pipelineStepTestDiscardBtn")
        action_row.addWidget(self.runBtn)
        action_row.addWidget(self.applyBtn)
        action_row.addWidget(self.discardBtn)
        root.addLayout(action_row)

        self.sandboxStatusLabel = QLabel("Sandbox is clean")
        self.sandboxStatusLabel.setObjectName("pipelineStepTestSandboxStatus")
        root.addWidget(self.sandboxStatusLabel)

        self.outputText = QPlainTextEdit(self)
        self.outputText.setObjectName("pipelineStepTestOutput")
        self.outputText.setReadOnly(True)
        self.outputText.setPlaceholderText("Run diagnostics will appear here.")
        root.addWidget(self.outputText, 1)

        self.setStyleSheet("""
            QWidget#pipelineStepTestPanel {
                background: #ffffff;
            }
            QFrame#pipelineStepTestConfigFrame {
                border: 1px solid #d7dde5;
                border-radius: 6px;
                background: #f8fafc;
            }
            QListWidget {
                border: 1px solid #d7dde5;
                border-radius: 6px;
                background: #ffffff;
            }
            QListWidget::item {
                padding: 6px;
            }
            QListWidget::item:selected {
                background: #ddf4ff;
                color: #0969da;
            }
            QPushButton {
                min-height: 28px;
            }
        """)
        self.set_dirty(False)

    def _wire_signals(self) -> None:
        self.stageList.currentRowChanged.connect(
            lambda _row: self._on_stage_selection_changed()
        )
        self.runBtn.clicked.connect(self._emit_run)
        self.editConfigBtn.clicked.connect(self._emit_edit)
        self.applyBtn.clicked.connect(self.applyRequested.emit)
        self.discardBtn.clicked.connect(self.discardRequested.emit)

    def _on_stage_selection_changed(self) -> None:
        stage = self.current_stage()
        has_stage = bool(stage)
        self.runBtn.setEnabled(has_stage)
        self.editConfigBtn.setEnabled(has_stage)
        if stage:
            self.stageChanged.emit(stage)

    def _emit_run(self) -> None:
        stage = self.current_stage()
        if stage:
            self.runRequested.emit(stage)

    def _emit_edit(self) -> None:
        stage = self.current_stage()
        if stage:
            self.editRequested.emit(stage)
