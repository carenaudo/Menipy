"""Controller for the setup panel widgets."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QLabel,
    QComboBox,
    QToolButton,
    QLineEdit,
    QCheckBox,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QMainWindow,
    QWidget,
    QMessageBox,
    QInputDialog,
)

from menipy.gui.views.image_view import DRAW_POINT, DRAW_LINE, DRAW_RECT
from menipy.gui.sop_controller import SopController

try:
    from menipy.pipelines.discover import PIPELINE_MAP
except Exception:
    PIPELINE_MAP = {}


class SetupPanelController(QObject):
    """Owns the "setup" panel widgets and exposes high-level signals/actions."""

    browse_requested = Signal()
    preview_requested = Signal()
    draw_mode_requested = Signal(int)
    clear_overlays_requested = Signal()
    run_all_requested = Signal()
    play_stage_requested = Signal(str)
    config_stage_requested = Signal(str)
    pipeline_changed = Signal(str)

    def __init__(
        self,
        window: QMainWindow,
        panel: QWidget,
        settings: Any,
        sops: Any,
        stage_order: Sequence[str],
        step_item_cls: Optional[type],
        pipeline_keys: Sequence[str],
    ) -> None:
        super().__init__(window)
        self.window = window
        self.panel = panel
        self.settings = settings
        self.sops = sops
        self.stage_order = list(stage_order)
        self.step_item_cls = step_item_cls
        self._pipeline_keys = {str(k).lower() for k in (pipeline_keys or PIPELINE_MAP.keys()) if k}

        self.testCombo: Optional[QComboBox] = panel.findChild(QComboBox, "testCombo")
        self.pipelineCombo: Optional[QComboBox] = panel.findChild(QComboBox, "pipelineCombo")
        self.sopCombo: Optional[QComboBox] = panel.findChild(QComboBox, "sopCombo")
        self.addSopBtn: Optional[QToolButton] = panel.findChild(QToolButton, "addSopBtn")

        self.imagePathEdit: Optional[QLineEdit] = panel.findChild(QLineEdit, "imagePathEdit")
        self.browseBtn: Optional[QToolButton] = panel.findChild(QToolButton, "browseBtn")
        self.previewBtn: Optional[QToolButton] = panel.findChild(QToolButton, "previewBtn")

        self.drawPointBtn: Optional[QPushButton] = panel.findChild(QPushButton, "drawPointBtn")
        self.drawLineBtn: Optional[QPushButton] = panel.findChild(QPushButton, "drawLineBtn")
        self.drawRectBtn: Optional[QPushButton] = panel.findChild(QPushButton, "drawRectBtn")
        self.clearOverlayBtn: Optional[QPushButton] = panel.findChild(QPushButton, "clearOverlayBtn")

        self.cameraCheck: Optional[QCheckBox] = panel.findChild(QCheckBox, "cameraCheck")
        self.cameraIdSpin: Optional[QSpinBox] = panel.findChild(QSpinBox, "cameraIdSpin")
        self.framesSpin: Optional[QSpinBox] = panel.findChild(QSpinBox, "framesSpin")

        self.stepsList: Optional[QListWidget] = panel.findChild(QListWidget, "stepsList")
        self.runAllBtn: Optional[QPushButton] = panel.findChild(QPushButton, "runAllBtn")

        self.sop_ctrl = SopController(
            window=self.window,
            sops=self.sops,
            stage_order=self.stage_order,
            step_item_cls=self.step_item_cls,
            steps_list=self.stepsList,
            sop_combo=self.sopCombo,
            pipeline_getter=self.current_pipeline_name,
            pipeline_changed_callback=lambda pipeline: self.pipeline_changed.emit(pipeline),
            play_callback=self.play_stage_requested.emit,
            config_callback=self.config_stage_requested.emit,
        )

        self._populate_pipeline_combo()
        self._restore_settings()
        self.sop_ctrl.initialize()
        self._wire_controls()
        self._initial_pipeline_refresh()

    # -------------------------- public API --------------------------

    def current_pipeline_name(self) -> Optional[str]:
        """Gets the canonical pipeline name from the combo box's user data."""
        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return None

        # Prefer the user data, which should hold the canonical key
        data = combo.currentData()
        if isinstance(data, str) and data.lower() in self._pipeline_keys:
            return data.lower()

        # Fallback to text if data is not set correctly
        text = combo.currentText().lower()
        return text if text in self._pipeline_keys else None

    def gather_run_params(self) -> dict[str, Any]:
        name = self.current_pipeline_name()
        use_camera = bool(self.cameraCheck.isChecked()) if self.cameraCheck else False
        frames = int(self.framesSpin.value()) if self.framesSpin else 1
        image = None if use_camera else (self.image_path() or None)
        cam_id = int(self.cameraIdSpin.value()) if (self.cameraIdSpin and use_camera) else None
        return {
            "name": name,
            "use_camera": use_camera,
            "frames": frames,
            "image": image,
            "cam_id": cam_id,
        }

    def image_path(self) -> Optional[str]:
        if self.imagePathEdit and hasattr(self.imagePathEdit, "text"):
            text = self.imagePathEdit.text().strip()
            return text or None
        return None

    def set_image_path(self, path: str) -> None:
        if self.imagePathEdit:
            self.imagePathEdit.setText(path)

    def set_camera_enabled(self, on: bool) -> None:
        if self.cameraCheck:
            self.cameraCheck.setChecked(bool(on))

    def collect_included_stages(self) -> list[str]:
        if hasattr(self, 'sop_ctrl'):
            return self.sop_ctrl.collect_included_stages()
        return list(self.stage_order)

    # -------------------------- internal helpers --------------------------

    def _populate_pipeline_combo(self) -> None:
        """Populates the pipeline combo box with discovered pipelines."""
        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return

        # Block signals to prevent premature updates while clearing/populating
        combo.blockSignals(True)
        combo.clear()

        # Use a more user-friendly name for the display text
        for key in sorted(list(self._pipeline_keys)):
            display_name = key.replace("_", " ").title()
            combo.addItem(display_name, userData=key)

        combo.blockSignals(False)

    def _restore_settings(self) -> None:
        if self.imagePathEdit and getattr(self.settings, "last_image_path", None):
            self.imagePathEdit.setText(self.settings.last_image_path)
        selected = getattr(self.settings, "selected_pipeline", None)
        combo = self.testCombo or self.pipelineCombo
        if combo and selected:
            index = combo.findData(selected)
            if index != -1:
                combo.setCurrentIndex(index)

    def _wire_controls(self) -> None:
        if self.browseBtn:
            self.browseBtn.clicked.connect(self.browse_requested.emit)
        if self.previewBtn:
            self.previewBtn.clicked.connect(self.preview_requested.emit)
        if self.drawPointBtn:
            self.drawPointBtn.clicked.connect(lambda: self.draw_mode_requested.emit(DRAW_POINT))
        if self.drawLineBtn:
            self.drawLineBtn.clicked.connect(lambda: self.draw_mode_requested.emit(DRAW_LINE))
        if self.drawRectBtn:
            self.drawRectBtn.clicked.connect(lambda: self.draw_mode_requested.emit(DRAW_RECT))
        if self.clearOverlayBtn:
            self.clearOverlayBtn.clicked.connect(self.clear_overlays_requested.emit)
        if self.runAllBtn:
            self.runAllBtn.clicked.connect(self.run_all_requested.emit)
        if self.addSopBtn:
            self.addSopBtn.clicked.connect(self.sop_ctrl.on_add_sop)
        combo = self.testCombo or self.pipelineCombo
        if combo:
            combo.currentTextChanged.connect(self.sop_ctrl.on_pipeline_changed)

    def _initial_pipeline_refresh(self) -> None:
        combo = self.testCombo or self.pipelineCombo
        if combo:
            self.sop_ctrl.on_pipeline_changed(combo.currentText())





