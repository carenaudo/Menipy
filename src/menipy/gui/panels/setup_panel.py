"""Controller for the setup panel widgets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QMainWindow,
    QWidget,
)

from menipy.gui.views.image_view import DRAW_POINT, DRAW_LINE, DRAW_RECT
from menipy.gui.sop_controller import SopController

try:
    from menipy.pipelines.discover import PIPELINE_MAP
except Exception:
    PIPELINE_MAP = {}


class SetupPanelController(QObject):
    """Owns the setup panel widgets and exposes high-level signals/actions."""

    MODE_SINGLE = "single"
    MODE_BATCH = "batch"
    MODE_CAMERA = "camera"

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    browse_requested = Signal()
    browse_batch_requested = Signal()
    preview_requested = Signal()
    analyze_requested = Signal()
    draw_mode_requested = Signal(int)
    clear_overlays_requested = Signal()
    run_all_requested = Signal()
    play_stage_requested = Signal(str)
    config_stage_requested = Signal(str)
    pipeline_changed = Signal(str)
    source_mode_changed = Signal(str)

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

        self.singleModeRadio: Optional[QRadioButton] = panel.findChild(QRadioButton, "singleModeRadio")
        self.batchModeRadio: Optional[QRadioButton] = panel.findChild(QRadioButton, "batchModeRadio")
        self.cameraModeRadio: Optional[QRadioButton] = panel.findChild(QRadioButton, "cameraModeRadio")

        self.imagePathEdit: Optional[QLineEdit] = panel.findChild(QLineEdit, "imagePathEdit")
        self.batchPathEdit: Optional[QLineEdit] = panel.findChild(QLineEdit, "batchPathEdit")
        self.sourceIdCombo: Optional[QComboBox] = panel.findChild(QComboBox, "sourceIdCombo")

        self.browseBtn: Optional[QToolButton] = panel.findChild(QToolButton, "browseBtn")
        self.batchBrowseBtn: Optional[QToolButton] = panel.findChild(QToolButton, "batchBrowseBtn")
        self.previewBtn: Optional[QToolButton] = panel.findChild(QToolButton, "previewBtn")

        self.drawPointBtn: Optional[QPushButton] = panel.findChild(QPushButton, "drawPointBtn")
        self.drawLineBtn: Optional[QPushButton] = panel.findChild(QPushButton, "drawLineBtn")
        self.drawRectBtn: Optional[QPushButton] = panel.findChild(QPushButton, "drawRectBtn")
        self.clearOverlayBtn: Optional[QPushButton] = panel.findChild(QPushButton, "clearOverlayBtn")

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

        self._mode_group = QButtonGroup(self)
        self._mode_map: dict[QRadioButton, str] = {}
        for btn, mode in (
            (self.singleModeRadio, self.MODE_SINGLE),
            (self.batchModeRadio, self.MODE_BATCH),
            (self.cameraModeRadio, self.MODE_CAMERA),
        ):
            if btn:
                self._mode_group.addButton(btn)
                self._mode_map[btn] = mode
        self._mode = self.MODE_SINGLE
        self._last_camera_id = "0"

        if self.singleModeRadio and not self.singleModeRadio.isChecked():
            self.singleModeRadio.setChecked(True)

        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(False)

        self._populate_pipeline_combo()
        self._restore_settings()
        self.sop_ctrl.initialize()
        self._wire_controls()
        self._apply_mode(self._mode, emit=False)
        self._initial_pipeline_refresh()

    # -------------------------- public API --------------------------

    def current_pipeline_name(self) -> Optional[str]:
        """Gets the canonical pipeline name from the combo box's user data."""
        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return None
        data = combo.currentData()
        if isinstance(data, str) and data.lower() in self._pipeline_keys:
            return data.lower()
        text = combo.currentText().lower()
        return text if text in self._pipeline_keys else None

    def current_mode(self) -> str:
        return self._mode

    def gather_run_params(self) -> dict[str, Any]:
        mode = self._mode
        frames = int(self.framesSpin.value()) if self.framesSpin else 1
        selected = self._selected_source_value()

        image: Optional[str] = None
        batch_folder: Optional[str] = None
        cam_id: Optional[int] = None

        if mode == self.MODE_CAMERA:
            if selected is not None:
                try:
                    cam_id = int(str(selected).strip())
                    self._last_camera_id = str(cam_id)
                except ValueError:
                    cam_id = None
            image = None
        elif mode == self.MODE_BATCH:
            batch_folder = self.batch_path()
            image = selected
        else:
            image = self.image_path() or selected

        return {
            "name": self.current_pipeline_name(),
            "mode": mode,
            "use_camera": mode == self.MODE_CAMERA,
            "frames": frames,
            "image": image,
            "batch_folder": batch_folder,
            "cam_id": cam_id,
        }

    def image_path(self) -> Optional[str]:
        if self.imagePathEdit and hasattr(self.imagePathEdit, "text"):
            text = self.imagePathEdit.text().strip()
            return text or None
        return None

    def batch_path(self) -> Optional[str]:
        if self.batchPathEdit and hasattr(self.batchPathEdit, "text"):
            text = self.batchPathEdit.text().strip()
            return text or None
        return None

    def set_image_path(self, path: str) -> None:
        if self.imagePathEdit:
            self.imagePathEdit.setText(path)
            self._refresh_source_items()

    def set_batch_path(self, path: str) -> None:
        if self.batchPathEdit:
            self.batchPathEdit.setText(path)
            self._refresh_source_items()

    def set_camera_enabled(self, on: bool) -> None:
        if on:
            if self.cameraModeRadio:
                self.cameraModeRadio.setChecked(True)
            self._apply_mode(self.MODE_CAMERA)
        elif self._mode == self.MODE_CAMERA:
            if self.singleModeRadio:
                self.singleModeRadio.setChecked(True)
            self._apply_mode(self.MODE_SINGLE)

    def collect_included_stages(self) -> list[str]:
        if hasattr(self, "sop_ctrl"):
            return self.sop_ctrl.collect_included_stages()
        return list(self.stage_order)

    # -------------------------- internal helpers --------------------------

    def _populate_pipeline_combo(self) -> None:
        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return
        combo.blockSignals(True)
        combo.clear()
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
        if self.batchBrowseBtn:
            self.batchBrowseBtn.clicked.connect(self.browse_batch_requested.emit)
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
        # The "Run All" button will now trigger the simple analysis.
        if self.runAllBtn:
            self.runAllBtn.setText("Analyze")
            self.runAllBtn.clicked.connect(self.analyze_requested.emit)
        if self.addSopBtn:
            self.addSopBtn.clicked.connect(self.sop_ctrl.on_add_sop)
        combo = self.testCombo or self.pipelineCombo
        if combo:
            combo.currentTextChanged.connect(self.sop_ctrl.on_pipeline_changed)
        if self.imagePathEdit:
            self.imagePathEdit.textChanged.connect(lambda _: self._refresh_source_items())
        if self.batchPathEdit:
            self.batchPathEdit.textChanged.connect(lambda _: self._refresh_source_items())
        if self.sourceIdCombo:
            self.sourceIdCombo.currentTextChanged.connect(self._on_combo_text_changed)
        self._mode_group.buttonToggled.connect(self._on_mode_toggled)

    def _initial_pipeline_refresh(self) -> None:
        combo = self.testCombo or self.pipelineCombo
        if combo:
            self.sop_ctrl.on_pipeline_changed(combo.currentText())

    def _on_mode_toggled(self, button: Optional[QRadioButton], checked: bool) -> None:
        if not checked or not button:
            return
        mode = self._mode_map.get(button)
        if mode:
            self._apply_mode(mode)

    def _apply_mode(self, mode: str, emit: bool = True) -> None:
        if mode not in {self.MODE_SINGLE, self.MODE_BATCH, self.MODE_CAMERA}:
            mode = self.MODE_SINGLE
        if mode == self._mode and not emit:
            pass
        elif mode == self._mode and emit:
            self.source_mode_changed.emit(mode)
        else:
            self._mode = mode
            self.source_mode_changed.emit(mode)
        self._update_widget_states()
        self._refresh_source_items()

    def _update_widget_states(self) -> None:
        single = self._mode == self.MODE_SINGLE
        batch = self._mode == self.MODE_BATCH
        camera = self._mode == self.MODE_CAMERA
        if self.imagePathEdit:
            self.imagePathEdit.setEnabled(single)
        if self.browseBtn:
            self.browseBtn.setEnabled(single)
        if self.batchPathEdit:
            self.batchPathEdit.setEnabled(batch)
        if self.batchBrowseBtn:
            self.batchBrowseBtn.setEnabled(batch)
        if self.framesSpin:
            self.framesSpin.setEnabled(camera)
        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(camera)
            self.sourceIdCombo.setEnabled(camera or single or batch)

    def _refresh_source_items(self) -> None:
        if self._mode == self.MODE_SINGLE:
            self._populate_single_selection()
        elif self._mode == self.MODE_BATCH:
            self._populate_batch_selection()
        else:
            self._populate_camera_selection()

    def _populate_single_selection(self) -> None:
        path = self.image_path()
        items = []
        if path:
            label = Path(path).name or path
            items.append((label, path))
        self._set_combo_items(items)

    def _populate_batch_selection(self) -> None:
        folder = self.batch_path()
        items = []
        if folder and Path(folder).is_dir():
            for child in sorted(Path(folder).iterdir()):
                if child.suffix.lower() in self._IMAGE_EXTENSIONS and child.is_file():
                    items.append((child.name, str(child)))
        self._set_combo_items(items)

    def _populate_camera_selection(self) -> None:
        if not self.sourceIdCombo:
            return
        combo = self.sourceIdCombo
        combo.blockSignals(True)
        combo.clear()
        combo.setEditable(True)
        combo.setEnabled(True)
        combo.setCurrentText(self._last_camera_id)
        combo.blockSignals(False)

    def _set_combo_items(self, items: list[tuple[str, str]]) -> None:
        if not self.sourceIdCombo:
            return
        combo = self.sourceIdCombo
        combo.blockSignals(True)
        combo.clear()
        combo.setEditable(False)
        for label, value in items:
            combo.addItem(label, userData=value)
        combo.setEnabled(bool(items))
        if items:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _selected_source_value(self) -> Optional[str]:
        if not self.sourceIdCombo:
            return None
        data = self.sourceIdCombo.currentData()
        if isinstance(data, str) and data:
            return data
        text = self.sourceIdCombo.currentText().strip()
        if not text:
            return None
        if self._mode == self.MODE_BATCH:
            folder = self.batch_path()
            if folder:
                candidate = Path(folder) / text
                return str(candidate)
        if self._mode == self.MODE_SINGLE:
            image = self.image_path()
            if image:
                return image
        return text

    def _on_combo_text_changed(self, text: str) -> None:
        if self._mode == self.MODE_CAMERA:
            self._last_camera_id = text.strip() or "0"
