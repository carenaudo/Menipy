"""SOP management helper for Menipy GUI."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QInputDialog,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QWidget,
)

from menipy.pipelines.base import canonical_stage_name

try:
    from menipy.gui.services.sop_service import Sop
except Exception:
    Sop = None  # type: ignore


class SopController:
    """Encapsulates SOP list management and step widgets.

    The step list shows the selected pipeline's real stages. Required stages
    always run; optional stages (``PipelineBase.OPTIONAL_STAGES``) carry an
    include checkbox, and Run Analysis leaves the unticked ones out.
    ``stage_order`` is only used for pipelines without a known class.
    """

    def __init__(
        self,
        window,
        sops: Any,
        stage_order: Sequence[str],
        step_item_cls: type | None,
        steps_list: QListWidget | None,
        sop_combo: QComboBox | None,
        pipeline_getter: Callable[[], str | None],
        pipeline_changed_callback: Callable[[str], None],
        play_callback: Callable[[str], None],
        config_callback: Callable[[str], None],
        pipeline_map: Mapping[str, type] | None = None,
    ) -> None:
        self.pipeline_map = {str(k).lower(): v for k, v in (pipeline_map or {}).items()}
        self.window = window
        self.sops = sops
        self.stage_order = list(stage_order)
        self.step_item_cls = step_item_cls
        self.steps_list = steps_list
        self.sop_combo = sop_combo
        self.pipeline_getter = pipeline_getter
        self.pipeline_changed_callback = pipeline_changed_callback
        self.play_callback = play_callback
        self.config_callback = config_callback

        self._step_widgets: list[Any] = []
        self._sop_combo_connected = False

    def initialize(self) -> None:
        self._populate_steps_list()
        self._refresh_sop_combo()
        self._apply_selected_sop()

    def on_pipeline_changed(self, _name: str) -> None:
        self._populate_steps_list()  # Stages differ between pipelines.
        if not self.sops:
            return
        pipeline = self.pipeline_getter() or "sessile"
        try:
            self.sops.ensure_default(pipeline, self.stages_for(pipeline))
        except Exception as exc:
            print("[SOP] pipeline change:", exc)
        self._refresh_sop_combo()
        self._apply_selected_sop()
        if pipeline:
            try:
                self.pipeline_changed_callback(pipeline)
            except Exception:
                pass

    def stages_for(self, pipeline: str | None) -> list[str]:
        """Stages of ``pipeline`` in run order.

        Parameters
        ----------
        pipeline : str or None
            Pipeline name.

        Returns
        -------
        list of str
            Canonical stage names, or ``stage_order`` for an unknown pipeline.
        """
        cls = self.pipeline_map.get(str(pipeline or "").lower())
        names = getattr(cls, "stage_names", None)
        return list(names()) if callable(names) else list(self.stage_order)

    def optional_stages_for(self, pipeline: str | None) -> set[str]:
        """Stages of ``pipeline`` that a run may leave out."""
        cls = self.pipeline_map.get(str(pipeline or "").lower())
        return set(getattr(cls, "OPTIONAL_STAGES", ()))

    def apply_included_stages(self, stages: Sequence[str]) -> None:
        """Tick the optional steps listed in ``stages`` and untick the others.

        Parameters
        ----------
        stages : sequence of str
            Included stages as saved in a SOP or preset; legacy names are
            translated and required steps stay included.
        """
        chosen = {canonical_stage_name(stage) for stage in stages}
        for widget in self._step_widgets:
            step_name = getattr(widget, "step_name", None)
            included = step_name in chosen
            try:
                if hasattr(widget, "set_included"):
                    widget.set_included(included)
                else:
                    widget.setEnabled(included)
                if not included:
                    widget.set_status("pending")
            except Exception:
                pass

    def collect_included_stages(self) -> list[str]:
        if not self._step_widgets:
            return self.stages_for(self.pipeline_getter())
        included: list[str] = []
        for widget in self._step_widgets:
            step_name = getattr(widget, "step_name", None)
            if not step_name:
                continue
            try:
                if hasattr(widget, "is_included"):
                    is_included = widget.is_included()
                else:
                    is_included = widget.isEnabled()
                if is_included:
                    included.append(step_name)
            except Exception:
                included.append(step_name)
        return included

    def on_add_sop(self) -> None:
        """Add on  sop."""
        if not self.sops:
            QMessageBox.warning(self.window, "SOP", "SOP service is not available.")
            return
        name, ok = QInputDialog.getText(self.window, "Add SOP", "SOP name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        include = self.collect_included_stages()
        pipeline_key = self.pipeline_getter() or "sessile"
        try:
            preset_controller = getattr(self.window, "preset_ctrl", None)
            if preset_controller is not None:
                if self.sops.get(pipeline_key, name):
                    raise ValueError(
                        "This name already exists. Select it and use Update."
                    )
                notes, ok = QInputDialog.getMultiLineText(
                    self.window,
                    "Add SOP",
                    "Notes (optional):",
                    preset_controller.legacy_notes(),
                )
                if not ok:
                    return
                preset_controller.save(preset_controller.capture(name, notes))
                preset_controller.forget_legacy_notes()
                return
            if Sop is not None:
                sop_obj = Sop(name=name, include_stages=include, params={})
                self.sops.upsert(pipeline_key, sop_obj)
            else:
                sop_like = type(
                    "SopLike",
                    (),
                    {"name": name, "include_stages": include, "params": {}},
                )
                self.sops.upsert(pipeline_key, sop_like)
        except Exception as exc:
            QMessageBox.critical(self.window, "SOP", f"Could not save SOP:\n{exc}")
            return
        self._refresh_sop_combo(select=name)
        self._apply_selected_sop()
        try:
            self.window.statusBar().showMessage(f"SOP '{name}' added", 1500)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------

    def _populate_steps_list(self) -> None:
        self._step_widgets.clear()
        if not self.steps_list:
            return
        self.steps_list.clear()
        self.steps_list.setFlow(QListView.TopToBottom)
        self.steps_list.setWrapping(False)
        self.steps_list.setResizeMode(QListView.Adjust)
        self.steps_list.setMovement(QListView.Static)
        self.steps_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.steps_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.steps_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.steps_list.setUniformItemSizes(False)
        pipeline = self.pipeline_getter()
        stages = self.stages_for(pipeline)
        optional = self.optional_stages_for(pipeline)
        if not self.step_item_cls:
            for stage in stages:
                self.steps_list.addItem(stage)
            return
        for stage in stages:
            widget = self.step_item_cls(stage, self.steps_list)
            item = QListWidgetItem(self.steps_list)
            self.steps_list.addItem(item)
            self.steps_list.setItemWidget(item, widget)
            try:
                if hasattr(widget, "set_optional"):
                    widget.set_optional(stage in optional)
                    widget.includedChanged.connect(self._on_step_toggled)
                widget.set_status("pending")
                widget.playClicked.connect(
                    lambda _=None, s=stage: self.play_callback(s)
                )
                widget.configClicked.connect(
                    lambda _=None, s=stage: self.config_callback(s)
                )
            except Exception:
                pass
            item.setSizeHint(widget.sizeHint())
            self._step_widgets.append(widget)

    def _on_step_toggled(self, _stage: str, _included: bool) -> None:
        """Keep a ticked/unticked optional step in the selected stage-only SOP.

        Complete presets are only rewritten by their explicit Update action.
        """
        if not self.sops:
            return
        pipeline = self.pipeline_getter() or "sessile"
        try:
            sop = self.sops.get(pipeline, self._selected_sop_key())
            if sop is None or "__preset__" in (sop.params or {}):
                return
            sop.include_stages = self.collect_included_stages()
            self.sops.upsert(pipeline, sop)
        except Exception as exc:
            print("[SOP] step toggle:", exc)

    def _refresh_sop_combo(self, select: str | None = None) -> None:
        if not (self.sops and self.sop_combo):
            return
        combo = self.sop_combo
        combo.blockSignals(True)
        combo.clear()
        default_key = self.sops.default_name()
        combo.addItem("Default (pipeline)", userData=default_key)
        try:
            pipeline = self.pipeline_getter() or "sessile"
            for name in self.sops.list(pipeline):
                if name != default_key:
                    combo.addItem(name, userData=name)
        except Exception:
            pass
        index = 0
        if select:
            for i in range(combo.count()):
                if combo.itemData(i) == select:
                    index = i
                    break
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        if not self._sop_combo_connected:
            combo.currentIndexChanged.connect(lambda _: self._apply_selected_sop())
            self._sop_combo_connected = True

    def _selected_sop_key(self) -> str:
        if not (self.sops and self.sop_combo):
            return "__default__"
        data = self.sop_combo.currentData()
        return data if data else self.sops.default_name()

    def _apply_selected_sop(self) -> None:
        if not self._step_widgets:
            return
        if not self.sops:
            return
        pipeline = self.pipeline_getter() or "sessile"
        try:
            sop = self.sops.get(pipeline, self._selected_sop_key())
            if sop and "__preset__" in (sop.params or {}):
                return  # Complete presets apply only after the explicit review action.
            include = sop.include_stages if sop else self.stages_for(pipeline)
        except Exception:
            include = self.stages_for(pipeline)
        self.apply_included_stages(include)
