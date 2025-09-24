"""SOP management helper for Menipy GUI."""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from PySide6.QtWidgets import QListWidget, QListWidgetItem, QComboBox, QMessageBox, QInputDialog, QWidget

try:
    from menipy.gui.services.sop_service import Sop
except Exception:
    Sop = None  # type: ignore


class SopController:
    """Encapsulates SOP list management and step widgets."""

    def __init__(
        self,
        window,
        sops: Any,
        stage_order: Sequence[str],
        step_item_cls: Optional[type],
        steps_list: Optional[QListWidget],
        sop_combo: Optional[QComboBox],
        pipeline_getter: Callable[[], Optional[str]],
        pipeline_changed_callback: Callable[[str], None],
        play_callback: Callable[[str], None],
        config_callback: Callable[[str], None],
    ) -> None:
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
        if not self.sops:
            return
        pipeline = self.pipeline_getter() or "sessile"
        try:
            self.sops.ensure_default(pipeline, self.stage_order)
        except Exception as exc:
            print("[SOP] pipeline change:", exc)
        self._refresh_sop_combo()
        self._apply_selected_sop()
        if pipeline:
            try:
                self.pipeline_changed_callback(pipeline)
            except Exception:
                pass

    def collect_included_stages(self) -> list[str]:
        if not self._step_widgets:
            return list(self.stage_order)
        included: list[str] = []
        for widget in self._step_widgets:
            step_name = getattr(widget, "step_name", None)
            if not step_name:
                continue
            try:
                if widget.isEnabled():
                    included.append(step_name)
            except Exception:
                included.append(step_name)
        return included

    def on_add_sop(self) -> None:
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
            if Sop is not None:
                sop_obj = Sop(name=name, include_stages=include, params={})
                self.sops.upsert(pipeline_key, sop_obj)
            else:
                sop_like = type("SopLike", (), {"name": name, "include_stages": include, "params": {}})
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
        if not self.step_item_cls:
            for stage in self.stage_order:
                self.steps_list.addItem(stage)
            return
        for stage in self.stage_order:
            widget = self.step_item_cls(stage, self.steps_list)
            item = QListWidgetItem(self.steps_list)
            item.setSizeHint(widget.sizeHint())
            self.steps_list.addItem(item)
            self.steps_list.setItemWidget(item, widget)
            try:
                widget.set_status("pending")
                widget.playClicked.connect(lambda _=None, s=stage: self.play_callback(s))
                widget.configClicked.connect(lambda _=None, s=stage: self.config_callback(s))
            except Exception:
                pass
            self._step_widgets.append(widget)

    def _refresh_sop_combo(self, select: Optional[str] = None) -> None:
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
            include = set(sop.include_stages if sop else self.stage_order)
        except Exception:
            include = set(self.stage_order)
        for widget in self._step_widgets:
            step_name = getattr(widget, "step_name", None)
            enabled = step_name in include if step_name else False
            try:
                widget.setEnabled(enabled)
                if not enabled:
                    widget.set_status("pending")
            except Exception:
                pass
