"""
Analysis Settings Dialog

Generic dialog that wraps preprocessing, edge-detection, and pipeline-specific
settings. Pipelines can provide an optional widget in
`menipy.gui.dialogs.analysis_settings.<pipeline>_settings` exposing a
`PipelineSettingsWidget` class with a `get_settings()` method.
"""
from __future__ import annotations

from importlib import import_module
from typing import Optional, Any
import json

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QCheckBox,
)

from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings
from menipy.gui.dialogs.preprocessing_config_dialog import PreprocessingConfigDialog


class AnalysisSettingsDialog(QDialog):
    """Collects analysis settings; optionally augments with pipeline-specific tab."""

    def __init__(
        self,
        pipeline: str,
        *,
        preprocessing: Optional[PreprocessingSettings] = None,
        edge: Optional[EdgeDetectionSettings] = None,
        pipeline_settings: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Analysis Settings")
        self.resize(720, 520)

        self._pipeline = pipeline
        self._settings_store = QSettings("Menipy", "ADSA")

        # Load persisted settings if available
        saved = self._load_saved()
        self._preproc = saved.get("preproc") or preprocessing or PreprocessingSettings()
        self._edge = saved.get("edge") or edge or EdgeDetectionSettings()
        self._pipeline_settings = saved.get("pipeline") or pipeline_settings or {}
        self._pipeline_widget = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, 1)

        # Preprocessing tab
        self._tabs.addTab(self._build_preproc_tab(), "Preprocessing")
        # Edge detection tab
        self._tabs.addTab(self._build_edge_tab(), "Edge Detection")
        # Pipeline tab (optional)
        self._maybe_add_pipeline_tab()

        # Footer buttons
        buttons = QHBoxLayout()
        buttons.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setProperty("secondary", True)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Save")
        btn_ok.clicked.connect(self.accept)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_ok)
        layout.addLayout(buttons)

    def _build_preproc_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        desc = QLabel(
            "Configure preprocessing stages (auto-detect, resize, filtering, background, normalization…)."
        )
        desc.setWordWrap(True)
        v.addWidget(desc)

        summary = QLabel(self._preproc_summary())
        summary.setObjectName("preprocSummary")
        summary.setStyleSheet("color: #7f8c8d;")
        v.addWidget(summary)

        btn = QPushButton("Open Preprocessing Config…")
        btn.clicked.connect(lambda: self._open_preproc_dialog(summary))
        v.addWidget(btn)
        v.addStretch(1)
        return w

    def _build_edge_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)

        self._edge_method = QComboBox()
        self._edge_method.addItems(
            ["canny", "sobel", "scharr", "laplacian", "threshold", "active_contour"]
        )
        self._edge_method.setCurrentText(self._edge.method)
        form.addRow("Method", self._edge_method)

        self._edge_t1 = QDoubleSpinBox()
        self._edge_t1.setRange(0, 255)
        self._edge_t1.setValue(self._edge.canny_threshold1)
        form.addRow("Threshold 1", self._edge_t1)

        self._edge_t2 = QDoubleSpinBox()
        self._edge_t2.setRange(0, 255)
        self._edge_t2.setValue(self._edge.canny_threshold2)
        form.addRow("Threshold 2", self._edge_t2)

        self._edge_aperture = QComboBox()
        self._edge_aperture.addItems(["3", "5", "7"])
        self._edge_aperture.setCurrentText(str(self._edge.canny_aperture_size))
        form.addRow("Aperture size", self._edge_aperture)

        self._edge_l2 = QCheckBox("Use L2 gradient (Canny)")
        self._edge_l2.setChecked(self._edge.canny_L2_gradient)
        form.addRow("", self._edge_l2)

        self._edge_thresh_type = QComboBox()
        self._edge_thresh_type.addItems(
            ["binary", "binary_inv", "trunc", "to_zero", "to_zero_inv"]
        )
        self._edge_thresh_type.setCurrentText(self._edge.threshold_type)
        form.addRow("Threshold type", self._edge_thresh_type)

        self._edge_thresh_val = QDoubleSpinBox()
        self._edge_thresh_val.setRange(0, 255)
        self._edge_thresh_val.setValue(self._edge.threshold_value)
        form.addRow("Threshold value", self._edge_thresh_val)

        return w

    def _maybe_add_pipeline_tab(self):
        module_name = f"menipy.gui.dialogs.analysis_settings.{self._pipeline}_settings"
        try:
            mod = import_module(module_name)
            widget_cls = getattr(mod, "PipelineSettingsWidget", None)
            if widget_cls:
                self._pipeline_widget = widget_cls(parent=self, settings=self._pipeline_settings)
                self._tabs.addTab(self._pipeline_widget, f"{self._pipeline.title()} Settings")
                return
        except ModuleNotFoundError:
            pass
        except Exception:
            # if module exists but errors, ignore gracefully
            pass
        # Fallback info tab
        info = QWidget()
        v = QVBoxLayout(info)
        lbl = QLabel(
            f"No pipeline-specific settings implemented for '{self._pipeline}'."
        )
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        v.addStretch(1)
        self._tabs.addTab(info, f"{self._pipeline.title()} Settings")

    # ------------------------------------------------------------------ helpers
    def _open_preproc_dialog(self, summary_label: QLabel):
        dlg = PreprocessingConfigDialog(self._preproc, self)
        if dlg.exec():
            self._preproc = dlg.settings()
            summary_label.setText(self._preproc_summary())

    def _preproc_summary(self) -> str:
        ad = "on" if self._preproc.auto_detect.enabled else "off"
        filt = self._preproc.filtering.method
        return f"Auto-detect: {ad} • Filter: {filt} • Resize: {self._preproc.resize.target_width or 'auto'}"

    # ------------------------------------------------------------------ persistence
    def _load_saved(self) -> dict:
        key = f"analysis/{self._pipeline}"
        raw = self._settings_store.value(key)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            pre = PreprocessingSettings(**data.get("preproc", {})) if data.get("preproc") else None
            edge = EdgeDetectionSettings(**data.get("edge", {})) if data.get("edge") else None
            pipe = data.get("pipeline") or {}
            return {"preproc": pre, "edge": edge, "pipeline": pipe}
        except Exception:
            return {}

    def persist(self):
        """Persist current selections to QSettings."""
        payload = {
            "preproc": self._preproc.model_dump(),
            "edge": self.edge_settings().model_dump(),
            "pipeline": self.pipeline_settings() or {},
        }
        key = f"analysis/{self._pipeline}"
        self._settings_store.setValue(key, json.dumps(payload))

    # ------------------------------------------------------------------ results
    def preprocessing_settings(self) -> PreprocessingSettings:
        return self._preproc

    def edge_settings(self) -> EdgeDetectionSettings:
        # update edge settings from UI
        self._edge.method = self._edge_method.currentText()
        self._edge.canny_threshold1 = int(self._edge_t1.value())
        self._edge.canny_threshold2 = int(self._edge_t2.value())
        try:
            self._edge.canny_aperture_size = int(self._edge_aperture.currentText())
        except Exception:
            pass
        self._edge.canny_L2_gradient = self._edge_l2.isChecked()
        self._edge.threshold_type = self._edge_thresh_type.currentText()
        self._edge.threshold_value = int(self._edge_thresh_val.value())
        return self._edge

    def pipeline_settings(self) -> Optional[Any]:
        if self._pipeline_widget and hasattr(self._pipeline_widget, "get_settings"):
            return self._pipeline_widget.get_settings()
        return None
