"""Source readiness and source-scoped calibration for the workbench."""

from pathlib import Path

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QAbstractButton, QLabel


class ReadinessController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setup = window.setup_panel_ctrl
        self.source_key = self.key()
        self.notice = QLabel(self.setup.panel)
        self.notice.setObjectName("sourceReadinessLabel")
        self.notice.setWordWrap(True)
        layout = self.setup.runAllBtn.parentWidget().layout()
        layout.insertWidget(layout.indexOf(self.setup.runAllBtn), self.notice)
        self.previous_notice = QLabel(window.previewHost)
        self.previous_notice.setWordWrap(True)
        window.previewHost.layout().insertWidget(0, self.previous_notice)
        for widget in (self.setup.imagePathEdit, self.setup.batchPathEdit):
            widget.textChanged.connect(self.source_changed)
        self.setup.sourceIdCombo.currentTextChanged.connect(self.source_changed)
        self.setup.source_mode_changed.connect(self.source_changed)
        self.setup.pipeline_changed.connect(self.source_changed)
        window.runner.state_changed.connect(
            lambda *_: QTimer.singleShot(0, self.refresh)
        )
        window.runner.finished.connect(self.completed)
        for button in window.findChildren(QAbstractButton):
            name = button.toolTip() or button.text() or button.objectName()
            if name:
                button.setAccessibleName(name.replace("&", ""))
        window.setStyleSheet(
            window.styleSheet()
            + "\nQPushButton:focus, QToolButton:focus { border: 2px solid #0969da; }"
        )
        from PySide6.QtGui import QAction

        mode_labels = QAction("Show analysis mode labels", window)
        mode_labels.setCheckable(True)
        mode_labels.setChecked(window.settings.show_mode_labels)

        def toggle_labels(visible):
            window.settings.show_mode_labels = visible
            window.settings.save()
            self.setup._sync_pipeline_button_presentation()

        mode_labels.toggled.connect(toggle_labels)
        window.menuConfig.addAction(mode_labels)
        for label_name, field_name in (
            ("needleLengthLabel", "needleLengthSpin"),
            ("dropDensityLabel", "dropDensitySpin"),
            ("fluidDensityLabel", "fluidDensitySpin"),
            ("dynamicFpsLabel", "dynamicFpsSpin"),
        ):
            label, field = (
                getattr(self.setup, label_name, None),
                getattr(self.setup, field_name, None),
            )
            if label is not None and field is not None:
                label.setBuddy(field)
                field.setAccessibleName(label.text())
        self.refresh()

    def key(self):
        params = self.setup.gather_run_params()
        return (
            params.get("name"),
            params.get("mode"),
            params.get("image"),
            params.get("batch_folder"),
            params.get("cam_id"),
        )

    def source_changed(self, *_):
        key = self.key()
        if key != self.source_key:
            self.source_key = key
            self.window._last_calibration_result = None
            self.window.main_controller._last_calibration_result = None
            self.window.preview_panel.clear_overlays()
            self.window.preprocessing_ctrl.update_markers({})
            self.window.preprocessing_ctrl.clear_geometry()
            self.window.edge_detection_ctrl.set_contact_line(None)
            self.window.main_controller.image_manager._cached_image_path = None
            self.window.main_controller.image_manager._cached_image_data = None
            self.previous_notice.setText(
                "Source changed. Previous results remain in history; the preview may show the previous source. Load/preview and recalibrate this source before analysis."
            )
            self.window.pipeline_ctrl.mark_setup_changed()
        self.refresh()
        QTimer.singleShot(25, self.refresh)

    def refresh(self):
        params = self.setup.gather_run_params()
        dynamic = params.get("name") == "sessile_dynamic"
        source = (
            params.get("batch_folder")
            if dynamic and params.get("mode") == "batch"
            else params.get("image")
        )
        ready = params.get("mode") == "camera" or bool(source and Path(source).exists())
        if dynamic and params.get("mode") == "batch":
            from menipy.gui.services.folder_execution import folder_images

            try:
                ready = bool(folder_images(source))
            except (ValueError, OSError):
                ready = False
        busy = self.window.runner.busy
        message = (
            "Ready. Calibrate this source before analysis."
            if ready
            else "Select an image, video, or a non-empty folder to enable analysis."
        )
        self.notice.setText(
            "An operation is running; setup remains editable." if busy else message
        )
        for button in (
            self.setup.runAllBtn,
            self.setup.autoCalibrateBtn,
            getattr(self.setup, "analyzeBtn", None),
        ):
            if button is not None:
                button.setEnabled(ready and not busy)
                button.setToolTip(message)
        for name in ("actionRunFull", "actionRunSelected"):
            action = getattr(self.window, name, None)
            if action is not None:
                action.setEnabled(ready and not busy)
        for mode, button in self.window.workflowSourceModeButtons.items():
            button.setAccessibleName(
                {
                    "single": "Open file source",
                    "batch": "Open folder source",
                    "camera": "Camera source",
                }[mode]
            )

    def completed(self, completion):
        if (
            completion.state == "completed"
            and completion.ctx is not None
            and completion.request.revision == self.window.pipeline_ctrl._revision()
        ):
            self.previous_notice.clear()
