"""Explicit folder actions and per-file outcomes; all computation stays in workers."""

import csv
import json

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from menipy.gui.services.folder_execution import folder_images, folder_task
from menipy.gui.services.pipeline_runner import RunRequest


class FolderController(QObject):
    progress = Signal(object)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setup = window.setup_panel_ctrl
        self.runner = window.runner
        self.batch_id = None
        self._running_batch = False
        self.requests = {}
        self.outcomes = {}
        self.rows = {}
        self.records = {}
        action_layout = self.setup.runAllBtn.parentWidget().layout()
        action_index = action_layout.indexOf(self.setup.runAllBtn) + 1
        self.button = QPushButton("Run folder", self.setup.panel)
        self.button.setObjectName("runFolderBtn")
        self.button.setStyleSheet(self.setup.autoCalibrateBtn.styleSheet())
        action_layout.insertWidget(action_index, self.button)
        self.button.clicked.connect(self.start)
        self.results_button = QPushButton("Folder results", self.setup.panel)
        self.results_button.setStyleSheet(self.setup.autoCalibrateBtn.styleSheet())
        action_layout.insertWidget(action_index + 1, self.results_button)
        self.dialog = QDialog(window)
        self.dialog.setWindowTitle("Folder results — independent images")
        self.dialog.resize(780, 430)
        self.results_button.clicked.connect(self.dialog.show)
        layout = QVBoxLayout(self.dialog)
        self.label = QLabel()
        layout.addWidget(self.label)
        self.table = QTableWidget(0, 3)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Details"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.stop = QPushButton("Stop")
        self.retry = QPushButton("Retry failures")
        self.retry.setObjectName("runFolderRetryBtn")
        export = QPushButton("Export CSV…")
        for button in (self.stop, self.retry, export):
            actions.addWidget(button)
        self.stop.clicked.connect(lambda: self.runner.cancel(self.batch_id))
        self.retry.clicked.connect(self.retry_failed)
        export.clicked.connect(self.export)
        self.progress.connect(self.on_progress, Qt.QueuedConnection)
        self.runner.finished.connect(self.on_finished)
        self.runner.state_changed.connect(self.on_state)
        self.setup.pipeline_changed.connect(self.refresh)
        self.setup.source_mode_changed.connect(self.refresh)
        self.setup.batchPathEdit.textChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, *_):
        dynamic = self.setup.current_pipeline_name() == "sessile_dynamic"
        folder = self.setup.current_mode() == self.setup.MODE_BATCH
        self.button.setVisible(folder and not dynamic)
        self.button.setEnabled(not self.runner.busy)
        self.results_button.setVisible(bool(self.requests))
        self.button.setToolTip(
            "Analyze each image independently. A confirmed sessile substrate line is reused and checked in every image."
        )
        try:
            count = len(folder_images(self.setup.batch_path())) if folder else 0
        except (ValueError, OSError):
            count = 0
        self.button.setText(f"Run folder ({count})")
        self.setup.runAllBtn.setText(
            "Run sequence" if dynamic else "Run selected" if folder else "Run Analysis"
        )
        camera = self.window.workflowSourceModeButtons[self.setup.MODE_CAMERA]
        camera.setEnabled(not dynamic)
        if dynamic and self.setup.current_mode() == self.setup.MODE_CAMERA:
            self.setup.singleModeRadio.click()
        self.window.workflowSourceModeButtons[self.setup.MODE_BATCH].setToolTip(
            "Frame sequence folder (requires FPS)" if dynamic else "Image folder"
        )
        self.stop.setEnabled(self._running_batch and self.runner.busy)
        self.retry.setEnabled(
            not self.runner.busy and "failed" in self.outcomes.values()
        )

    def on_state(self, job_id, state):
        self.refresh()
        if job_id == self.batch_id and state == "stopping":
            self.label.setText("Stopping… Completed files remain in history.")
            self.stop.setEnabled(False)

    def start(self):
        if self.runner.busy:
            return
        if self.setup.current_pipeline_name() == "sessile_dynamic":
            return self.window.pipeline_ctrl.run_full()
        try:
            files = folder_images(self.setup.batch_path() or "")
            if not files:
                raise ValueError("The folder contains no supported images.")
            name, _, parameters, warnings = (
                self.window.pipeline_ctrl._build_pipeline_run_kwargs()
            )
            requests = []
            for path in files:
                # Independent images must not inherit drop/needle detections.
                # A confirmed sessile substrate is a sample reference and is
                # deliberately retained for every image in the set.
                owned = dict(parameters)
                for key in (
                    "image",
                    "image_path",
                    "roi",
                    "roi_rect",
                    "needle_rect",
                    "contact_line",
                    "substrate_line",
                    "substrate_profile",
                    "drop_contour",
                    "detected_contour",
                    "contact_points",
                    "apex_point",
                    "scale",
                    "detector_diagnostics",
                    "preprocessing_markers",
                    "calibration_provenance",
                ):
                    if key not in {"substrate_line", "substrate_profile"}:
                        owned.pop(key, None)
                owned.update(image=path, camera=None, auto_calibrate=True)
                requests.append(
                    RunRequest.create(name, owned, revision="folder-history")
                )
            self.requests.clear()
            self.outcomes.clear()
            self.records.clear()
            self.rows.clear()
            self.table.setRowCount(0)
            self.submit(requests)
        except Exception as exc:
            QMessageBox.warning(self.window, "Folder analysis", str(exc))

    def submit(self, requests):
        batch = RunRequest.create(requests[0].pipeline, {}, operation="batch")
        self.batch_id = batch.job_id
        self._running_batch = True
        for request in requests:
            self.requests[request.job_id] = request
            self.outcomes[request.job_id] = "queued"
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.rows[request.job_id] = row
            for column, text in enumerate((request.source, "queued", "")):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.dialog.show()
        self.runner.submit(
            batch, folder_task(batch.job_id, requests, self.progress.emit)
        )
        self.update_count()

    def on_progress(self, event):
        if event.batch_id != self.batch_id or event.request.job_id not in self.rows:
            return
        job = event.request.job_id
        if self.outcomes[job] not in ("queued", "running"):
            return
        state, detail = "running", ""
        if event.completion is not None:
            state = event.completion.state
            if state == "completed":
                result = self.window.pipeline_ctrl.on_completed(event.completion)
                if result is not None:
                    self.records[job] = result.model_dump(mode="json")
                    state = result.display_status.lower()
                    detail = "; ".join(result.rejection_reasons)
                    if result.diagnostics.get("calibration"):
                        detail = "; ".join(
                            filter(None, (detail, result.calibration_summary))
                        )
            else:
                detail = event.completion.error or state
        self.outcomes[job] = state
        row = self.rows[job]
        self.table.item(row, 1).setText(state)
        self.table.item(row, 2).setText(detail)
        self.update_count()

    def update_count(self):
        done = sum(s not in ("queued", "running") for s in self.outcomes.values())
        self.label.setText(
            f"{done} / {len(self.outcomes)} files finished. "
            + (
                "History is unsaved; use Retry save or Save recovery copy in Results."
                if getattr(self.window.results_panel_ctrl.history, "unsaved", False)
                else "Results are saved to history."
            )
        )

    def on_finished(self, completion):
        if completion.request.job_id != self.batch_id:
            return
        self._running_batch = False
        for job, state in self.outcomes.items():
            if state in ("queued", "running"):
                self.outcomes[job] = completion.state
                self.table.item(self.rows[job], 1).setText(completion.state)
                self.table.item(self.rows[job], 2).setText(completion.error or "")
        self.update_count()
        self.refresh()

    def retry_failed(self):
        if self.runner.busy:
            return
        failed = [job for job in self.requests if self.outcomes[job] == "failed"]
        requests = [
            RunRequest.create(
                self.requests[job].pipeline,
                self.requests[job].parameters,
                revision=self.requests[job].revision,
            )
            for job in failed
        ]
        if requests:
            self.submit(requests)
            for job in failed:
                self.outcomes[job] = "retried"
                self.table.item(self.rows[job], 1).setText("retried")

    def export(self):
        path, _ = QFileDialog.getSaveFileName(
            self.dialog, "Export folder results", "folder_results.csv", "CSV (*.csv)"
        )
        if path:
            try:
                self.write_csv(path)
            except Exception as exc:
                QMessageBox.warning(self.dialog, "Export failed", str(exc))

    def write_csv(self, path):
        metrics = sorted(
            {key for record in self.records.values() for key in record["results"]}
        )
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "job_id",
                    "image_path",
                    "pipeline",
                    "status",
                    "details",
                    "diagnostics_json",
                    "run_metadata",
                    *metrics,
                ],
            )
            writer.writeheader()
            for job, request in self.requests.items():
                record = self.records.get(job, {})
                writer.writerow(
                    dict(
                        job_id=job,
                        image_path=request.source,
                        pipeline=request.pipeline,
                        status=self.outcomes[job],
                        details=self.table.item(self.rows[job], 2).text(),
                        diagnostics_json=json.dumps(record.get("diagnostics", {})),
                        run_metadata=json.dumps(
                            record.get("run_metadata", request.metadata())
                        ),
                        **record.get("results", {}),
                    )
                )
