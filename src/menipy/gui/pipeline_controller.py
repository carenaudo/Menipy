"""Pipeline execution helper for Menipy GUI."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtWidgets import QMessageBox, QPlainTextEdit, QMainWindow


class PipelineController:
    """Handles pipeline execution and VM callbacks for the main window."""

    def __init__(
        self,
        window: QMainWindow,
        setup_ctrl,
        preview_panel,
        results_panel,
        pipeline_map: Mapping[str, type],
        sops: Optional[Any],
        run_vm: Optional[Any],
        log_view: Optional[QPlainTextEdit],
    ) -> None:
        self.window = window
        self.setup_ctrl = setup_ctrl
        self.preview_panel = preview_panel
        self.results_panel = results_panel
        self.sops = sops
        self.run_vm = run_vm
        self.log_view = log_view
        self.pipeline_map = {str(k).lower(): v for k, v in (pipeline_map or {}).items()}

    # ------------------------------------------------------------------
    # Slots wired by MainWindow
    # ------------------------------------------------------------------

    def run_full(self) -> None:
        params = self.setup_ctrl.gather_run_params()
        name = (params.get("name") or "sessile" or "").lower()
        pipeline_cls = self.pipeline_map.get(name)
        if not pipeline_cls:
            QMessageBox.warning(self.window, "Run", f"Unknown pipeline: {name}")
            return

        image = params.get("image")
        cam_id = params.get("cam_id")
        frames = params.get("frames")

        self.window.statusBar().showMessage(f"Running {name}.")

        if self.run_vm:
            try:
                self.run_vm.run(pipeline=name, image=image, camera=cam_id, frames=frames)
                return
            except Exception as exc:
                print("[run_vm] fallback to direct run:", exc)

        self._run_pipeline_direct(pipeline_cls, image=image, camera=cam_id, frames=frames)

    def run_all(self) -> None:
        if not self.sops:
            return self.run_full()

        params = self.setup_ctrl.gather_run_params()
        name = (params.get("name") or "sessile" or "").lower()
        image = params.get("image")
        cam_id = params.get("cam_id")
        frames = params.get("frames")

        stages = self.setup_ctrl.collect_included_stages()
        if not stages:
            QMessageBox.warning(self.window, "Run All", "No stages enabled in the current SOP.")
            return

        if self.run_vm and hasattr(self.run_vm, "run_subset"):
            try:
                self.window.statusBar().showMessage(f"Running {name} (SOP) .")
                self.run_vm.run_subset(name, only=stages, image=image, camera=cam_id, frames=frames)
                return
            except Exception as exc:
                print("[run_vm subset] falling back to full run:", exc)

        self.run_full()

    def run_stage(self, stage_name: str) -> None:
        if self.run_vm and hasattr(self.run_vm, "run_subset"):
            params = self.setup_ctrl.gather_run_params()
            try:
                self.run_vm.run_subset(
                    params.get("name"),
                    only=[stage_name],
                    image=params.get("image"),
                    camera=params.get("cam_id"),
                    frames=params.get("frames"),
                )
                return
            except Exception as exc:
                print("[run_vm single step] falling back to pipeline:", exc)

        params = self.setup_ctrl.gather_run_params()
        name = (params.get("name") or "sessile" or "").lower()
        pipeline_cls = self.pipeline_map.get(name)
        if not pipeline_cls:
            QMessageBox.warning(self.window, "Run", f"Unknown pipeline: {name}")
            return
        pipe = pipeline_cls()
        try:
            pipe.run_with_plan(
                only=[stage_name],
                include_prereqs=True,
                image=params.get("image"),
                camera=params.get("cam_id"),
                frames=params.get("frames"),
            )
        except Exception as exc:
            self.on_pipeline_error(str(exc))

    def on_preview_ready(self, payload: Any) -> None:
        try:
            self.preview_panel.display(payload)
        except Exception:
            pass
        self.window.statusBar().showMessage("Preview updated", 1000)

    def on_results_ready(self, results: Mapping[str, Any]) -> None:
        self.results_panel.update(results)
        self.window.statusBar().showMessage("Results ready", 1000)

    def append_logs(self, lines: Any) -> None:
        if not self.log_view:
            return
        try:
            if not lines:
                return
            if isinstance(lines, (list, tuple)):
                for line in lines:
                    self.log_view.appendPlainText(str(line))
            else:
                self.log_view.appendPlainText(str(lines))
        except Exception:
            pass

    def on_pipeline_error(self, message: str) -> None:
        QMessageBox.critical(self.window, "Pipeline Error", message)
        self.window.statusBar().showMessage("Error", 1500)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_pipeline_direct(self, pipeline_cls: type, **kwargs: Any) -> None:
        try:
            pipeline = pipeline_cls()
            ctx = pipeline.run(**kwargs)
            if getattr(ctx, "preview", None) is not None:
                self.preview_panel.display(ctx.preview)
            if getattr(ctx, "results", None):
                self.results_panel.update(ctx.results)
            self.window.statusBar().showMessage("Done", 1500)
        except Exception as exc:
            self.on_pipeline_error(str(exc))
