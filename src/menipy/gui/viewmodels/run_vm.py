"""
View model for pipeline run management.
"""

# src/menipy/gui/viewmodels/run_vm.py
from __future__ import annotations
from PySide6.QtCore import QObject, Signal
from menipy.gui.services.pipeline_runner import PipelineRunner
from menipy.gui.services.image_convert import to_pixmap


class RunViewModel(QObject):
    preview_ready = Signal(object)  # QPixmap
    results_ready = Signal(dict)
    error_occurred = Signal(str)
    status_ready = Signal(str)
    logs_ready = Signal(object)

    def __init__(self, runner: PipelineRunner):
        super().__init__()
        self.runner = runner
        self.runner.finished.connect(self._done)

    def run(
        self,
        pipeline: str | None = None,
        image: str | None = None,
        camera: int | None = None,
        frames: int = 1,
        **overlays,
    ) -> None:
        """Run the pipeline. Accept positional or keyword *pipeline* for compatibility with MainWindow callers."""
        self.runner.run(pipeline, image, camera, frames, **overlays)

    def run_subset(
        self,
        pipeline: str | None,
        *,
        only: list[str],
        image: str | None = None,
        camera: int | None = None,
        frames: int = 1,
        **overlays,
    ) -> None:
        self.runner.run_subset(
            pipeline, only=only, image=image, camera=camera, frames=frames, **overlays
        )

    def _done(self, payload):
        if not payload["ok"]:
            self.error_occurred.emit(payload["err"] or "Unknown error")
            return
        ctx = payload["ctx"]
        if getattr(ctx, "preview", None) is not None:
            self.preview_ready.emit(to_pixmap(ctx.preview))
        if getattr(ctx, "results", None) is not None:
            self.results_ready.emit(dict(ctx.results))
        # Forward any short human-readable status message attached to the Context
        try:
            sm = getattr(ctx, "status_message", None)
            if sm:
                self.status_ready.emit(str(sm))
        except Exception:
            pass
        # Forward collected log lines (list[str]) if present
        try:
            logs = getattr(ctx, "log", None)
            if logs:
                # emit as-is (usually a list of strings)
                self.logs_ready.emit(list(logs))
        except Exception:
            pass
