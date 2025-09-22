# src/adsa/gui/services/pipeline_runner.py
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from menipy.pipelines.base import PipelineBase, PipelineError
from menipy.common import acquisition as acq

# Import the same map the GUI uses to ensure consistency
try:
    from menipy.pipelines.discover import PIPELINE_MAP
except ImportError:
    PIPELINE_MAP = {}


class _Job(QRunnable):
    def __init__(self, pipeline_cls: type[PipelineBase], image: Optional[str], camera: Optional[int], frames: int, callback):
        super().__init__()
        self.pipeline_cls = pipeline_cls
        self.image = image
        self.camera = camera
        self.frames = frames
        self.callback = callback

    def run(self):
        try:
            p = self.pipeline_cls()
            # patch acquisition
            if self.image:
                p.do_acquisition = (lambda ctx: setattr(ctx, "frames", acq.from_file([self.image])) or ctx)  # type: ignore
            else:
                p.do_acquisition = (lambda ctx: setattr(ctx, "frames", acq.from_camera(device=self.camera or 0, n_frames=self.frames)) or ctx)  # type: ignore
            ctx = p.run()
            self.callback(success=True, ctx=ctx, err=None)
        except Exception as e:
            self.callback(success=False, ctx=None, err=str(e))


def _pick(name: str):
    """Look up the pipeline class from the central map."""
    p_cls = PIPELINE_MAP.get(name.lower())
    if p_cls is None:
        raise PipelineError(f"Unknown pipeline '{name}'")
    return p_cls

class PipelineRunner(QObject):
    finished = Signal(object)  # ctx or error dict

    def __init__(self):
        super().__init__()
        self.pool = QThreadPool.globalInstance()

    def run(self, pipeline: str, image: Optional[str], camera: Optional[int], frames: int = 1):
        pipeline_cls = _pick(pipeline)
        job = _Job(pipeline_cls, image, camera, frames, callback=self._emit)
        self.pool.start(job)

    def _emit(self, success: bool, ctx, err: Optional[str]):
        self.finished.emit({"ok": success, "ctx": ctx, "err": err})
