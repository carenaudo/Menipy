# src/adsa/gui/services/pipeline_runner.py
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from menipy.pipelines.base import PipelineError
from menipy.common import acquisition as acq

class _Job(QRunnable):
    def __init__(self, pipeline_name: str, image: Optional[str], camera: Optional[int], frames: int, callback):
        super().__init__()
        self.pipeline_name = pipeline_name
        self.image = image
        self.camera = camera
        self.frames = frames
        self.callback = callback

    def run(self):
        try:
            p = _pick(self.pipeline_name)
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
    name = name.lower()
    if name == "sessile":
        from menipy.pipelines.sessile.stages import SessilePipeline
        return SessilePipeline()
    if name == "oscillating":
        from menipy.pipelines.oscillating.stages import OscillatingPipeline
        return OscillatingPipeline()
    if name == "capillary_rise":
        from menipy.pipelines.capillary_rise.stages import CapillaryRisePipeline
        return CapillaryRisePipeline()
    if name == "pendant":
        from menipy.pipelines.pendant.stages import PendantPipeline
        return PendantPipeline()
    if name == "captive_bubble":
        from menipy.pipelines.captive_bubble.stages import CaptiveBubblePipeline
        return CaptiveBubblePipeline()
    raise PipelineError(f"Unknown pipeline '{name}'")

class PipelineRunner(QObject):
    finished = Signal(object)  # ctx or error dict

    def __init__(self):
        super().__init__()
        self.pool = QThreadPool.globalInstance()

    def run(self, pipeline: str, image: Optional[str], camera: Optional[int], frames: int = 1):
        job = _Job(pipeline, image, camera, frames, callback=self._emit)
        self.pool.start(job)

    def _emit(self, success: bool, ctx, err: Optional[str]):
        self.finished.emit({"ok": success, "ctx": ctx, "err": err})
