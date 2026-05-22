"""Service for running pipelines in the GUI context."""

# src/adsa/gui/services/pipeline_runner.py
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from menipy.common import acquisition as acq
from menipy.pipelines.base import PipelineBase, PipelineError

# Import the same map the GUI uses to ensure consistency
try:
    from menipy.pipelines.discover import PIPELINE_MAP
except ImportError:
    PIPELINE_MAP = {}


class _Job(QRunnable):
    def __init__(
        self,
        pipeline_cls: type[PipelineBase],
        image: str | None,
        camera: int | None,
        frames: int,
        callback,
        *,
        roi=None,
        roi_rect=None,
        detected_roi=None,
        needle_rect=None,
        contact_line=None,
        substrate_line=None,
        drop_contour=None,
        detected_contour=None,
        contact_points=None,
        apex_point=None,
        auto_detect_features=None,
        preprocessing_settings=None,
        preprocessing_markers=None,
        edge_detection_settings=None,
        calibration_params=None,
        scale=None,
        physics=None,
        stages: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.pipeline_cls = pipeline_cls
        self.image = image
        self.camera = camera
        self.frames = frames
        self.callback = callback
        self.roi = roi
        self.roi_rect = roi_rect
        self.detected_roi = detected_roi
        self.needle_rect = needle_rect
        self.contact_line = contact_line
        self.substrate_line = substrate_line
        self.drop_contour = drop_contour
        self.detected_contour = detected_contour
        self.contact_points = contact_points
        self.apex_point = apex_point
        self.auto_detect_features = auto_detect_features
        self.preprocessing_settings = preprocessing_settings
        self.preprocessing_markers = preprocessing_markers
        self.edge_detection_settings = edge_detection_settings
        self.calibration_params = calibration_params
        self.scale = scale
        self.physics = physics
        self.stages = stages

    def run(self):
        try:
            # Instantiate the pipeline with the provided settings.
            p = self.pipeline_cls(
                preprocessing_settings=self.preprocessing_settings,
                edge_detection_settings=self.edge_detection_settings,
            )

            # patch acquisition - DISABLED to allow pipeline class method to run (and use logging)
            # if self.image:
            #     p.do_acquisition = (lambda ctx: setattr(ctx, "frames", acq.from_file([self.image])) or ctx)  # type: ignore
            # else:
            #     p.do_acquisition = (lambda ctx: setattr(ctx, "frames", acq.from_camera(device=self.camera or 0, n_frames=self.frames)) or ctx)  # type: ignore
            run_kwargs = {
                "roi": self.roi,
                "roi_rect": self.roi_rect,
                "detected_roi": self.detected_roi,
                "needle_rect": self.needle_rect,
                "contact_line": self.contact_line,
                "substrate_line": self.substrate_line,
                "drop_contour": self.drop_contour,
                "detected_contour": self.detected_contour,
                "contact_points": self.contact_points,
                "apex_point": self.apex_point,
                "auto_detect_features": self.auto_detect_features,
                "preprocessing_markers": self.preprocessing_markers,
                "calibration_params": self.calibration_params,
                "scale": self.scale,
                "physics": self.physics,
                "image": self.image,
                "camera": self.camera,
                "frames": self.frames,
            }
            run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}
            if self.stages:
                ctx = p.run_with_plan(
                    only=self.stages, include_prereqs=True, **run_kwargs
                )
            else:
                ctx = p.run(**run_kwargs)
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

    def run(
        self,
        pipeline: str,
        image: str | None,
        camera: int | None,
        frames: int = 1,
        **overlays,
    ):
        pipeline_cls = _pick(pipeline)
        job = _Job(pipeline_cls, image, camera, frames, callback=self._emit, **overlays)
        self.pool.start(job)

    def run_subset(
        self,
        pipeline: str,
        *,
        only: list[str],
        image: str | None,
        camera: int | None,
        frames: int = 1,
        **overlays,
    ) -> None:
        """Run subset of pipeline.

        Parameters
        ----------
        pipeline : str
            Pipeline name.
        only : list[str]
            List of stage names to run.
        image : str, optional
            Image path.
        camera : int, optional
            Camera index.
        frames : int, optional
            Number of frames. Default is 1.
        **overlays
            Additional overlay settings.
        """
        pipeline_cls = _pick(pipeline)
        job = _Job(
            pipeline_cls,
            image,
            camera,
            frames,
            callback=self._emit,
            stages=only,
            **overlays,
        )
        self.pool.start(job)

    def _emit(self, success: bool, ctx, err: str | None):
        self.finished.emit({"ok": success, "ctx": ctx, "err": err})
