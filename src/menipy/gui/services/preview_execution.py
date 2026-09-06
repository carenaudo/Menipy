"""Background preview computations; Qt controllers only publish completed data."""

import logging

import numpy as np
from PySide6.QtCore import QObject

from menipy.common import edge_detection, preprocessing, registry
from menipy.common.cancellation import check_cancelled
from menipy.common.geometry import find_contact_points_from_contour
from menipy.gui.services.pipeline_runner import RunRequest
from menipy.models.context import Context
from menipy.models.frame import Frame

logger = logging.getLogger(__name__)


def preprocessing_preview(parameters, token):
    image = parameters.pop("image")
    settings = parameters["preprocessing_settings"]
    ctx = Context(
        current_frame=Frame(image=image),
        image=image,
        cancellation_token=token,
        **parameters,
    )
    if getattr(settings, "auto_detect", None) and settings.auto_detect.enabled:
        detector = registry.PREPROCESSORS.get("auto_detect")
        if detector:
            try:
                detector(ctx)
            except Exception:
                logger.warning("Preview auto-detection failed", exc_info=True)
    check_cancelled()
    preprocessing.run(ctx, settings)
    return ctx


def edge_preview(parameters, token):
    image = parameters.pop("image")
    settings = parameters["edge_detection_settings"]
    ctx = Context(
        current_frame=Frame(image=image),
        image=image,
        cancellation_token=token,
        **parameters,
    )
    edge_detection.run(ctx, settings)
    check_cancelled()
    contour = getattr(ctx.contour, "xy", None)
    preview = image.copy()
    if preview.ndim == 2:
        preview = np.stack((preview,) * 3, axis=-1)
    metadata = {}
    if contour is not None:
        points = None
        if ctx.contact_line is not None:
            try:
                points = find_contact_points_from_contour(
                    np.asarray(contour, dtype=float), ctx.contact_line
                )
            except Exception:
                logger.debug("Preview contact detection failed", exc_info=True)
        metadata = {
            "contact_points": points,
            "contour_xy": np.asarray(contour, dtype=float),
        }
    return preview, metadata


class PreviewExecution(QObject):
    def __init__(self, window, owner, task, publish):
        super().__init__(owner)
        self.window, self.owner = window, owner
        self.task, self.publish = task, publish
        self.job_id = None
        window.runner.finished.connect(self._done)

    def submit(self, parameters):
        if self.window.runner.busy:
            self.window.statusBar().showMessage(
                "Preview not updated while an operation is running."
            )
            return
        request = RunRequest.create(
            self.window.setup_panel_ctrl.current_pipeline_name(),
            parameters,
            operation="preview",
            revision=self.window.pipeline_ctrl._revision(),
        )
        self.job_id = request.job_id
        try:
            self.window.runner.submit(request, self.task)
        except Exception as exc:
            self.job_id = None
            self.owner.errorOccurred.emit(str(exc))

    def _done(self, completion):
        if completion.request.job_id != self.job_id:
            return
        self.job_id = None
        if completion.state == "failed":
            self.owner.errorOccurred.emit(completion.error)
        elif (
            completion.state == "completed"
            and completion.request.revision == self.window.pipeline_ctrl._revision()
        ):
            self.publish(completion.value)
