"""Owned, cancellable GUI execution with immutable submission identity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot

from menipy.common.cancellation import (
    AnalysisCancelled,
    CancellationToken,
    cancellation_scope,
)
from menipy.pipelines.base import PipelineError
from menipy.pipelines.discover import PIPELINE_MAP

RunState = Literal["queued", "running", "stopping", "completed", "failed", "cancelled"]
RunOperation = Literal[
    "analysis",
    "quick_analysis",
    "sop",
    "stage",
    "stage_test",
    "calibration",
    "preview",
    "batch",
]


def json_settings(value):
    """Serialize configuration only; never persist images or runtime objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "model_dump"):
        return json_settings(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {
            str(k): json_settings(v)
            for k, v in value.items()
            if not isinstance(v, CancellationToken)
            and not (
                k in {"image", "frames", "original_image"}
                and isinstance(v, (np.ndarray, list))
            )
        }
    if isinstance(value, (list, tuple)):
        return [json_settings(v) for v in value if not isinstance(v, np.ndarray)]
    return None


@dataclass(frozen=True)
class RunRequest:
    pipeline: str
    source: str | None
    operation: RunOperation
    parameters: dict[str, Any] = field(repr=False)
    stages: tuple[str, ...] = ()
    revision: str = ""
    job_id: str = field(default_factory=lambda: uuid4().hex)
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    warnings: tuple[str, ...] = ()
    # Optional stages left out of the run (unticked in the step list).
    skip_stages: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        pipeline,
        parameters,
        *,
        operation="analysis",
        stages=(),
        revision="",
        warnings=(),
        skip_stages=(),
    ):
        params = deepcopy(parameters)
        job_id = uuid4().hex
        source = params.get("sequence_path") or params.get("image_path")
        if source is None and isinstance(params.get("image"), (str, Path)):
            source = str(params["image"])
        if source is None and params.get("camera") is not None:
            source = f"camera:{params['camera']}"
        if source is None and isinstance(params.get("image"), np.ndarray):
            source = f"memory:{job_id}"
        return cls(
            pipeline.lower(),
            str(source) if source is not None else None,
            operation,
            params,
            tuple(stages or ()),
            revision,
            job_id=job_id,
            warnings=tuple(warnings),
            skip_stages=tuple(skip_stages or ()),
        )

    def metadata(self):
        return {
            "job_id": self.job_id,
            "pipeline": self.pipeline,
            "submitted_at": self.submitted_at.isoformat(),
            "operation": self.operation,
            "source": self.source,
            "stages": list(self.stages),
            "skipped_stages": list(self.skip_stages),
            "settings": json_settings(self.parameters),
        }


@dataclass(frozen=True)
class RunCompletion:
    request: RunRequest
    state: RunState
    ctx: Any = None
    error: str | None = None
    value: Any = None
    warnings: tuple[str, ...] = ()


class _Job(QRunnable):
    def __init__(self, request, token, started, finished, task=None):
        super().__init__()
        self.request, self.token = request, token
        self.started, self.finished, self.task = started, finished, task

    def run(self):
        request = self.request
        try:
            with cancellation_scope(self.token):
                self.started.emit(request.job_id)
                parameters = deepcopy(request.parameters)
                if self.task is not None:
                    value = self.task(parameters, self.token)
                    completion = RunCompletion(request, "completed", value=value)
                else:
                    completion = execute_request(request, self.token)
            self.token.check()
        except AnalysisCancelled:
            completion = RunCompletion(request, "cancelled")
        except Exception as exc:
            completion = RunCompletion(request, "failed", error=str(exc))
        self.finished.emit(completion)


def execute_request(request, token):
    """Execute one owned analysis inside the caller cancellation scope."""
    token.check()
    parameters = deepcopy(request.parameters)
    warnings = list(request.warnings)
    if parameters.pop("auto_calibrate", False):
        from menipy.gui.services.calibration_service import (
            prepare_stage_calibration,
        )

        parameters, auto_warnings = prepare_stage_calibration(
            request.pipeline, parameters
        )
        warnings.extend(auto_warnings)
    pipeline = _pick(request.pipeline)(
        preprocessing_settings=parameters.get("preprocessing_settings"),
        edge_detection_settings=parameters.get("edge_detection_settings"),
    )
    parameters["cancellation_token"] = token
    parameters["measurement_id"] = request.job_id
    from menipy.common.runtime_provenance import runtime_provenance

    provenance_at_start = runtime_provenance()
    provenance = parameters.pop("calibration_provenance", None)
    if request.stages:
        ctx = pipeline.run_with_plan(
            only=list(request.stages),
            include_prereqs=True,
            skip_stages=list(request.skip_stages),
            **parameters,
        )
    elif request.skip_stages:
        ctx = pipeline.run(skip_stages=list(request.skip_stages), **parameters)
    else:
        ctx = pipeline.run(**parameters)
    from menipy.gui.services.calibration_provenance import describe_calibration

    ctx.execution_provenance = provenance_at_start
    parameters["calibration_provenance"] = provenance
    ctx.calibration_provenance = describe_calibration(parameters, ctx, warnings)
    warnings = ctx.calibration_provenance.warnings
    return RunCompletion(request, "completed", ctx=ctx, warnings=tuple(warnings))


def _pick(name):
    pipeline = PIPELINE_MAP.get(name.lower())
    if pipeline is None:
        raise PipelineError(f"Unknown pipeline '{name}'")
    return pipeline


class PipelineRunner(QObject):
    finished = Signal(object)
    state_changed = Signal(str, str)
    _started = Signal(str)
    _finished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._active = None
        self._token = None
        self._closing = False
        self._started.connect(self._on_started, Qt.QueuedConnection)
        self._finished.connect(self._on_finished, Qt.QueuedConnection)

    @property
    def busy(self):
        return self._active is not None

    def submit(self, request: RunRequest, task: Callable | None = None):
        if self.busy or self._closing:
            raise PipelineError(
                "An operation is already running or the window is closing."
            )
        if task is None:
            _pick(request.pipeline)
        request = deepcopy(request)
        self._active, self._token = request, CancellationToken()
        job = _Job(request, self._token, self._started, self._finished, task)
        self.state_changed.emit(request.job_id, "queued")
        try:
            self.pool.start(job)
        except Exception as exc:
            completion = RunCompletion(
                request, "failed", error=f"Submission failed: {exc}"
            )
            QTimer.singleShot(0, lambda: self._on_finished(completion))
        return request.job_id

    def run(self, pipeline, image=None, camera=None, frames=1, **parameters):
        return self.submit(
            RunRequest.create(
                pipeline, dict(parameters, image=image, camera=camera, frames=frames)
            )
        )

    def run_subset(
        self, pipeline, *, only, image=None, camera=None, frames=1, **parameters
    ):
        return self.submit(
            RunRequest.create(
                pipeline,
                dict(parameters, image=image, camera=camera, frames=frames),
                stages=only,
            )
        )

    def cancel(self, job_id=None):
        if self._active is None or (
            job_id is not None and self._active.job_id != job_id
        ):
            return
        if not self._token.cancelled:
            self._token.cancel()
            self.state_changed.emit(self._active.job_id, "stopping")

    def shutdown(self):
        self._closing = True
        self.cancel()

    @Slot(str)
    def _on_started(self, job_id):
        if self._active and self._active.job_id == job_id and not self._token.cancelled:
            self.state_changed.emit(job_id, "running")

    @Slot(object)
    def _on_finished(self, completion):
        if self._active is None or completion.request.job_id != self._active.job_id:
            return
        if self.pool.activeThreadCount():
            QTimer.singleShot(5, lambda: self._on_finished(completion))
            return
        if self._token.cancelled:
            completion = replace(
                completion, state="cancelled", ctx=None, value=None, error=None
            )
        self._active = self._token = None
        self.state_changed.emit(completion.request.job_id, completion.state)
        self.finished.emit(completion)
