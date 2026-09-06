"""Behavioral coverage for A01–A03; all work and persistence are isolated."""

import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QMainWindow

from menipy.common.cancellation import (
    AnalysisCancelled,
    CancellationToken,
    cancellation_scope,
)
from menipy.gui.controllers.pipeline_controller import PipelineController
from menipy.gui.services.pipeline_runner import (
    PipelineRunner,
    RunCompletion,
    RunRequest,
)
from menipy.gui.viewmodels.run_vm import RunViewModel
from menipy.models.context import Context
from menipy.pipelines.base import PipelineBase


@pytest.fixture
def runner(qtbot):
    service = PipelineRunner()
    yield service
    service.cancel()
    qtbot.waitUntil(lambda: not service.busy, timeout=5000)


def test_worker_heartbeat_owned_inputs_and_single_submission(runner, qtbot):
    inputs = {"image": np.ones((8, 8)), "settings": {"value": [3]}}
    request = RunRequest.create("sessile", inputs)
    received = []
    beats = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: beats.append(time.perf_counter()))
    timer.start()
    gui_thread = QThread.currentThread()

    def work(params, token):
        assert QThread.currentThread() != gui_thread
        time.sleep(0.5)
        assert params["settings"]["value"] == [3]
        assert params["image"][0, 0] == 1
        return "done"

    runner.finished.connect(received.append)
    start = time.perf_counter()
    runner.submit(request, work)
    inputs["settings"]["value"][0] = 99
    inputs["image"][:] = 0
    with pytest.raises(Exception, match="already running"):
        runner.submit(request, work)
    qtbot.waitUntil(lambda: bool(received), timeout=3000)
    timer.stop()
    assert received[0].value == "done"
    assert len(beats) >= 5
    assert any(start + 0.05 < beat < start + 0.45 for beat in beats)
    assert "image" not in request.metadata()["settings"]


@pytest.mark.parametrize(
    "when", ["before_start", "cooperative", "native_call", "completion_race"]
)
def test_cancellation_is_one_terminal_without_results(runner, qtbot, when):
    entered, release, returned = Event(), Event(), Event()
    outcomes, states = [], []
    runner.finished.connect(outcomes.append)
    runner.state_changed.connect(lambda _, state: states.append(state))

    def work(params, token):
        entered.set()
        while not release.wait(0.005):
            if when == "cooperative":
                token.check()
        returned.set()
        return "must not publish"

    request = RunRequest.create("sessile", {})
    runner.submit(request, work)
    if when != "before_start":
        qtbot.waitUntil(entered.is_set)
    if when == "completion_race":
        release.set()
        assert returned.wait(1)  # Do not deliver the queued completion yet.
    runner.cancel()
    runner.cancel()
    if when == "native_call":
        qtbot.wait(30)
        assert runner.busy
        assert not outcomes
    release.set()
    qtbot.waitUntil(lambda: bool(outcomes))
    assert len(outcomes) == 1
    assert outcomes[0].state == "cancelled"
    assert outcomes[0].ctx is None and outcomes[0].value is None
    assert states.count("stopping") == 1
    assert states.count("cancelled") == 1


def test_duplicate_old_completion_cannot_clear_new_job(runner, qtbot):
    outcomes = []
    runner.finished.connect(outcomes.append)
    first = RunRequest.create("sessile", {})
    runner.submit(first, lambda params, token: None)
    qtbot.waitUntil(lambda: len(outcomes) == 1)
    release = Event()
    second = RunRequest.create("pendant", {})
    runner.submit(second, lambda params, token: release.wait(1))
    runner._on_finished(outcomes[0])
    assert runner.busy
    assert len(outcomes) == 1
    release.set()
    qtbot.waitUntil(lambda: len(outcomes) == 2)


def test_submission_failure_is_reported_without_execution(runner, monkeypatch, qtbot):
    outcomes, ran = [], []
    runner.finished.connect(outcomes.append)

    def fail(*args):
        raise RuntimeError("pool unavailable")

    monkeypatch.setattr(runner.pool, "start", fail)
    runner.submit(RunRequest.create("sessile", {}), lambda *_: ran.append(True))
    qtbot.waitUntil(lambda: bool(outcomes))
    assert outcomes[0].state == "failed"
    assert "pool unavailable" in outcomes[0].error
    assert not ran and not runner.busy


@pytest.fixture
def controller(qtbot, runner):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.settings = SimpleNamespace(pipeline_settings={})
    selection = {
        "name": "sessile",
        "image": "original.png",
        "analysis_params": {"angle": 2},
    }
    setup = SimpleNamespace(
        gather_run_params=lambda: selection.copy(),
        current_pipeline_name=lambda: selection["name"],
        get_calibration_params=lambda: {"needle_diameter_mm": 1.0},
    )
    displayed, stored = [], []
    preview = SimpleNamespace(
        roi_rect=lambda: (0, 0, 8, 8),
        needle_rect=lambda: (0, 0, 2, 4),
        contact_line_segment=lambda: None,
        display_context=displayed.append,
    )
    panel = SimpleNamespace(
        add_measurement=lambda measurement, **kwargs: stored.append(
            (measurement, kwargs)
        )
    )
    vm = RunViewModel(runner)
    ctrl = PipelineController(
        window,
        setup,
        preview,
        panel,
        None,
        None,
        {"sessile": PipelineBase, "pendant": PipelineBase},
        None,
        vm,
        None,
    )
    vm.completed.connect(ctrl.on_completed)
    return ctrl, selection, displayed, stored


@pytest.mark.parametrize("rejected", [False, True])
@pytest.mark.parametrize("changed", [None, "pipeline", "source", "settings"])
def test_history_uses_submitted_identity_and_validation(controller, rejected, changed):
    ctrl, selection, displayed, stored = controller
    request = RunRequest.create(
        "sessile",
        {"image": "original.png", "analysis_params": {"angle": 2}},
        revision=ctrl._revision(),
    )
    if changed == "pipeline":
        selection["name"] = "pendant"
    elif changed == "source":
        selection["image"] = "new.png"
    elif changed == "settings":
        selection["analysis_params"] = {"angle": 8}
    ctx = Context(
        results={"diameter_mm": 123},
        qa={"ok": not rejected, "rejection_reasons": ["bad_fit"] if rejected else []},
    )
    completion = RunCompletion(request, "completed", ctx=ctx)
    ctrl.on_completed(completion)
    ctrl.on_completed(completion)
    assert len(stored) == 1
    measurement, options = stored[0]
    assert measurement.id == request.job_id
    assert measurement.pipeline == "sessile"
    assert measurement.file_path == "original.png"
    assert measurement.accepted is not rejected
    assert measurement.results == ({} if rejected else {"diameter_mm": 123})
    assert measurement.run_metadata["settings"]["analysis_params"] == {"angle": 2}
    assert bool(displayed) == (changed is None)
    assert options["activate"] == (changed is None)


def test_empty_rejection_is_persisted_but_empty_stage_is_not(controller):
    ctrl, _, _, stored = controller
    for accepted in (True, False):
        req = RunRequest.create("sessile", {}, revision=ctrl._revision())
        ctrl.on_completed(
            RunCompletion(req, "completed", ctx=Context(qa={"ok": accepted}))
        )
    assert len(stored) == 1
    assert not stored[0][0].accepted


def test_dynamic_summary_has_one_history_record(controller):
    ctrl, _, _, stored = controller
    req = RunRequest.create(
        "sessile_dynamic", {"sequence_path": "sequence"}, revision=ctrl._revision()
    )
    ctx = Context(results={"series": [1, 2], "n_valid_frames": 20}, qa={"ok": True})
    ctrl.on_completed(RunCompletion(req, "completed", ctx=ctx))
    assert stored[0][0].results == {"n_valid_frames": 20}
    assert ctx.results["series"] == [1, 2]


def test_core_cancellation_stops_before_next_stage():
    token = CancellationToken()
    called = []

    class Pipeline(PipelineBase):
        def build_plan(self, **kwargs):
            def first(ctx):
                token.cancel()

            def second(ctx):
                called.append(True)

            return [("first", first), ("second", second)]

    with pytest.raises(AnalysisCancelled):
        Pipeline().run(cancellation_token=token)
    assert not called
    assert "cancellation_token" not in Context(cancellation_token=token).model_dump()


def test_solver_cancellation_escapes_scientific_fallbacks():
    from menipy.math.young_laplace import young_laplace_ode

    token = CancellationToken()
    with pytest.raises(AnalysisCancelled):
        with cancellation_scope(token):
            token.cancel()
            young_laplace_ode(np.array([1.0, 0.1]), {})


def test_video_cancellation_releases_capture(monkeypatch, tmp_path):
    from menipy.common import sequence_acquisition as acquisition

    source = tmp_path / "video.avi"
    source.touch()
    token = CancellationToken()
    released = []

    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 30.0

        def read(self):
            token.cancel()
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            released.append(True)

    monkeypatch.setattr(acquisition.cv2, "VideoCapture", lambda *_: Capture())
    with pytest.raises(AnalysisCancelled):
        acquisition.load_video(source, check_cancelled=token.check)
    assert released == [True]


def test_temporal_cancellation_stops_detector_loop(monkeypatch):
    from menipy.common import temporal_sessile
    from menipy.models.frame import Frame

    token = CancellationToken()
    calls = []

    def detect(*args):
        calls.append(True)
        token.cancel()
        return None

    monkeypatch.setattr(temporal_sessile, "auto_detect_features", detect)
    with pytest.raises(AnalysisCancelled):
        temporal_sessile.analyze_dynamic_sessile(
            [Frame(image=np.zeros((8, 8), dtype=np.uint8)) for _ in range(3)],
            None,
            px_per_mm=1,
            needle_diameter_mm=1,
            check_cancelled=token.check,
        )
    assert calls == [True]


def test_submission_warnings_survive_worker_completion(runner, qtbot, monkeypatch):
    from menipy.gui.services.pipeline_runner import PIPELINE_MAP

    class Pipeline(PipelineBase):
        def run(self, **kwargs):
            return Context()

    monkeypatch.setitem(PIPELINE_MAP, "sessile", Pipeline)
    outcomes = []
    runner.finished.connect(outcomes.append)
    runner.submit(RunRequest.create("sessile", {}, warnings=["fallback scale"]))
    qtbot.waitUntil(lambda: bool(outcomes))
    assert outcomes[0].warnings == ("fallback scale",)


def test_stage_calibration_keeps_fallback_warning_when_source_is_missing():
    from menipy.gui.services.calibration_service import prepare_stage_calibration

    _, warnings = prepare_stage_calibration("sessile", {})
    assert "Needle width was unavailable; using fallback scale." in warnings
