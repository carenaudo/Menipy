"""Controller smoke tests exercise the actual asynchronous execution boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMainWindow, QMessageBox

from menipy.gui.controllers.pipeline_controller import PipelineController
from menipy.gui.services.pipeline_runner import PIPELINE_MAP, PipelineRunner
from menipy.gui.viewmodels.run_vm import RunViewModel
from menipy.pipelines.base import PipelineBase


@pytest.fixture
def flow(qtbot, monkeypatch):
    window = QMainWindow()
    qtbot.addWidget(window)
    window.settings = SimpleNamespace(
        pipeline_settings={}, acquisition_requires_contact_line=False
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    selected = {"name": "sessile", "image": image, "frames": 1}
    setup = SimpleNamespace(
        gather_run_params=lambda: selected.copy(),
        current_pipeline_name=lambda: selected["name"],
        get_calibration_params=lambda: {
            "needle_diameter_mm": 0.5,
            "drop_density_kg_m3": 900,
            "fluid_density_kg_m3": 2,
        },
        collect_included_stages=lambda: ["acquisition", "physics"],
    )
    displayed, stored, errors, finished, contexts = [], [], [], [], []
    preview = SimpleNamespace(
        roi_rect=lambda: (0, 0, 12, 12),
        needle_rect=lambda: (0, 0, 5, 5),
        contact_line_segment=lambda: ((0, 10), (12, 10)),
        display_context=displayed.append,
    )
    results = SimpleNamespace(add_measurement=lambda m, **kw: stored.append(m))
    gui_thread = QThread.currentThread()

    class RecordingPipeline(PipelineBase):
        def build_plan(self, only=None, include_prereqs=True):
            def acquire(ctx):
                assert QThread.currentThread() != gui_thread
                contexts.append(ctx)

            def metrics(ctx):
                ctx.results = {"diameter_mm": 2.0}
                ctx.qa = {"ok": True}

            stages = [
                ("acquisition", acquire),
                ("preprocessing", lambda c: c),
                ("contour_extraction", lambda c: c),
                ("geometric_features", metrics),
                ("physics", lambda c: c),
                ("validation", lambda c: c),
            ]
            if only:
                names = [n for n, _ in stages]
                stages = stages[: max(names.index(n) for n in only) + 1]
            return stages

    monkeypatch.setitem(PIPELINE_MAP, "sessile", RecordingPipeline)
    monkeypatch.setitem(PIPELINE_MAP, "pendant", RecordingPipeline)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    runner = PipelineRunner(window)
    vm = RunViewModel(runner)
    ctrl = PipelineController(
        window,
        setup,
        preview,
        results,
        None,
        None,
        {"sessile": RecordingPipeline, "pendant": RecordingPipeline},
        object(),
        vm,
        None,
    )
    vm.completed.connect(ctrl.on_completed)
    runner.finished.connect(finished.append)
    yield SimpleNamespace(
        ctrl=ctrl,
        runner=runner,
        selected=selected,
        displayed=displayed,
        stored=stored,
        errors=errors,
        finished=finished,
        contexts=contexts,
        preview=preview,
    )
    runner.cancel()
    qtbot.waitUntil(lambda: not runner.busy)


@pytest.mark.parametrize(
    "entry,args",
    [
        ("run_full", ()),
        ("run_simple_analysis", ()),
        ("run_all", ()),
        ("run_stage", ("acquisition",)),
        ("run_stage", ("geometric_features",)),
        ("run_stage", ("edge_detection",)),
        ("run_stage", ("physics",)),
        ("test_stage", ("preprocessing", {})),
    ],
)
def test_every_entry_uses_worker_and_submitted_context(
    flow, qtbot, monkeypatch, entry, args
):
    from menipy.gui.services import calibration_service

    monkeypatch.setattr(
        calibration_service,
        "prepare_stage_calibration",
        lambda name, params: (params, ["test warning"]),
    )
    job = getattr(flow.ctrl, entry)(*args)
    assert isinstance(job, str)
    assert flow.runner.busy
    qtbot.waitUntil(lambda: bool(flow.finished))
    assert not flow.errors
    completion = flow.finished[0]
    assert completion.state == "completed"
    assert completion.request.job_id == job
    ctx = flow.contexts[0]
    assert ctx.scale == {"px_per_mm": 10.0}
    assert ctx.physics["rho1"] == 900 and ctx.physics["rho2"] == 2
    assert ctx.roi == (0, 0, 12, 12)
    assert ctx.measurement_id == job
    assert ctx.image.shape == (12, 12, 3)
    if entry == "run_all":
        # Every stage runs; only the unticked optional stage is left out.
        assert completion.request.stages == ()
        assert completion.request.skip_stages == ("overlay",)
    if entry == "test_stage":
        assert completion.warnings == ("test warning",)
    assert len(flow.stored) == bool(ctx.results)
    assert len(flow.displayed) == 1


def test_quick_analysis_uses_selected_canonical_pipeline(flow, qtbot):
    flow.selected["name"] = "pendant"
    flow.ctrl.run_simple_analysis()
    qtbot.waitUntil(lambda: bool(flow.finished))
    assert flow.stored[0].pipeline == "pendant"


def test_pendant_approximator_selection_reaches_context(flow, qtbot):
    flow.selected["name"] = "pendant"
    flow.ctrl.window.settings.pipeline_settings = {
        "pendant_approximation_methods": ["volume_apex_lookup", "clothoid_zones"],
    }
    flow.ctrl.run_full()
    qtbot.waitUntil(lambda: bool(flow.finished))
    assert flow.finished[0].state == "completed"
    assert flow.contexts[0].pendant_approximation_methods == [
        "volume_apex_lookup",
        "clothoid_zones",
    ]


def test_stage_test_physics_comes_from_setup_panel(flow):
    flow.ctrl.setup_ctrl.get_calibration_params = lambda: {
        "needle_diameter_mm": 0.5,
        "drop_density_kg_m3": 998.2,
        "fluid_density_kg_m3": 1.2,
        "g": 9.79,
    }
    # A stale sandbox physics object must not override the setup panel values.
    sandbox = {"physics_params": SimpleNamespace(g=1.0)}
    _, _, run_kwargs, _ = flow.ctrl._build_pipeline_run_kwargs(sandbox_config=sandbox)
    assert run_kwargs["physics"] == {"rho1": 998.2, "rho2": 1.2, "g": 9.79}


def test_pipeline_settings_apply_only_to_their_pipeline(flow, qtbot):
    from menipy.gui.services.settings_service import AppSettings

    settings = AppSettings(path=None)
    settings.set_pipeline_settings(
        "pendant",
        {
            "pendant_approximation_methods": ["clothoid_zones"],
            "pendant_approximator_settings": {"minimize_adsa": {"maxiter": 7}},
        },
    )
    # Saving the sessile dialog afterwards must not replace the pendant settings.
    settings.set_pipeline_settings("sessile", {"contact_angle_method": "circle_fit"})
    flow.ctrl.window.settings = settings

    for pipeline in ("pendant", "sessile"):
        flow.selected["name"] = pipeline
        flow.ctrl.run_full()
        qtbot.waitUntil(lambda: not flow.runner.busy)

    pendant_ctx, sessile_ctx = flow.contexts
    assert pendant_ctx.pendant_approximation_methods == ["clothoid_zones"]
    assert pendant_ctx.pendant_approximator_settings == {"minimize_adsa": {"maxiter": 7}}
    assert pendant_ctx.contact_angle_method == "tangent"  # Context default
    assert sessile_ctx.contact_angle_method == "circle_fit"
    assert sessile_ctx.pendant_approximation_methods is None


def test_per_pipeline_settings_persist_and_legacy_fallback(tmp_path):
    from types import SimpleNamespace

    from menipy.gui.services.settings_service import (
        AppSettings,
        pipeline_settings_for,
    )

    legacy = SimpleNamespace(pipeline_settings={"contact_angle_method": "circle_fit"})
    assert pipeline_settings_for(legacy, "pendant") == legacy.pipeline_settings

    path = tmp_path / "settings.json"
    settings = AppSettings(path=path)
    settings.set_pipeline_settings("pendant", {"pendant_approximation_methods": []})
    settings.save()
    loaded = AppSettings.load(path)
    assert pipeline_settings_for(loaded, "pendant") == {
        "pendant_approximation_methods": []
    }
    # Once settings are stored per pipeline, others no longer inherit them.
    assert pipeline_settings_for(loaded, "sessile") == {}


@pytest.mark.parametrize(
    "failure", ["missing_runner", "unknown_pipeline", "missing_roi"]
)
def test_invalid_submission_reports_error_without_running(flow, failure):
    if failure == "missing_runner":
        flow.ctrl.run_vm = None
    elif failure == "unknown_pipeline":
        flow.selected["name"] = "unknown"
    else:
        flow.preview.roi_rect = lambda: None
    assert flow.ctrl.run_full() is None
    assert flow.errors
    assert not flow.contexts and not flow.runner.busy


def test_pipeline_failure_does_not_publish_or_retry(flow, qtbot, monkeypatch):
    calls = []

    class Broken(PipelineBase):
        def run(self, **kwargs):
            calls.append(True)
            raise ValueError("synthetic failure")

    monkeypatch.setitem(PIPELINE_MAP, "sessile", Broken)
    flow.ctrl.run_full()
    qtbot.waitUntil(lambda: bool(flow.finished))
    assert flow.finished[0].state == "failed"
    assert len(calls) == 1
    assert flow.errors and not flow.stored and not flow.displayed
