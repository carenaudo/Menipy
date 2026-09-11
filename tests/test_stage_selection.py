"""Stage selection: canonical names, optional stages and what a run executes."""

from __future__ import annotations

import pytest

from menipy.gui.services import pipeline_runner
from menipy.gui.services.pipeline_runner import RunRequest, execute_request
from menipy.models.context import Context
from menipy.pipelines.base import PipelineBase, PipelineError, canonical_stage_name
from menipy.pipelines.discover import PIPELINE_MAP


class RecordingPipeline(PipelineBase):
    """Base stages that only record their own execution."""

    name = "recording"

    def _call_stage(self, ctx: Context, stage_name: str, fn):
        ctx.timings_ms[stage_name] = 0.0
        return ctx


def _ran(ctx: Context) -> list[str]:
    return list(ctx.timings_ms)


def test_legacy_stage_names_map_to_canonical_ones():
    assert canonical_stage_name("edge_detection") == "contour_extraction"
    assert canonical_stage_name("outputs") == "compute_metrics"
    assert canonical_stage_name("optimization") is None
    assert canonical_stage_name("overlay") == "overlay"


@pytest.mark.parametrize("name", sorted(PIPELINE_MAP))
def test_only_overlay_is_optional_in_every_pipeline(name):
    cls = PIPELINE_MAP[name]
    stages = cls.stage_names()
    assert "overlay" in stages and stages[-1] == "validation"
    assert set(cls.OPTIONAL_STAGES) == {"overlay"}
    # Legacy selections: only a missing optional stage is skipped.
    assert cls.skipped_stages(["acquisition", "outputs"]) == ["overlay"]
    assert cls.skipped_stages(stages) == []


def test_run_skips_optional_stage_and_refuses_required_ones():
    pipeline = RecordingPipeline()
    assert "overlay" not in _ran(pipeline.run(skip_stages=["overlay"]))
    assert _ran(pipeline.run())[-2:] == ["overlay", "validation"]
    with pytest.raises(PipelineError, match="Required stages"):
        pipeline.run(skip_stages=["preprocessing"])


def test_run_up_to_stage_accepts_legacy_names_and_skip():
    pipeline = RecordingPipeline()
    ran = _ran(pipeline.run_with_plan(only=["geometry"], skip_stages=["overlay"]))
    assert ran[-1] == "geometric_features"
    ran = _ran(pipeline.run_with_plan(only=["validation"], skip_stages=["overlay"]))
    assert "overlay" not in ran and ran[-1] == "validation"
    with pytest.raises(PipelineError, match="None of the requested stages"):
        pipeline.run_with_plan(only=["optimization"])


def test_worker_leaves_out_skipped_stages(monkeypatch):
    monkeypatch.setitem(PIPELINE_MAP, "recording", RecordingPipeline)
    monkeypatch.setattr(pipeline_runner, "PIPELINE_MAP", PIPELINE_MAP)
    request = RunRequest.create("recording", {}, skip_stages=("overlay",))
    completion = execute_request(request, pipeline_runner.CancellationToken())
    assert "overlay" not in _ran(completion.ctx)
    assert "validation" in _ran(completion.ctx)
    assert request.metadata()["skipped_stages"] == ["overlay"]
