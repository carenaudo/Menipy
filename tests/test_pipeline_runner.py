import pytest
from menipy.pipelines.discover import PIPELINE_MAP
from menipy.gui.services.pipeline_runner import _pick
from menipy.pipelines.base import PipelineBase


def test_discover_pipelines_not_empty():
    """Ensures that the central PIPELINE_MAP is populated."""
    assert (
        PIPELINE_MAP
    ), "PIPELINE_MAP should not be empty. Check imports in discover.py."
    for name, cls in PIPELINE_MAP.items():
        assert isinstance(name, str)
        assert issubclass(cls, PipelineBase)


@pytest.mark.parametrize("pipeline_name", PIPELINE_MAP.keys())
def test_pick_known_pipelines(pipeline_name):
    """Verify that the runner's _pick function can find all discovered pipelines."""
    pipeline_class = _pick(pipeline_name)
    assert pipeline_class is not None
    assert issubclass(pipeline_class, PipelineBase)
    assert pipeline_class == PIPELINE_MAP[pipeline_name]
