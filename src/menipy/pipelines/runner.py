"""Simple runner for executing pipelines non-interactively."""

from __future__ import annotations

from typing import Any, Optional

from menipy.models.context import Context
from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings
from .base import PipelineBase, PipelineError
from .discover import PIPELINE_MAP


class PipelineRunner:
    """
    A simple, non-GUI runner for executing pipelines.

    This can be used for scripting, testing, or in a CLI. It ensures
    that both instantiation-time and run-time settings are correctly
    passed to the pipeline.
    """

    def __init__(
        self,
        pipeline_name: str,
        *,
        preprocessing_settings: Optional[PreprocessingSettings] = None,
        edge_detection_settings: Optional[EdgeDetectionSettings] = None,
    ):
        """Initialize.

        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline to create.
        preprocessing_settings : PreprocessingSettings, optional
            Preprocessing settings.
        edge_detection_settings : EdgeDetectionSettings, optional
            Edge detection settings.
        """
        pipeline_cls = PIPELINE_MAP.get(pipeline_name.lower())
        if not pipeline_cls:
            raise PipelineError(f"Unknown pipeline '{pipeline_name}'")

        self.pipeline: PipelineBase = pipeline_cls(
            preprocessing_settings=preprocessing_settings,
            edge_detection_settings=edge_detection_settings,
        )

    def run(self, **kwargs: Any) -> Context:
        """
        Execute the pipeline with the given runtime arguments.

        Returns:
            The final populated Context object.
        """
        return self.pipeline.run(**kwargs)
