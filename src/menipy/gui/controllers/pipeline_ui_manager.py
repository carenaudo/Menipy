"""Pipeline UI Manager for plugin-centric dynamic UI generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from menipy.common.plugin_db import PluginDB

if TYPE_CHECKING:
    from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class PipelineUIManager:
    """Manages dynamic pipeline UI generation and configuration based on plugin metadata."""

    def __init__(self, plugin_db: PluginDB | None = None):
        """Initialize.

        Parameters
        ----------
        plugin_db : type
        Description.
        """
        self.plugin_db = plugin_db or PluginDB()
        self._pipeline_metadata_cache: dict[str, dict] = {}
        self._refresh_metadata_cache()

    def _refresh_metadata_cache(self) -> None:
        """Refresh the pipeline metadata cache from PluginDB."""
        try:
            metadata_list = self.plugin_db.list_pipeline_metadata()
            self._pipeline_metadata_cache = {
                meta["pipeline_name"]: meta for meta in metadata_list
            }
            logger.debug(f"Loaded {len(metadata_list)} pipeline metadata entries")
        except Exception as e:
            logger.warning(f"Failed to load pipeline metadata: {e}")
            self._pipeline_metadata_cache = {}

    def get_pipeline_metadata(self, pipeline_name: str) -> dict | None:
        """Get metadata for a specific pipeline."""
        return self._pipeline_metadata_cache.get(pipeline_name)

    def get_all_pipeline_metadata(self) -> list[dict]:
        """Get metadata for all pipelines."""
        return list(self._pipeline_metadata_cache.values())

    def register_pipeline_metadata(
        self,
        pipeline_name: str,
        display_name: str,
        icon: str | None = None,
        color: str | None = None,
        stages: list[str] | None = None,
        calibration_params: list[str] | None = None,
        primary_metrics: list[str] | None = None,
    ) -> None:
        """Register pipeline metadata in the PluginDB."""
        try:
            self.plugin_db.upsert_pipeline_metadata(
                pipeline_name=pipeline_name,
                display_name=display_name,
                icon=icon,
                color=color,
                stages=stages,
                calibration_params=calibration_params,
                primary_metrics=primary_metrics,
            )
            # Refresh cache after registration
            self._refresh_metadata_cache()
            logger.info(f"Registered metadata for pipeline: {pipeline_name}")
        except Exception as e:
            logger.error(
                f"Failed to register pipeline metadata for {pipeline_name}: {e}"
            )

    def get_required_stages(self, pipeline_name: str) -> list[str]:
        """Get the list of required stages for a pipeline."""
        metadata = self.get_pipeline_metadata(pipeline_name)
        return metadata.get("stages", []) if metadata else []

    def get_calibration_params(self, pipeline_name: str) -> list[str]:
        """Get the list of calibration parameters for a pipeline."""
        metadata = self.get_pipeline_metadata(pipeline_name)
        return metadata.get("calibration_params", []) if metadata else []

    def get_primary_metrics(self, pipeline_name: str) -> list[str]:
        """Get the list of primary metrics for a pipeline."""
        metadata = self.get_pipeline_metadata(pipeline_name)
        return metadata.get("primary_metrics", []) if metadata else []

    def get_display_info(self, pipeline_name: str) -> dict:
        """Get display information (name, icon, color) for a pipeline."""
        metadata = self.get_pipeline_metadata(pipeline_name)
        if metadata:
            return {
                "display_name": metadata.get("display_name", pipeline_name.title()),
                "icon": metadata.get("icon"),
                "color": metadata.get("color", "#6c757d"),  # Default gray
            }
        return {"display_name": pipeline_name.title(), "icon": None, "color": "#6c757d"}

    def initialize_default_metadata(self) -> None:
        """Initialize default metadata for built-in pipelines by reading from pipeline classes."""
        from menipy.pipelines.discover import PIPELINE_MAP

        for pipeline_name, pipeline_cls in PIPELINE_MAP.items():
            if hasattr(pipeline_cls, "ui_metadata"):
                metadata = pipeline_cls.ui_metadata.copy()
                metadata["pipeline_name"] = pipeline_name
                try:
                    self.register_pipeline_metadata(**metadata)
                    logger.info(f"Registered UI metadata for pipeline: {pipeline_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to register metadata for {pipeline_name}: {e}"
                    )
            else:
                logger.warning(f"Pipeline {pipeline_name} has no ui_metadata attribute")

    def validate_pipeline_config(self, pipeline_name: str, config: dict) -> list[str]:
        """Validate pipeline configuration against metadata requirements."""
        errors = []
        metadata = self.get_pipeline_metadata(pipeline_name)

        if not metadata:
            errors.append(f"No metadata found for pipeline: {pipeline_name}")
            return errors

        # Check required stages
        required_stages = metadata.get("stages", [])
        if required_stages:
            # This would be expanded to check actual stage availability
            pass

        # Check calibration parameters
        required_cal_params = metadata.get("calibration_params", [])
        if required_cal_params:
            # This would be expanded to validate calibration parameter values
            pass

        return errors
