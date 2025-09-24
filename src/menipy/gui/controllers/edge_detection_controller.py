from __future__ import annotations

"""GUI controller for Menipy edge detection pipeline."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from menipy.models.datatypes import EdgeDetectionSettings

logger = logging.getLogger(__name__)


class EdgeDetectionPipelineController(QObject):
    """Coordinates edge detection settings, execution, and history management."""

    settingsChanged = Signal(object)
    previewRequested = Signal(object)
    errorOccurred = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settings = EdgeDetectionSettings()
        self._source_image: Optional[np.ndarray] = None

    def settings(self) -> EdgeDetectionSettings:
        return self._settings

    def set_settings(self, data: EdgeDetectionSettings | Dict[str, Any]) -> None:
        if isinstance(data, EdgeDetectionSettings):
            self._settings = data
        else:
            self._settings = EdgeDetectionSettings(**data)
        self.settingsChanged.emit(self._settings)

    def set_source(self, image: np.ndarray) -> None:
        self._source_image = image

    def has_source(self) -> bool:
        return self._source_image is not None

    def run(self) -> None:
        if self._source_image is None:
            self.errorOccurred.emit("No source image available for edge detection")
            return

        # TODO: Implement actual edge detection logic here
        # For now, just emit a preview request with current settings
        logger.info("Running edge detection with settings: %s", self._settings.model_dump_json())
        self.previewRequested.emit(self._settings)

    def reset(self) -> None:
        self._settings = EdgeDetectionSettings()
        self.settingsChanged.emit(self._settings)
        self.run()

