from __future__ import annotations

"""GUI controller for Menipy edge detection pipeline."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from menipy.models.datatypes import EdgeDetectionSettings, Context
from menipy.common import edge_detection

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

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

    @property
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

        ctx = Context()
        ctx.frame = self._source_image
        ctx.edge_detection_settings = self._settings

        try:
            edge_detection.run(ctx, self._settings)
        except Exception as exc:
            logger.exception("Edge detection execution failed: %s", exc)
            self.errorOccurred.emit(str(exc))
            return

        # Assuming ctx.contour.xy contains the detected contour
        # We need to draw this contour on the original image for preview
        if hasattr(ctx, 'contour') and ctx.contour.xy is not None:
            # Create a blank image or use the source image to draw the contour
            preview_image = self._source_image.copy()
            # Convert to BGR if grayscale for drawing
            if preview_image.ndim == 2:
                preview_image = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)

            # Draw the contour
            # The contour points are (x, y)
            contour_points = ctx.contour.xy.astype(np.int32)
            # Reshape for cv2.drawContours
            contour_points = contour_points.reshape((-1, 1, 2))
            cv2.drawContours(preview_image, [contour_points], -1, (0, 255, 0), 2) # Green contour, thickness 2

            self.previewRequested.emit(preview_image)
        else:
            logger.warning("Edge detection did not return a contour.")
            self.previewRequested.emit(self._source_image)

    def reset(self) -> None:
        self._settings = EdgeDetectionSettings()
        self.settingsChanged.emit(self._settings)
        self.run()

