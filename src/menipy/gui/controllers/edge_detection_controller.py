from __future__ import annotations

"""GUI controller for Menipy edge detection pipeline."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from menipy.models.config import EdgeDetectionSettings
from menipy.models.context import Context
from menipy.common import edge_detection
from menipy.common.geometry import find_contact_points_from_contour

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


class EdgeDetectionPipelineController(QObject):
    """Coordinates edge detection settings, execution, and history management."""

    settingsChanged = Signal(object)
    previewRequested = Signal(object, dict)
    errorOccurred = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settings = EdgeDetectionSettings()
        self._source_image: Optional[np.ndarray] = None
        self._contact_line: Optional[tuple] = None
        self._last_contact_points: tuple | None = None

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

    def set_contact_line(self, contact_line: tuple | None) -> None:
        """Set a user-drawn contact line to be considered during edge detection preview."""
        self._contact_line = contact_line

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
            # Provide contact line to the preprocessing/edge detection context if available
            if self._contact_line is not None:
                ctx.contact_line = self._contact_line
            edge_detection.run(ctx, self._settings)
        except Exception as exc:
            logger.exception("Edge detection execution failed: %s", exc)
            self.errorOccurred.emit(str(exc))
            return

        # Safely access contour and its xy attribute (avoid attribute access on None)
        contour = getattr(ctx, 'contour', None)
        contour_xy = getattr(contour, 'xy', None) if contour is not None else None
        # Prepare a preview image (do not bake overlay markers into it)
        preview_image = self._source_image.copy()
        if preview_image.ndim == 2:
            if cv2 is not None:
                preview_image = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)
            else:
                preview_image = np.stack((preview_image,) * 3, axis=-1)

        if contour_xy is None:
            logger.warning("Edge detection did not return a contour.")
            # Emit source image and empty metadata
            self.previewRequested.emit(preview_image, {})
            return

        # If a contact line was provided, attempt to detect contact points using curvature
        try:
            contact_line = getattr(ctx, 'contact_line', None)
            if contact_line is not None:
                xy = np.asarray(contour_xy, dtype=float)
                left_pt, right_pt = find_contact_points_from_contour(xy, contact_line)
                self._last_contact_points = (left_pt, right_pt)
                logger.info("Detected contact points: %s %s", left_pt, right_pt)
        except Exception:
            logger.debug("Contact point detection failed", exc_info=True)

        # Emit preview image and metadata (contour points + detected contact points)
        metadata = {
            "contact_points": self._last_contact_points,
            "contour_xy": np.asarray(contour_xy, dtype=float),
        }
        self.previewRequested.emit(preview_image, metadata)

    def reset(self) -> None:
        self._settings = EdgeDetectionSettings()
        self.settingsChanged.emit(self._settings)
        self.run()

