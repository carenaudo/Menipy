from __future__ import annotations

"""GUI controller for Menipy preprocessing pipeline."""

from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np
from PySide6.QtCore import QObject, Signal

from menipy.models.datatypes import (
    Context,
    PreprocessingSettings,
    PreprocessingState,
    MarkerSet,
)
from menipy.common import preprocessing

logger = logging.getLogger(__name__)


class PreprocessingPipelineController(QObject):
    """Coordinates preprocessing settings, execution, and history management."""

    settingsChanged = Signal(object)
    stateChanged = Signal(object)
    markersChanged = Signal(object)
    previewReady = Signal(object, dict)
    errorOccurred = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settings = PreprocessingSettings()
        self._markers = MarkerSet()
        self._state: Optional[PreprocessingState] = None
        self._history: list[PreprocessingState] = []
        self._redo: list[PreprocessingState] = []

        self._image: Optional[np.ndarray] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None
        self._roi_mask: Optional[np.ndarray] = None
        self._contact_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def settings(self) -> PreprocessingSettings:
        return self._settings

    def set_settings(self, data: PreprocessingSettings | Dict[str, Any]) -> None:
        if isinstance(data, PreprocessingSettings):
            self._settings = data
        else:
            self._settings = PreprocessingSettings(**data)
        self.settingsChanged.emit(self._settings)

    def settings_dict(self) -> Dict[str, Any]:
        return self._settings.model_dump()

    @property
    def markers(self) -> MarkerSet:
        return self._markers

    def update_markers(self, markers: MarkerSet | Dict[str, Any]) -> None:
        if isinstance(markers, MarkerSet):
            self._markers = markers
        else:
            try:
                self._markers = MarkerSet(**markers)
            except Exception as exc:
                logger.warning("Invalid markers payload: %s", exc)
                return
        self.markersChanged.emit(self._markers)

    def set_source(
        self,
        image: np.ndarray,
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,
        roi_mask: Optional[np.ndarray] = None,
        contact_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("source image must be a numpy array")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.copy()
        self._image = image
        self._roi = tuple(map(int, roi)) if roi else None
        self._roi_mask = roi_mask.copy() if isinstance(roi_mask, np.ndarray) else None
        self._contact_line = contact_line
        self._history.clear()
        self._redo.clear()
        self._state = None

    def update_geometry(
        self,
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,
        roi_mask: Optional[np.ndarray] = None,
        contact_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> None:
        if roi is not None:
            self._roi = tuple(map(int, roi))
        if roi_mask is not None:
            self._roi_mask = roi_mask.copy()
        if contact_line is not None:
            self._contact_line = contact_line
        if roi is not None or roi_mask is not None or contact_line is not None:
            self._history.clear()
            self._redo.clear()
            self._state = None

    def has_source(self) -> bool:
        return self._image is not None

    def run(self) -> Optional[PreprocessingState]:
        if self._image is None:
            self.errorOccurred.emit("No source image available for preprocessing")
            return None
        ctx = Context()
        ctx.frame = self._image
        if self._roi:
            ctx.roi = self._roi
        if self._roi_mask is not None:
            ctx.roi_mask = self._roi_mask
        if self._contact_line:
            ctx.contact_line = self._contact_line
        if self._markers:
            ctx.preprocessing_markers = self._markers.model_copy(deep=True)
        ctx.preprocessing_settings = self._settings

        try:
            preprocessing.run(ctx, self._settings)
        except Exception as exc:  # pragma: no cover - guard for runtime issues
            logger.exception("Preprocessing execution failed: %s", exc)
            self.errorOccurred.emit(str(exc))
            return None

        fresh_state = ctx.preprocessed_state.clone()
        self._state = fresh_state
        self._history.append(fresh_state.clone())
        self._redo.clear()

        self.stateChanged.emit(self._state)
        if ctx.preprocessed is not None:
            metadata = {"roi": self._state.roi_bounds, "scale": self._state.scale}
            if fresh_state.metadata.get("roi_resized"):
                metadata["roi_resized_shape"] = fresh_state.metadata.get("roi_resized_shape")
            self.previewReady.emit(ctx.preprocessed, metadata)
        return self._state

    def undo(self) -> Optional[PreprocessingState]:
        if len(self._history) <= 1:
            return None
        latest = self._history.pop()
        self._redo.append(latest)
        self._state = self._history[-1].clone()
        self.stateChanged.emit(self._state)
        self._emit_preview_from_state()
        return self._state

    def redo(self) -> Optional[PreprocessingState]:
        if not self._redo:
            return None
        state = self._redo.pop()
        self._history.append(state.clone())
        self._state = state.clone()
        self.stateChanged.emit(self._state)
        self._emit_preview_from_state()
        return self._state

    def current_state(self) -> Optional[PreprocessingState]:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit_preview_from_state(self) -> None:
        if self._image is None or self._state is None:
            return
        roi = self._state.roi_bounds
        if roi is None:
            return
        x, y, w, h = roi
        roi_image = self._state.normalized_roi or self._state.filtered_roi or self._state.working_roi or self._state.raw_roi
        if roi_image is None:
            return
        payload: np.ndarray
        metadata = {"roi": roi, "scale": self._state.scale}
        if roi_image.shape[:2] == (h, w) and self._image is not None:
            preview = self._image.copy()
            preview[y : y + h, x : x + w] = roi_image
            payload = preview
        else:
            metadata["roi_resized_shape"] = (roi_image.shape[1], roi_image.shape[0])
            payload = roi_image
        self.previewReady.emit(payload, metadata)



