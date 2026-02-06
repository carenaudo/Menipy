"""Interactive image marking and annotation tools."""

from __future__ import annotations

"""Interactive helpers for preprocessing markers on the preview image."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtGui import QColor

from menipy.gui.controllers.preprocessing_controller import (
    PreprocessingPipelineController,
)
from menipy.gui.panels.preview_panel import PreviewPanel


logger = logging.getLogger(__name__)


@dataclass
class _MarkerTags:
    center: str = "marker_center"
    anchors: List[str] = field(default_factory=list)
    background: List[str] = field(default_factory=list)
    contact_line: str = "marker_contact_line"


class ImageMarkerHelper(QObject):
    """Maps preview interactions into preprocessing markers and overlays."""

    def __init__(
        self,
        preview_panel: PreviewPanel,
        controller: PreprocessingPipelineController,
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        preview_panel : PreviewPanel
            Preview panel widget.
        controller : PreprocessingPipelineController
            Preprocessing pipeline controller.
        parent : QObject, optional
            Parent object.
        """
        super().__init__(parent or preview_panel.panel)
        self._panel = preview_panel
        self._controller = controller
        self._tags = _MarkerTags()

        self._view = getattr(preview_panel, "image_view", None)
        if self._view is None:
            logger.debug(
                "ImageMarkerHelper: preview panel has no image view; helper disabled."
            )
            return

        self._view.point_clicked.connect(self._on_point_clicked)
        self._view.double_clicked.connect(self._on_double_clicked)
        controller.markersChanged.connect(self._on_markers_changed)
        controller.stateChanged.connect(self._on_state_changed)
        self._sync_overlays()

        # ------------------------------------------------------------------
        # Slots
        # ------------------------------------------------------------------
    def _on_point_clicked(self, point: QPointF, button: int, modifiers: int) -> None:
        if self._view is None:
            return
        if button != Qt.LeftButton:
            return
        marker_point = (float(point.x()), float(point.y()))
        mods = Qt.KeyboardModifiers(modifiers)
        markers = self._controller.markers.model_copy(deep=True)
        if mods & Qt.ShiftModifier:
            markers.background_samples.append(marker_point)
            logger.info("Background sample added at %.1f, %.1f", *marker_point)
        else:
            refined = self._refine_center(marker_point)
            markers.drop_center = refined
            logger.info("Droplet center set to %.1f, %.1f", *refined)
        self._controller.update_markers(markers)
        self._rerun_if_ready()

    def _on_double_clicked(self, point: QPointF, button: int, modifiers: int) -> None:
        if self._view is None:
            return
        if button != Qt.LeftButton:
            return
        marker_point = (float(point.x()), float(point.y()))
        markers = self._controller.markers.model_copy(deep=True)
        anchors = markers.contact_line_anchors
        existing_idx = self._find_near_anchor(anchors, marker_point)
        if existing_idx is not None:
            anchors.pop(existing_idx)
            logger.info("Removed contact-line anchor near %.1f, %.1f", *marker_point)
        else:
            anchors.append(marker_point)
            logger.info("Added contact-line anchor at %.1f, %.1f", *marker_point)
        markers.contact_line_anchors = anchors
        self._controller.update_markers(markers)
        self._rerun_if_ready()

    def _on_markers_changed(self, markers) -> None:  # type: ignore[override]
        self._sync_overlays()

    def _on_state_changed(self, state) -> None:  # type: ignore[override]
        # Contact line mask or scale changes may require overlay refresh.
        self._sync_overlays()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rerun_if_ready(self) -> None:
        state = self._controller.current_state()
        if state is None and not self._controller.has_source():
            return
        try:
            self._controller.run()
        except Exception as exc:  # pragma: no cover - guard UI responsiveness
            logger.debug("Preprocessing re-run failed after marker update: %s", exc)

    def _refine_center(self, candidate: Tuple[float, float]) -> Tuple[float, float]:
        state = self._controller.current_state()
        if state is None or state.roi_bounds is None:
            return candidate
        roi = (
            state.normalized_roi
            or state.filtered_roi
            or state.working_roi
            or state.raw_roi
        )
        if roi is None:
            return candidate
        x0, y0, w, h = state.roi_bounds
        local_x = int(round(candidate[0] - x0))
        local_y = int(round(candidate[1] - y0))
        if local_x < 0 or local_y < 0 or local_x >= w or local_y >= h:
            return candidate
        patch_radius = 15
        x_start = max(0, local_x - patch_radius)
        x_end = min(w, local_x + patch_radius + 1)
        y_start = max(0, local_y - patch_radius)
        y_end = min(h, local_y + patch_radius + 1)
        patch = roi[y_start:y_end, x_start:x_end]
        if patch.size == 0:
            return candidate
        if patch.ndim == 3:
            patch_gray = patch.mean(axis=2)
        else:
            patch_gray = patch
        try:
            idx = np.argmin(patch_gray)
            offset_y, offset_x = np.unravel_index(idx, patch_gray.shape)
            refined = (x0 + x_start + float(offset_x), y0 + y_start + float(offset_y))
            return refined
        except Exception:
            return candidate

    def _find_near_anchor(
        self,
        anchors: List[Tuple[float, float]],
        point: Tuple[float, float],
        threshold: float = 8.0,
    ) -> Optional[int]:
        for idx, anchor in enumerate(anchors):
            dx = anchor[0] - point[0]
            dy = anchor[1] - point[1]
            if (dx * dx + dy * dy) ** 0.5 <= threshold:
                return idx
        return None

    def _sync_overlays(self) -> None:
        if self._view is None:
            return
        view = self._view
        # Clear previous overlays
        view.remove_overlay(self._tags.center)
        for tag in self._tags.anchors:
            view.remove_overlay(tag)
        for tag in self._tags.background:
            view.remove_overlay(tag)
        view.remove_overlay(self._tags.contact_line)

        self._tags.anchors.clear()
        self._tags.background.clear()

        markers = self._controller.markers
        if markers.drop_center:
            view.add_marker_point(
                QPointF(*markers.drop_center),
                color=QColor(0, 255, 0),
                tag=self._tags.center,
            )

        for idx, anchor in enumerate(markers.contact_line_anchors):
            tag = f"marker_anchor_{idx}"
            self._tags.anchors.append(tag)
            view.add_marker_point(QPointF(*anchor), color=QColor(255, 200, 0), tag=tag)

        if len(markers.contact_line_anchors) >= 2:
            p1 = QPointF(*markers.contact_line_anchors[0])
            p2 = QPointF(*markers.contact_line_anchors[-1])
            view.add_marker_line(
                p1, p2, color=QColor(255, 140, 0), tag=self._tags.contact_line
            )

        for idx, sample in enumerate(markers.background_samples):
            tag = f"marker_bg_{idx}"
            self._tags.background.append(tag)
            view.add_marker_point(
                QPointF(*sample), color=QColor(100, 149, 237), tag=tag
            )
