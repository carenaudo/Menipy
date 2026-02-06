"""Overlay management logic extracted from MainController."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple, Any

from PySide6.QtCore import QRectF, QPointF
from PySide6.QtGui import QColor
import numpy as np

if TYPE_CHECKING:
    from menipy.gui.views.image_view import ImageView

logger = logging.getLogger(__name__)


class OverlayManager:
    """Manages drawing of overlays on the ImageView."""

    def __init__(self, image_view: ImageView, settings: Any):
        """Initialize.

        Parameters
        ----------
        image_view : type
        Description.
        settings : type
        Description.
        """
        self.image_view = image_view
        self.settings = settings

    def clear_calibration_overlays(self):
        """Clears all calibration-related overlays."""
        if not self.image_view:
            return
        tags = (
            'roi', 'needle', 'contact_line', 
            'cal_roi', 'cal_needle', 'cal_substrate', 'cal_drop', 
            'cal_contact_left', 'cal_contact_right'
        )
        for tag in tags:
            try:
                self.image_view.remove_overlay(tag)
            except Exception:
                pass

    def draw_calibration_result(self, result: Any, controller=None):
        """Draws calibration results (ROI, needle, substrate, drop)."""
        if not self.image_view or not result:
            return

        self.clear_calibration_overlays()

        # Draw ROI
        if result.roi_rect:
            x, y, w, h = result.roi_rect
            self.image_view.add_marker_rect(
                QRectF(x, y, w, h),
                color=QColor(255, 255, 0),
                width=2.0,
                tag='roi'
            )

        # Draw needle
        if result.needle_rect:
            x, y, w, h = result.needle_rect
            self.image_view.add_marker_rect(
                QRectF(x, y, w, h),
                color=QColor(0, 0, 255),
                width=2.0,
                tag='needle'
            )

        # Draw substrate line
        if result.substrate_line:
            p1, p2 = result.substrate_line
            self.image_view.add_marker_line(
                QPointF(p1[0], p1[1]),
                QPointF(p2[0], p2[1]),
                color=QColor(255, 0, 255),
                tag='contact_line'
            )

        # Draw contact points
        if result.contact_points:
            left, right = result.contact_points
            self.image_view.add_marker_point(
                QPointF(left[0], left[1]),
                color=QColor(255, 0, 0),
                radius=5.0,
                tag='cal_contact_left'
            )
            self.image_view.add_marker_point(
                QPointF(right[0], right[1]),
                color=QColor(255, 0, 0),
                radius=5.0,
                tag='cal_contact_right'
            )

        # Draw drop contour
        if result.drop_contour is not None:
            contour = np.asarray(result.drop_contour)
            if contour.size > 0:
                self.image_view.add_marker_contour(
                    contour,
                    color=QColor(0, 255, 0),
                    width=2.0,
                    tag='cal_drop'
                )

    def draw_edge_detection_preview(self, metadata: dict):
        """Draws edge detection preview overlays (contour, contact points)."""
        if not self.image_view or not isinstance(metadata, dict):
            return

        # Clear existing
        for tag in ('detected_left', 'detected_right', 'detected_contour'):
            try:
                self.image_view.remove_overlay(tag)
            except Exception:
                pass

        # Draw contour
        contour_xy = metadata.get('contour_xy')
        if contour_xy is not None:
            self._draw_contour(contour_xy)

        # Draw contact points
        contact_points = metadata.get('contact_points')
        if contact_points is not None:
            self._draw_contact_points(contact_points)

    def _draw_contour(self, contour_xy):
        try:
            overlay_cfg = getattr(self.settings, 'overlay_config', None) or {}
            c_visible = bool(overlay_cfg.get('contour_visible', True))
            if c_visible:
                c_color = QColor(overlay_cfg.get('contour_color', '#ff0000'))
                c_width = float(overlay_cfg.get('contour_thickness', 2.0))
                c_dashed = bool(overlay_cfg.get('contour_dashed', False))
                dash_len = float(overlay_cfg.get('contour_dash_length', 6.0))
                dash_space = float(overlay_cfg.get('contour_dash_space', 6.0))
                c_alpha = float(overlay_cfg.get('contour_alpha', 1.0))
                c_color.setAlphaF(max(0.0, min(1.0, c_alpha)))
                dash = (dash_len, dash_space) if c_dashed else None
                self.image_view.add_marker_contour(
                    contour_xy, 
                    color=c_color, 
                    width=c_width, 
                    dash_pattern=dash, 
                    alpha=c_alpha, 
                    tag='detected_contour'
                )
        except Exception:
             logger.debug('Failed to add contour overlay', exc_info=True)

    def _draw_contact_points(self, contact_points):
        try:
            left_pt, right_pt = contact_points
            p_cfg = getattr(self.settings, 'overlay_config', None) or {}
            p_visible = bool(p_cfg.get('points_visible', True))
            if not p_visible:
                return

            p_color = QColor(p_cfg.get('point_color', '#00ff00')) # Default green/red mix in original code
            # But let's stick to config or defaults
            p_alpha = float(p_cfg.get('point_alpha', 1.0))
            p_radius = float(p_cfg.get('point_radius', 4.0))
            
            # Left point
            if left_pt is not None:
                color = QColor(p_cfg.get('point_color', '#00ff00')) # Use config color
                color.setAlphaF(max(0.0, min(1.0, p_alpha)))
                self.image_view.add_marker_point(
                    QPointF(left_pt[0], left_pt[1]), 
                    color=color, 
                    radius=p_radius, 
                    tag='detected_left'
                )
            
            # Right point
            if right_pt is not None:
                color = QColor(p_cfg.get('point_color', '#ff0000')) # Original code had red for right?
                # Actually original code:
                    # Left: #00ff00 (Green)
                # Right: #ff0000 (Red)
                # But it read from config 'point_color' for both? 
                # Original code:
                    # left: p_color = QColor(p_cfg.get('point_color', '#00ff00'))
                # right: p_color = QColor(p_cfg.get('point_color', '#ff0000'))
                # This suggests if 'point_color' is in config, it overrides BOTH to the same color?
                # Or maybe the default is different if key missing.
                # Let's replicate exact behavior.
                
                color = QColor(p_cfg.get('point_color', '#ff0000'))
                color.setAlphaF(max(0.0, min(1.0, p_alpha)))
                self.image_view.add_marker_point(
                    QPointF(right_pt[0], right_pt[1]), 
                    color=color, 
                    radius=p_radius, 
                    tag='detected_right'
                )
        except Exception:
            logger.debug('Failed to add contact point overlays', exc_info=True)
