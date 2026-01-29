"""
Interactive Image Viewer

A widget for displaying images with interactive tool support (zooming, panning,
drawing lines/ROIs).
"""
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame, QSizePolicy
)

from menipy.gui import theme


class InteractiveImageViewer(QWidget):
    """
    Image viewer that supports zoom, pan, and interactive drawing tools.
    
    Tools:
    - None: Navigation only (pan/zoom)
    - Line: Draw a line (click point A, drag/click point B)
    - Rect: Draw a rectangle (not yet implemented)
    
    Signals:
        line_drawn(start_point, end_point, distance_px): Emitted when a line is drawn
        clicked(point): Emitted on click
    """
    
    line_drawn = Signal(QPointF, QPointF, float)
    clicked = Signal(QPointF)
    overlay_painted = Signal(object, object, float)  # painter, visible_rect, zoom
    
    TOOL_NONE = "none"
    TOOL_LINE = "line"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._tool = self.TOOL_NONE
        
        # Interaction state
        self._last_mouse_pos = QPointF()
        self._is_panning = False
        self._is_drawing = False
        self._draw_start = QPointF()
        self._draw_current = QPointF()
        self._static_line: tuple[QPointF, QPointF] | None = None
        
        self.setMouseTracking(True)
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the UI components."""
        self.setStyleSheet(f"background-color: {theme.BG_TERTIARY};")
        
        # We draw directly on the widget, so no layout needed for basics,
        # but we use a specialized paint event.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_static_line(self, p1: QPointF, p2: QPointF):
        """Set a static line to display (e.g. from auto-detect)."""
        self._static_line = (p1, p2)
        self.update()
        
    def set_image(self, pixmap: QPixmap | None):
        """Set the image to display."""
        self._pixmap = pixmap
        if pixmap:
            # Reset view to fit
            self.fit_to_view()
        self.update()
        
    def set_tool(self, tool: str):
        """Set the active tool."""
        self._tool = tool
        if tool == self.TOOL_LINE:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
    def set_zoom(self, zoom: float):
        """Set zoom level manually."""
        self._zoom = max(0.1, min(20.0, zoom))
        self.update()
        
    def fit_to_view(self):
        """Fit image to current widget size."""
        if self._pixmap and not self._pixmap.isNull() and self.width() > 0 and self.height() > 0:
            w_ratio = self.width() / self._pixmap.width()
            h_ratio = self.height() / self._pixmap.height()
            self._zoom = min(w_ratio, h_ratio) * 0.95
            
            # Center image
            self._view_offset = QPointF(
                (self.width() - self._pixmap.width() * self._zoom) / 2,
                (self.height() - self._pixmap.height() * self._zoom) / 2
            )
            self.update()
            
    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    
    def paintEvent(self, event):
        """Custom paint event."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(theme.BG_TERTIARY))
        
        if self._pixmap and not self._pixmap.isNull():
            # Apply transforms
            # For simplicity, we calculate target rect
            # We assume zooming is centered or handled via simple scaling for now
            # A proper implementation would handle offset (pan) + zoom
            
            # Use stored offset if we have panning logic, otherwise center
            if not hasattr(self, '_view_offset'):
                self.fit_to_view()
                
            # Draw image
            target_rect = QRectF(
                self._view_offset.x(),
                self._view_offset.y(),
                self._pixmap.width() * self._zoom,
                self._pixmap.height() * self._zoom
            )
            painter.drawPixmap(target_rect.toRect(), self._pixmap)
            
            # Emit overlay signal
            self.overlay_painted.emit(painter, target_rect, self._zoom)
            
            # Draw active drawing (e.g. line being dragged)
            if self._is_drawing and self._tool == self.TOOL_LINE:
                pen = QPen(QColor(theme.ACCENT_BLUE))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(self._draw_start, self._draw_current)
                
                # Draw coordinates text
                dist_px = self._image_dist(self._draw_start, self._draw_current)
                mid = (self._draw_start + self._draw_current) / 2
                painter.drawText(mid.toPoint(), f"{dist_px:.1f} px")
                
            elif self._static_line is not None and self._tool == self.TOOL_LINE:
                # Draw static line (stored in IMAGE coordinates)
                p1_img, p2_img = self._static_line
                
                # Transform to view coordinates
                p1 = self._view_offset + p1_img * self._zoom
                p2 = self._view_offset + p2_img * self._zoom
                
                pen = QPen(QColor(theme.ACCENT_BLUE))
                pen.setWidth(2)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(p1, p2)
                
                # Distance is just the length in image coords (which is what we want)
                dist_px = ((p1_img.x() - p2_img.x())**2 + (p1_img.y() - p2_img.y())**2)**0.5
                mid = (p1 + p2) / 2
                painter.drawText(mid.toPoint(), f"{dist_px:.1f} px")

        else:
            painter.setPen(QColor(theme.TEXT_SECONDARY))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")

    def mousePressEvent(self, event: QMouseEvent):
        if not self._pixmap:
            return
            
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool == self.TOOL_LINE:
                self._is_drawing = True
                self._draw_start = event.position()
                self._draw_current = event.position()
                self.update()
            else:
                self._is_panning = True
                self._last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_drawing:
            self._draw_current = event.position()
            self.update()
        elif self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self._view_offset += delta
            self.update()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_drawing and self._tool == self.TOOL_LINE:
                self._is_drawing = False
                self._draw_current = event.position()
                self.update()
                
                # Emit result mapped to image coordinates?
                # For simplicity, sending widget coords involves complexity in receiver.
                # Let's send PIXEL distance on image.
                dist = self._image_dist(self._draw_start, self._draw_current)
                if dist > 0:
                    self.line_drawn.emit(self._draw_start, self._draw_current, dist)
                    
            elif self._is_panning:
                self._is_panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom."""
        if not self._pixmap:
            return
            
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        
        # Zoom centered on cursor would be better, but center is safer for MVP
        self._zoom *= factor
        self._zoom = max(0.1, min(20.0, self._zoom))
        
        # Adjust offset to keep center?
        # For now, simple zoom
        self.update()
        
    def zoom_in(self):
        """Zoom in."""
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self):
        """Zoom out."""
        self.set_zoom(self._zoom / 1.25)
        
    def reset_view(self):
        """Reset zoom and fit."""
        self.fit_to_view()

    def _image_dist(self, p1: QPointF, p2: QPointF) -> float:
        """Calculate distance in IMAGE pixels (accounting for zoom)."""
        widget_dist = ((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2)**0.5
        return widget_dist / self._zoom
