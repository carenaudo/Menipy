from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class ImageView(QGraphicsView):
    """Graphics view that automatically fits a pixmap to the viewport."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._pixmap_orig: QPixmap | None = None
        self._scale = 1.0
        self._zoom = 1.0

    @property
    def pixmap_item(self) -> QGraphicsPixmapItem | None:
        return self._pixmap_item

    @property
    def scale_factor(self) -> float:
        """Return the current view-to-image scale factor."""
        return self._scale * self._zoom

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set ``pixmap`` as the displayed image."""
        self.scene().clear()
        self._pixmap_orig = pixmap
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._fit_pixmap()

    # ------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._fit_pixmap()

    # ------------------------------------------------------------------
    def set_zoom(self, factor: float) -> None:
        self._zoom = factor
        self._update_transform()

    # ------------------------------------------------------------------
    def _fit_pixmap(self) -> None:
        if not self._pixmap_item:
            return
        view_rect = self.viewport().rect()
        if view_rect.isEmpty():
            return
        pm = self._pixmap_item.pixmap()
        scale_x = view_rect.width() / pm.width()
        scale_y = view_rect.height() / pm.height()
        self._scale = min(scale_x, scale_y, 1.0)
        self._update_transform()

    def _update_transform(self) -> None:
        factor = self._scale * self._zoom
        transform = QTransform().scale(factor, factor)
        self.setTransform(transform)

    # ------------------------------------------------------------------
    def view_to_image(self, pt_view: QPointF) -> QPointF:
        """Convert a point from view coordinates to image coordinates."""
        return pt_view / self.scale_factor

    def image_to_view(self, pt_img: QPointF) -> QPointF:
        """Convert a point from image coordinates to view coordinates."""
        return pt_img * self.scale_factor
