"""
Overlay rendering and display utilities.
"""
import numpy as np
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF, Qt

def draw_analysis_overlay(image, metrics):
    """Return a ``QPixmap`` with analysis overlays.

    This is a generic overlay function that can draw metrics for any
    pipeline that produces a compatible metrics object (e.g., PendantMetrics,
    SessileMetrics).
    """
    center_pt = getattr(metrics, "diameter_center", None)
    contact_line = getattr(metrics, "contact_line", None)
    center_apex_line = None
    center_contact_line = None

    if center_pt is not None:
        center_apex_line = (center_pt, metrics.apex)
        if contact_line is not None:
            cl_center = (
                (contact_line[0][0] + contact_line[1][0]) // 2,
                (contact_line[0][1] + contact_line[1][1]) // 2,
            )
            center_contact_line = (center_pt, cl_center)

    return draw_drop_overlay(
        image,
        contour=getattr(metrics, "contour", None),
        diameter_line=getattr(metrics, "diameter_line", None),
        contact_line=contact_line,
        apex=getattr(metrics, "apex", None),
        contact_pts=contact_line,
        center_pt=center_pt,
        center_apex_line=center_apex_line,
        center_contact_line=center_contact_line,
    )

def draw_drop_overlay(
    image: np.ndarray,
    contour: np.ndarray | None = None,
    diameter_line: tuple | None = None,
    contact_line: tuple | None = None,
    apex: tuple | None = None,
    contact_pts: tuple | None = None,
    center_pt: tuple | None = None,
    center_apex_line: tuple | None = None,
    center_contact_line: tuple | None = None,
) -> QPixmap:
    """Draws geometric overlays on an image."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:  # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
    else:
        pixmap = image.copy()

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # --- Draw Contour ---
    if contour is not None and len(contour) > 1:
        pen = QPen(QColor("red"), 2, Qt.SolidLine)
        painter.setPen(pen)
        poly = [QPointF(p[0][0], p[0][1]) for p in contour]
        painter.drawPolyline(poly)

    # --- Draw Lines ---
    line_pen = QPen(QColor("cyan"), 1, Qt.DashLine)
    painter.setPen(line_pen)
    if diameter_line:
        p1, p2 = diameter_line
        painter.drawLine(QPointF(*p1), QPointF(*p2))
    if contact_line:
        pen = QPen(QColor("lightgreen"), 2, Qt.DashLine)
        painter.setPen(pen)
        p1, p2 = contact_line
        painter.drawLine(QPointF(*p1), QPointF(*p2))

    # --- Draw Center Lines ---
    center_line_pen = QPen(QColor("magenta"), 1, Qt.DotLine)
    painter.setPen(center_line_pen)
    if center_apex_line:
        p1, p2 = center_apex_line
        painter.drawLine(QPointF(*p1), QPointF(*p2))
    if center_contact_line:
        p1, p2 = center_contact_line
        painter.drawLine(QPointF(*p1), QPointF(*p2))

    # --- Draw Points ---
    painter.setBrush(QBrush(Qt.NoBrush))
    if apex:
        pen = QPen(QColor("yellow"), 2)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(*apex), 4, 4)
    if contact_pts:
        pen = QPen(QColor("yellow"), 2)
        painter.setPen(pen)
        for pt in contact_pts:
            painter.drawEllipse(QPointF(*pt), 3, 3)
    if center_pt:
        pen = QPen(QColor("magenta"), 2)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(*center_pt), 3, 3)

    painter.end()
    return pixmap
