"""Image format conversion utilities for Qt."""

# src/menipy/gui/services/image_convert.py
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def to_pixmap(bgr: np.ndarray) -> QPixmap:
    """to pixmap.

    Parameters
    ----------
    bgr : type
        Description.

    Returns
    -------
    type
        Description.
    """
    h, w = bgr.shape[:2]
    if bgr.ndim == 2:
        qimg = QImage(bgr.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    rgb = bgr[..., ::-1].copy()
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)
