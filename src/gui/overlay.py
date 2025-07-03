import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def _to_qimage(arr: np.ndarray) -> QImage:
    """Return a QImage wrapping ``arr``."""
    if arr.ndim == 2:
        fmt = QImage.Format_Grayscale8
    else:
        fmt = QImage.Format_BGR888
    return QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], fmt).copy()


def draw_drop_overlay(
    image: np.ndarray,
    contour: np.ndarray | None = None,
    *,
    diameter_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    axis_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    apex: tuple[int, int] | None = None,
) -> QPixmap:
    """Return a ``QPixmap`` of ``image`` with droplet overlays drawn.

    Parameters
    ----------
    image:
        Source image, BGR or grayscale.
    contour:
        Contour points ``(x, y)``.
    diameter_line:
        Optional line for the maximum diameter.
    axis_line:
        Optional symmetry/height axis line.
    apex:
        Optional apex point ``(x, y)``.
    """
    canvas = image.copy()
    if contour is not None:
        cv2.drawContours(canvas, [np.round(contour).astype(np.int32)], -1, (0, 0, 255), 2)
    if diameter_line is not None:
        cv2.line(canvas, diameter_line[0], diameter_line[1], (255, 0, 0), 2)
    if axis_line is not None:
        cv2.line(canvas, axis_line[0], axis_line[1], (0, 0, 255), 2)
    if apex is not None:
        cv2.circle(canvas, apex, 3, (255, 255, 0), -1)
    return QPixmap.fromImage(_to_qimage(canvas))

