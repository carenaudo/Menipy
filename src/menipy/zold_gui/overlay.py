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


def _draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    *,
    dash: int = 5,
    thickness: int = 1,
) -> None:
    """Draw a dashed line on ``img``."""
    p1 = np.array(p1, float)
    p2 = np.array(p2, float)
    length = np.linalg.norm(p2 - p1)
    if length == 0:
        return
    direction = (p2 - p1) / length
    n = int(length // dash)
    for i in range(0, n, 2):
        start = p1 + direction * (i * dash)
        end = p1 + direction * (min((i + 1) * dash, length))
        cv2.line(
            img,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            color,
            thickness,
            lineType=cv2.LINE_8,
        )


def draw_drop_overlay(
    image: np.ndarray,
    contour: np.ndarray | None = None,
    *,
    diameter_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    axis_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    contact_line: tuple[tuple[int, int], tuple[int, int]] | np.ndarray | None = None,
    apex: tuple[int, int] | None = None,
    contact_pts: tuple[tuple[int, int], tuple[int, int]] | None = None,
    center_pt: tuple[int, int] | None = None,
    center_apex_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    center_contact_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
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
    contact_pts:
        Optional pair of contact points ``(P1, P2)``.
    center_pt:
        Center of the maximum radius line.
    center_apex_line:
        Optional dashed line from ``center_pt`` to the apex.
    center_contact_line:
        Optional dashed line from ``center_pt`` to the contact line center.
    """
    canvas = image.copy()
    if contour is not None:
        cv2.drawContours(canvas, [np.round(contour).astype(np.int32)], -1, (0, 0, 255), 2)
    if diameter_line is not None:
        cv2.line(canvas, diameter_line[0], diameter_line[1], (255, 0, 0), 2)
    if axis_line is not None:
        cv2.line(canvas, axis_line[0], axis_line[1], (0, 0, 255), 2)
    if contact_line is not None:
        if isinstance(contact_line, np.ndarray):
            pts = np.round(contact_line).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (0, 165, 255), 2)
        else:
            cv2.line(canvas, contact_line[0], contact_line[1], (0, 165, 255), 2)
    if apex is not None:
        cv2.circle(canvas, apex, 3, (255, 255, 0), -1)
    if contact_pts is not None:
        for pt in contact_pts:
            cv2.circle(canvas, tuple(pt), 3, (255, 255, 0), -1)
    if center_pt is not None:
        cv2.circle(canvas, center_pt, 3, (255, 255, 255), -1)
    if center_apex_line is not None:
        _draw_dashed_line(canvas, center_apex_line[0], center_apex_line[1], (255, 255, 0))
    if center_contact_line is not None:
        _draw_dashed_line(canvas, center_contact_line[0], center_contact_line[1], (255, 255, 0))
    return QPixmap.fromImage(_to_qimage(canvas))

