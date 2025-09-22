"""Utility plotting functions for droplet contours."""

from __future__ import annotations

import numpy as np
import cv2


def _side_contours(contour: np.ndarray, apex_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return left and right contour points split at the apex.

    Parameters
    ----------
    contour:
        External contour points ``(x, y)``.
    apex_idx:
        Index of the apex point within ``contour``.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")
    if not (0 <= apex_idx < len(contour)):
        raise ValueError("apex_idx out of range")

    x_min = int(np.floor(contour[:, 0].min()))
    y_min = int(np.floor(contour[:, 1].min()))
    x_max = int(np.ceil(contour[:, 0].max()))
    y_max = int(np.ceil(contour[:, 1].max()))

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = np.round(contour - [x_min, y_min]).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    axis_x = int(round(contour[apex_idx, 0])) - x_min
    left_edges = np.argmax(mask > 0, axis=1)
    right_edges = mask.shape[1] - 1 - np.argmax(mask[:, ::-1] > 0, axis=1)
    rows = np.where(mask.sum(axis=1) > 0)[0]

    y_coords = rows + y_min
    left = np.column_stack([left_edges[rows] + x_min, y_coords])
    right = np.column_stack([right_edges[rows] + x_min, y_coords])
    return left.astype(float), right.astype(float)


def save_contour_sides_image(
    contour: np.ndarray, apex_idx: int, out_path: str, format: str | None = None
) -> None:
    """Save a plot of contour sides split at the apex.

    ``out_path`` determines the image format (png/jpg, etc.).
    """
    import importlib

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is required for plotting")

    import matplotlib

    matplotlib.use("Agg")  # ensure headless backend
    import matplotlib.pyplot as plt

    left, right = _side_contours(contour, apex_idx)

    fig, ax = plt.subplots()
    ax.plot(contour[:, 0], contour[:, 1], "k-", linewidth=1, label="contour")
    ax.plot(left[:, 0], left[:, 1], "r-", linewidth=2, label="left")
    ax.plot(right[:, 0], right[:, 1], "b-", linewidth=2, label="right")
    ax.plot(
        contour[apex_idx, 0],
        contour[apex_idx, 1],
        "go",
        markersize=4,
        label="apex",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend()

    fig.savefig(out_path, format=format)
    plt.close(fig)


from datetime import datetime
from pathlib import Path


def save_contour_side_profiles(
    contour: np.ndarray, apex_idx: int, out_dir: str, fmt: str = "png"
) -> list[str]:
    """Save separate left and right profile plots.

    Parameters
    ----------
    contour:
        External contour points ``(x, y)``.
    apex_idx:
        Index of the apex point within ``contour``.
    out_dir:
        Directory where the images will be saved.
    fmt:
        Image format (``"png"`` or ``"jpg"``).
    """

    import importlib

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is required for plotting")

    import matplotlib

    matplotlib.use("Agg")  # ensure headless backend
    import matplotlib.pyplot as plt

    left, right = _side_contours(contour, apex_idx)
    out_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for data, side, color in [(left, "left", "r"), (right, "right", "b")]:
        fig, ax = plt.subplots()
        ax.plot(data[:, 0], data[:, 1], f"{color}-", linewidth=2, label=side)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend()
        file_path = out_path / f"{ts}_{side}.{fmt}"
        fig.savefig(file_path, format=fmt)
        out_paths.append(str(file_path))
        plt.close(fig)
    return out_paths


__all__ = ["save_contour_sides_image", "save_contour_side_profiles"]
