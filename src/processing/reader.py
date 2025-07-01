"""Image reading utilities for Menipy."""
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_image(path: Union[str, Path], as_gray: bool = False) -> np.ndarray:
    """Load an image from disk.

    Parameters
    ----------
    path:
        Path to the image file.
    as_gray:
        If ``True``, return a single-channel grayscale image.

    Returns
    -------
    np.ndarray
        The loaded image array.
    """
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image

