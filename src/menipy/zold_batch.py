"""Batch processing orchestration for Menipy."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .zold_processing.reader import load_image
from .zold_processing.segmentation import (
    otsu_threshold,
    morphological_cleanup,
    external_contour_mask,
    find_contours,
)


def _iter_image_paths(directory: Path) -> Iterable[Path]:
    """Yield paths to supported image files in ``directory``."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in exts:
            yield path


def run_batch(directory: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Process all images in ``directory`` and return results as a DataFrame.

    Parameters
    ----------
    directory:
        Folder containing images to process.
    output_csv:
        Optional path to save the aggregated results as CSV.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``filename``, ``area`` and ``num_contours``.
    """

    dir_path = Path(directory)
    records: list[dict[str, object]] = []
    for img_path in _iter_image_paths(dir_path):
        image = load_image(img_path, as_gray=True)
        mask = otsu_threshold(image)
        mask = morphological_cleanup(mask, kernel_size=3, iterations=1)
        mask = external_contour_mask(mask)
        contours = find_contours(mask)
        area = int((mask > 0).sum())
        records.append(
            {
                "filename": img_path.name,
                "area": area,
                "num_contours": len(contours),
            }
        )

    df = pd.DataFrame.from_records(records)
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return df


