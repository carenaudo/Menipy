"""Temporary, lossless frame storage with bounded decoded-pixel memory."""

from collections.abc import Sequence
from tempfile import TemporaryFile

import numpy as np

from .frame import Frame


class DiskFrameStore(Sequence):
    """Own one temporary file; reads return independent arrays, never memmaps.

    The owning pipeline closes the store after analysis. TemporaryFile also
    closes on finalization if acquisition is run as a standalone stage.
    """

    def __init__(self) -> None:
        self._file = TemporaryFile(mode="w+b")
        self._offsets: list[int] = []
        self.timestamps_s: list[float] = []
        self.shape: tuple[int, ...] | None = None

    def append(self, image: np.ndarray) -> None:
        if self.shape is not None and image.shape[:2] != self.shape[:2]:
            raise ValueError("sequence_variable_dimensions")
        self.shape = image.shape
        self._file.seek(0, 2)
        offset = self._file.tell()
        np.save(self._file, image, allow_pickle=False)
        self._offsets.append(offset)

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        self._file.seek(self._offsets[index])
        return Frame(
            image=np.load(self._file, allow_pickle=False),
            ms_from_start=self.timestamps_s[index] * 1000.0,
        )

    def close(self) -> None:
        self._file.close()

    @property
    def closed(self) -> bool:
        return self._file.closed
