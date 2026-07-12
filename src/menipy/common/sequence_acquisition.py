"""Deterministic video and image-sequence loading for temporal pipelines."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from menipy.models.frame import Frame
from menipy.models.temporal import SequenceMetadata

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}


class SequenceAcquisitionError(ValueError):
    """Raised when a temporal source cannot produce a scientifically timed sequence."""


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _digest_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_images(images: list[np.ndarray]) -> tuple[int, int]:
    if not images:
        raise SequenceAcquisitionError("sequence_no_frames")
    height, width = images[0].shape[:2]
    if width <= 0 or height <= 0:
        raise SequenceAcquisitionError("sequence_invalid_dimensions")
    if any(image.shape[:2] != (height, width) for image in images):
        raise SequenceAcquisitionError("sequence_variable_dimensions")
    return width, height


def load_image_sequence(path: str | Path, *, fps: float | None) -> tuple[list[Frame], SequenceMetadata]:
    """Load a naturally sorted directory as one sequence, never as a batch."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SequenceAcquisitionError("sequence_directory_missing")
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise SequenceAcquisitionError("sequence_fps_required")
    paths = sorted(
        (candidate for candidate in root.iterdir() if candidate.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )
    images: list[np.ndarray] = []
    for candidate in paths:
        image = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SequenceAcquisitionError(f"sequence_corrupt_image:{candidate.name}")
        images.append(image)
    width, height = _validate_images(images)
    timestamps = [index / float(fps) for index in range(len(images))]
    frames = [Frame(image=image, ms_from_start=timestamp * 1000.0) for image, timestamp in zip(images, timestamps)]
    metadata = SequenceMetadata(
        source_type="image_sequence",
        source_id=str(root),
        sha256=_digest_files(paths),
        width=width,
        height=height,
        fps=float(fps),
        timestamps_s=timestamps,
        frame_count=len(frames),
    )
    return frames, metadata


def load_video(path: str | Path) -> tuple[list[Frame], SequenceMetadata]:
    """Decode a video and preserve monotonic container timestamps when available."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SequenceAcquisitionError("video_missing")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SequenceAcquisitionError("video_open_failed")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    images: list[np.ndarray] = []
    timestamps: list[float] = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            images.append(image)
            timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    finally:
        capture.release()
    width, height = _validate_images(images)
    if not np.isfinite(fps) or fps <= 0:
        raise SequenceAcquisitionError("video_fps_invalid")
    monotonic = len(timestamps) == len(images) and all(
        timestamps[index] > timestamps[index - 1] for index in range(1, len(timestamps))
    )
    if not monotonic:
        timestamps = [index / fps for index in range(len(images))]
    frames = [Frame(image=image, ms_from_start=timestamp * 1000.0) for image, timestamp in zip(images, timestamps)]
    metadata = SequenceMetadata(
        source_type="video",
        source_id=str(source),
        sha256=_digest_files([source]),
        width=width,
        height=height,
        fps=fps,
        timestamps_s=timestamps,
        frame_count=len(frames),
    )
    return frames, metadata


def frames_from_memory(frames: list[Frame] | list[np.ndarray], *, fps: float | None) -> tuple[list[Frame], SequenceMetadata]:
    """Normalize caller-owned frames using explicit or embedded timing."""
    if not frames:
        raise SequenceAcquisitionError("sequence_no_frames")
    normalized = [frame if isinstance(frame, Frame) else Frame(image=frame) for frame in frames]
    images = [frame.image for frame in normalized]
    width, height = _validate_images(images)
    embedded = [frame.ms_from_start for frame in normalized]
    if all(value is not None for value in embedded):
        timestamps = [float(value) / 1000.0 for value in embedded if value is not None]
        if any(timestamps[index] <= timestamps[index - 1] for index in range(1, len(timestamps))):
            raise SequenceAcquisitionError("sequence_timestamps_not_monotonic")
        inferred_fps = 1.0 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else fps
    else:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise SequenceAcquisitionError("sequence_fps_required")
        inferred_fps = fps
        timestamps = [index / float(fps) for index in range(len(normalized))]
        normalized = [Frame(image=frame.image, timestamp=frame.timestamp, ms_from_start=t * 1000.0, camera=frame.camera, calibration=frame.calibration) for frame, t in zip(normalized, timestamps)]
    if inferred_fps is None or inferred_fps <= 0:
        raise SequenceAcquisitionError("sequence_fps_required")
    digest = hashlib.sha256()
    for image in images:
        digest.update(np.ascontiguousarray(image).tobytes())
    metadata = SequenceMetadata(
        source_type="memory",
        source_id="memory",
        sha256=digest.hexdigest(),
        width=width,
        height=height,
        fps=float(inferred_fps),
        timestamps_s=timestamps,
        frame_count=len(normalized),
    )
    return normalized, metadata
