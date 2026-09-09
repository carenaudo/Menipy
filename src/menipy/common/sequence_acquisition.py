"""Deterministic video and image-sequence loading for temporal pipelines."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from menipy.common.cancellation import check_cancelled
from menipy.models.frame import Frame
from menipy.models.frame_store import DiskFrameStore
from menipy.models.temporal import SequenceMetadata

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}


class SequenceAcquisitionError(ValueError):
    """Raised when a temporal source cannot produce a scientifically timed sequence."""


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _digest_files(paths: Iterable[Path], check_cancelled=check_cancelled) -> str:
    digest = hashlib.sha256()
    for path in paths:
        check_cancelled()
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                check_cancelled()
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


def load_image_sequence(
    path: str | Path,
    *,
    fps: float | None,
    check_cancelled=check_cancelled,
    disk_backed: bool = False,
) -> tuple[list[Frame] | DiskFrameStore, SequenceMetadata]:
    """Load a naturally sorted directory as one sequence, never as a batch."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SequenceAcquisitionError("sequence_directory_missing")
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise SequenceAcquisitionError("sequence_fps_required")
    paths = sorted(
        (
            candidate
            for candidate in root.iterdir()
            if candidate.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )
    if disk_backed:
        return _load_disk_sequence(root, paths, float(fps), check_cancelled)
    images: list[np.ndarray] = []
    for candidate in paths:
        check_cancelled()
        image = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SequenceAcquisitionError(f"sequence_corrupt_image:{candidate.name}")
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        images.append(image)
    width, height = _validate_images(images)
    timestamps = [index / float(fps) for index in range(len(images))]
    frames = [
        Frame(image=image, ms_from_start=timestamp * 1000.0)
        for image, timestamp in zip(images, timestamps)
    ]
    metadata = SequenceMetadata(
        source_type="image_sequence",
        source_id=str(root),
        sha256=_digest_files(paths, check_cancelled),
        width=width,
        height=height,
        fps=float(fps),
        timestamps_s=timestamps,
        frame_count=len(frames),
    )
    return frames, metadata


def load_video(
    path: str | Path, *, check_cancelled=check_cancelled, disk_backed: bool = False
) -> tuple[list[Frame] | DiskFrameStore, SequenceMetadata]:
    """Decode a video and preserve monotonic container timestamps when available."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise SequenceAcquisitionError("video_missing")
    if disk_backed:
        return _load_disk_sequence(source, None, None, check_cancelled)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise SequenceAcquisitionError("video_open_failed")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    images: list[np.ndarray] = []
    timestamps: list[float] = []
    try:
        while True:
            check_cancelled()
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
    frames = [
        Frame(image=image, ms_from_start=timestamp * 1000.0)
        for image, timestamp in zip(images, timestamps)
    ]
    metadata = SequenceMetadata(
        source_type="video",
        source_id=str(source),
        sha256=_digest_files([source], check_cancelled),
        width=width,
        height=height,
        fps=fps,
        timestamps_s=timestamps,
        frame_count=len(frames),
    )
    return frames, metadata


def _load_disk_sequence(source, paths, fps, check_cancelled):
    """Decode once to scratch storage, preserving the eager loader's timing rules."""
    store = DiskFrameStore()
    capture = None
    timestamps = []
    try:
        if paths is None:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise SequenceAcquisitionError("video_open_failed")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            while True:
                check_cancelled()
                ok, image = capture.read()
                if not ok:
                    break
                store.append(image)
                timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
        else:
            for path in paths:
                check_cancelled()
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise SequenceAcquisitionError(
                        f"sequence_corrupt_image:{path.name}"
                    )
                if image.ndim == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                store.append(image)
        if not store:
            raise SequenceAcquisitionError("sequence_no_frames")
        if not np.isfinite(fps) or fps <= 0:
            raise SequenceAcquisitionError("video_fps_invalid")
        if len(timestamps) != len(store) or not all(
            timestamps[i] > timestamps[i - 1] for i in range(1, len(timestamps))
        ):
            timestamps = [i / fps for i in range(len(store))]
        store.timestamps_s = timestamps
        height, width = store.shape[:2]
        metadata = SequenceMetadata(
            source_type="video" if paths is None else "image_sequence",
            source_id=str(source),
            sha256=_digest_files(paths or [source], check_cancelled),
            width=width,
            height=height,
            fps=fps,
            timestamps_s=timestamps,
            frame_count=len(store),
        )
        return store, metadata
    except ValueError as exc:
        store.close()
        if str(exc) == "sequence_variable_dimensions":
            raise SequenceAcquisitionError(str(exc)) from exc
        raise
    except BaseException:
        store.close()
        raise
    finally:
        if capture is not None:
            capture.release()


def frames_from_memory(
    frames: list[Frame] | list[np.ndarray],
    *,
    fps: float | None,
    check_cancelled=check_cancelled,
) -> tuple[list[Frame], SequenceMetadata]:
    """Normalize caller-owned frames using explicit or embedded timing."""
    if not frames:
        raise SequenceAcquisitionError("sequence_no_frames")
    normalized = [
        frame if isinstance(frame, Frame) else Frame(image=frame) for frame in frames
    ]
    images = [frame.image for frame in normalized]
    width, height = _validate_images(images)
    embedded = [frame.ms_from_start for frame in normalized]
    if all(value is not None for value in embedded):
        timestamps = [float(value) / 1000.0 for value in embedded if value is not None]
        if any(
            timestamps[index] <= timestamps[index - 1]
            for index in range(1, len(timestamps))
        ):
            raise SequenceAcquisitionError("sequence_timestamps_not_monotonic")
        inferred_fps = (
            1.0 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else fps
        )
    else:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise SequenceAcquisitionError("sequence_fps_required")
        inferred_fps = fps
        timestamps = [index / float(fps) for index in range(len(normalized))]
        normalized = [
            Frame(
                image=frame.image,
                timestamp=frame.timestamp,
                ms_from_start=t * 1000.0,
                camera=frame.camera,
                calibration=frame.calibration,
            )
            for frame, t in zip(normalized, timestamps)
        ]
    if inferred_fps is None or inferred_fps <= 0:
        raise SequenceAcquisitionError("sequence_fps_required")
    digest = hashlib.sha256()
    for image in images:
        check_cancelled()
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
