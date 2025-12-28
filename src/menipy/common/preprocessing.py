"""
Preprocessing pipeline stage implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from menipy.models.context import Context
from menipy.models.config import PreprocessingSettings
from menipy.models.state import PreprocessingState, MarkerSet

from .preprocessing_helpers import (  # Assuming this file exists or will be created
    PreprocessingContext,
    PreprocessingError,
    convert_to_grayscale,
    crop_to_roi,
    rescale_roi,
    apply_filter,
    subtract_background,
    normalize_intensity,
    detect_contact_line,
    fill_holes,
)

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


def run(
    ctx: Context, settings: PreprocessingSettings | dict[str, Any] | None = None
) -> Context:
    """
    Executes the full preprocessing pipeline on the data within the context.
    Args:
        ctx: The shared data context containing the source image and geometry.
        settings: The preprocessing settings to apply.
    """
    frame = _resolve_source_image(ctx)
    if frame is None:
        ctx.error = "No frame available for preprocessing. Please ensure an image is loaded or captured first."
        ctx.note("Preprocessing failed: no source image available")
        raise PreprocessingError(
            "Preprocessing: no frame available. Run acquisition first."
        )

    resolved_settings = _resolve_settings(ctx, settings)
    markers = _resolve_markers(ctx)
    roi_bounds = (
        ctx.roi
        if (resolved_settings.crop_to_roi and getattr(ctx, "roi", None))
        else None
    )
    contact_segment = getattr(ctx, "contact_line", None)
    roi_mask_full = getattr(ctx, "roi_mask", None)

    pre_ctx = PreprocessingContext(
        source_image=frame,
        settings=resolved_settings,
        roi_bounds=roi_bounds,
        roi_mask_full=roi_mask_full,
        contact_line_segment=contact_segment,  # type: ignore
        markers=markers,
    )

    # --- Execute pipeline stages using helpers ---
    crop_to_roi(pre_ctx)
    # Optional: clean mask holes and spurious points close to contact line
    fill_holes(pre_ctx)
    convert_to_grayscale(pre_ctx)
    rescale_roi(pre_ctx)
    apply_filter(pre_ctx)
    subtract_background(pre_ctx)
    normalize_intensity(pre_ctx)
    detect_contact_line(pre_ctx)

    processed_roi = pre_ctx.current_image
    state = pre_ctx.state

    full_image = _compose_full_image(frame, processed_roi, state)

    # --- Finalize context with results ---
    ctx.preprocessed_state = state
    ctx.preprocessed_settings = resolved_settings
    ctx.preprocessed_history = [record.model_dump() for record in state.history]
    ctx.preprocessed_roi = processed_roi
    ctx.preprocessed_mask = state.roi_mask
    ctx.preprocessed_scale = state.scale
    ctx.contact_line_mask = state.contact_line_mask
    ctx.preprocessed = full_image if full_image is not None else processed_roi

    # Keep original frame intact for preview; stash processed ROI separately.
    ctx.note("Preprocessing complete") if hasattr(ctx, "note") else None
    return ctx


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _resolve_source_image(ctx: Context) -> Optional[np.ndarray]:
    candidates = [
        getattr(ctx, "current_frame", None),
        None,
    ]
    frames = getattr(ctx, "frames", None)
    if candidates[0] is None and isinstance(frames, (list, tuple)) and frames:
        candidates[0] = frames[0]
    elif candidates[0] is None and isinstance(frames, np.ndarray):
        candidates[0] = frames
    if candidates[0] is None:
        image = getattr(ctx, "image", None)
        candidates[0] = image
    if candidates[0] is None:
        preview = getattr(ctx, "preview", None)
        candidates[0] = preview
    img = candidates[0]
    if img is None:
        return None
    if isinstance(img, list) and img:
        img = img[0]
    if not isinstance(img, np.ndarray):
        return None
    if hasattr(img, "image") and isinstance(
        getattr(img, "image"), np.ndarray
    ):  # Handle Frame object
        img = img.image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resolve_settings(
    ctx: Context, override: PreprocessingSettings | dict[str, Any] | None
) -> PreprocessingSettings:
    if isinstance(override, PreprocessingSettings):
        return override
    if isinstance(override, dict):
        return PreprocessingSettings(**override)
    existing = getattr(ctx, "preprocessed_settings", None) or getattr(
        ctx, "preprocessing_settings", None
    )
    if isinstance(existing, PreprocessingSettings):
        return existing
    if isinstance(existing, dict):
        return PreprocessingSettings(**existing)
    return PreprocessingSettings()


def _resolve_markers(ctx: Context) -> Optional[MarkerSet]:
    markers = getattr(ctx, "preprocessing_markers", None)
    if isinstance(markers, MarkerSet):
        return markers
    if isinstance(markers, dict):
        try:
            return MarkerSet(**markers)
        except Exception:
            logger.debug("Preprocessing: could not coerce markers dict", exc_info=True)
    return None


def _compose_full_image(
    base: np.ndarray, roi: np.ndarray, state: PreprocessingState
) -> Optional[np.ndarray]:
    if roi is None or state.roi_bounds is None:
        return None
    x, y, w, h = state.roi_bounds
    if roi.shape[0] != h or roi.shape[1] != w:
        # Resized ROI cannot be reinserted in-place; expose in metadata for downstream consumers.
        state.metadata.setdefault("roi_resized", True)
        state.metadata["roi_resized_shape"] = (roi.shape[1], roi.shape[0])
        return None
    composite = base.copy()

    # Ensure ROI and composite have compatible channel shapes before assignment.
    roi_to_insert = roi

    # If composite is color (H,W,3) but roi is grayscale (H,W), convert roi -> (H,W,3)
    if composite.ndim == 3 and roi.ndim == 2:
        if cv2 is not None:
            roi_to_insert = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            roi_to_insert = np.dstack([roi] * composite.shape[2])

    # If composite is grayscale but roi is color, convert composite to color to preserve roi channels.
    elif composite.ndim == 2 and roi.ndim == 3:
        if cv2 is not None:
            composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
        else:
            composite = np.dstack([composite] * roi.shape[2])
        # roi already color; leave roi_to_insert as-is

    # If number of channels still mismatches (e.g., roi has alpha), adapt by trimming or padding.
    if (
        roi_to_insert.ndim == 3
        and composite.ndim == 3
        and roi_to_insert.shape[2] != composite.shape[2]
    ):
        c_comp = composite.shape[2]
        c_roi = roi_to_insert.shape[2]
        if c_roi > c_comp:
            # drop extra channels (e.g., alpha)
            roi_to_insert = roi_to_insert[..., :c_comp]
        else:
            # pad by repeating last channel
            pad = np.repeat(roi_to_insert[..., -1:], c_comp - c_roi, axis=2)
            roi_to_insert = np.concatenate([roi_to_insert, pad], axis=2)

    try:
        composite[y : y + h, x : x + w] = roi_to_insert
    except ValueError as e:
        # Log and surface the error to the preprocessing state for easier debugging.
        logger.exception("Failed to insert ROI into composite image: %s", e)
        state.metadata.setdefault("compose_error", str(e))
        return None

    return composite
