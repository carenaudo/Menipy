from __future__ import annotations

"""Menipy preprocessing stage orchestration."""

import logging
from typing import Any, Optional

import numpy as np

from menipy.models.datatypes import (
    Context,
    PreprocessingSettings,
    PreprocessingState,
    MarkerSet,
)

from .preprocessing_helpers import (
    PreprocessingContext,
    PreprocessingError,
    crop_to_roi,
    rescale_roi,
    apply_filter,
    subtract_background,
    normalize_intensity,
    detect_contact_line,
)

logger = logging.getLogger(__name__)


def run(ctx: Context, settings: PreprocessingSettings | dict[str, Any] | None = None) -> Context:
    """Execute the preprocessing pipeline on ``ctx`` and populate outputs."""

    frame = _resolve_source_image(ctx)
    if frame is None:
        raise PreprocessingError("Preprocessing: no frame available. Run acquisition first.")

    resolved_settings = _resolve_settings(ctx, settings)
    markers = _resolve_markers(ctx)
    roi_bounds = ctx.roi if (resolved_settings.crop_to_roi and getattr(ctx, "roi", None)) else None
    contact_segment = getattr(ctx, "contact_line", None)
    roi_mask_full = getattr(ctx, "roi_mask", None)

    pre_ctx = PreprocessingContext(
        source_image=frame,
        settings=resolved_settings,
        roi_bounds=roi_bounds,
        roi_mask_full=roi_mask_full,
        contact_line_segment=contact_segment,
        markers=markers,
    )

    crop_to_roi(pre_ctx)
    rescale_roi(pre_ctx)
    apply_filter(pre_ctx)
    subtract_background(pre_ctx)
    normalize_intensity(pre_ctx)
    detect_contact_line(pre_ctx)

    processed_roi = pre_ctx.current_image
    state = pre_ctx.state

    full_image = _compose_full_image(frame, processed_roi, state)

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
        getattr(ctx, "frame", None),
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
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resolve_settings(ctx: Context, override: PreprocessingSettings | dict[str, Any] | None) -> PreprocessingSettings:
    if isinstance(override, PreprocessingSettings):
        return override
    if isinstance(override, dict):
        return PreprocessingSettings(**override)
    existing = getattr(ctx, "preprocessed_settings", None) or getattr(ctx, "preprocessing_settings", None)
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


def _compose_full_image(base: np.ndarray, roi: np.ndarray, state: PreprocessingState) -> Optional[np.ndarray]:
    if roi is None or state.roi_bounds is None:
        return None
    x, y, w, h = state.roi_bounds
    if roi.shape[0] != h or roi.shape[1] != w:
        # Resized ROI cannot be reinserted in-place; expose in metadata for downstream consumers.
        state.metadata.setdefault("roi_resized", True)
        state.metadata["roi_resized_shape"] = (roi.shape[1], roi.shape[0])
        return None
    composite = base.copy()
    composite[y : y + h, x : x + w] = roi
    return composite

