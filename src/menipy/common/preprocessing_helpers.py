from __future__ import annotations

"""Helper utilities for Menipy preprocessing pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np

from menipy.models.config import PreprocessingSettings
from menipy.models.state import PreprocessingState, PreprocessingStageRecord, MarkerSet

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - test environments may lack OpenCV
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from skimage import exposure, morphology
except Exception:  # pragma: no cover - avoid hard dependency
    exposure = None
    morphology = None


class PreprocessingError(RuntimeError):
    """Raised when the preprocessing pipeline cannot proceed."""


@dataclass
class PreprocessingContext:
    """Holds immutable inputs and mutable state for preprocessing helpers."""

    source_image: np.ndarray
    settings: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    roi_bounds: Optional[Tuple[int, int, int, int]] = None
    roi_mask_full: Optional[np.ndarray] = None
    contact_line_segment: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    markers: Optional[MarkerSet] = None

    state: PreprocessingState = field(default_factory=PreprocessingState)

    def __post_init__(self) -> None:
        height, width = self.source_image.shape[:2]
        if self.roi_bounds is None:
            self.roi_bounds = (0, 0, width, height)
        else:
            self.roi_bounds = _clamp_roi(self.roi_bounds, width, height)
        self.state.roi_bounds = self.roi_bounds
        if self.markers is not None:
            self.state.markers = self.markers
        if self.contact_line_segment is not None:
            self.state.contact_line_presence = True
            self.state.metadata["contact_line_segment"] = self.contact_line_segment

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self.state.roi_mask

    @property
    def active_mask(self) -> Optional[np.ndarray]:
        return self.state.roi_mask if self.settings.work_on_roi_mask else None

    @property
    def current_image(self) -> np.ndarray:
        for name in ("normalized_roi", "filtered_roi", "working_roi", "raw_roi"):
            img = getattr(self.state, name)
            if img is not None:
                return img
        raise PreprocessingError("PreprocessingContext has no image buffers populated")

    def push_history(self, stage: str, params: Dict[str, Any] | None = None) -> None:
        record = PreprocessingStageRecord(name=stage, params=params or {})
        self.state.history.append(record)

    def update_working(self, array: np.ndarray) -> None:
        self.state.working_roi = array

    def set_filtered(self, array: np.ndarray) -> None:
        self.state.filtered_roi = array

    def set_normalized(self, array: np.ndarray) -> None:
        self.state.normalized_roi = array


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def convert_to_grayscale(context: PreprocessingContext) -> None:
    """Convert the source image to grayscale if configured."""
    if not context.settings.convert_to_grayscale:
        return

    current = context.current_image
    if current is None:
        return

    if current.ndim == 3:
        if cv2 is None:
            logger.warning("OpenCV not found, cannot convert to grayscale.")
            return
        logger.debug("Converting image to grayscale.")
        grayscaled = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        context.update_working(grayscaled)
        context.push_history("grayscale")

def crop_to_roi(context: PreprocessingContext) -> None:
    """Extract ROI from the source image and derive a binary mask."""

    x, y, w, h = context.roi_bounds  # type: ignore[misc]
    x, y, w, h = int(x), int(y), int(w), int(h)
    roi = context.source_image[y : y + h, x : x + w]
    if roi.size == 0:
        raise PreprocessingError("ROI crop produced an empty array")
    context.state.raw_roi = roi.copy()
    context.update_working(roi.copy())

    if context.roi_mask_full is not None:
        mask = context.roi_mask_full[y : y + h, x : x + w]
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255
    context.state.roi_mask = mask

    if context.contact_line_segment is not None:
        _embed_contact_line_mask(context, mask)
    context.push_history("crop", {"roi": (x, y, w, h)})


def rescale_roi(context: PreprocessingContext) -> None:
    """Resize ROI according to settings, preserving mask and scale factor."""

    settings = context.settings.resize
    if not settings.enabled or not settings.has_target:
        return

    current = context.state.working_roi
    if current is None:
        return
    target_w, target_h = settings.target_width, settings.target_height
    src_h, src_w = current.shape[:2]

    if settings.preserve_aspect:
        scale_w = target_w / src_w if target_w else float('inf')
        scale_h = target_h / src_h if target_h else float('inf')
        
        # Use the more constraining scale factor if both are provided
        scale = min(scale_w, scale_h)
        
        # If neither was provided, scale is inf, so we do nothing.
        if scale is None:
            return

        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
    else:
        new_w = target_w or src_w
        new_h = target_h or src_h

    interpolation = _cv2_interpolation(settings.interpolation)
    resized = _resize_array(current, (new_w, new_h), interpolation)
    if resized is None:
        logger.warning("Preprocessing: resize skipped due to missing backend")
        return

    context.update_working(resized)
    mask = context.state.roi_mask
    if mask is not None:
        resized_mask = _resize_mask(mask, (new_w, new_h))
        context.state.roi_mask = resized_mask
    sx = new_w / src_w
    sy = new_h / src_h
    context.state.scale = (sx, sy)
    context.push_history("resize", {"width": new_w, "height": new_h, "interpolation": settings.interpolation})


def apply_filter(context: PreprocessingContext) -> None:
    """Apply smoothing filters within the ROI mask."""

    settings = context.settings.filtering
    if not settings.enabled or settings.method == "none":
        return

    current = context.state.working_roi
    if current is None:
        return

    method = settings.method
    if method == "gaussian":
        filtered = _gaussian_blur(current, settings.kernel_size, settings.sigma)
    elif method == "median":
        filtered = _median_blur(current, settings.kernel_size)
    elif method == "bilateral":
        filtered = _bilateral_filter(current, settings.kernel_size, settings.sigma_color, settings.sigma_space)
    else:
        filtered = None

    if filtered is None:
        logger.warning("Preprocessing: filter '%s' unavailable; skipping", method)
        return

    filtered = _apply_mask(current, filtered, context.active_mask)
    context.update_working(filtered)
    context.set_filtered(filtered.copy())
    context.push_history("filter", settings.model_dump())


def subtract_background(context: PreprocessingContext) -> None:
    """Perform background subtraction limited to the ROI mask."""

    settings = context.settings.background
    if not settings.enabled:
        return

    current = context.state.working_roi
    if current is None:
        return

    if settings.mode == "flat":
        background = _estimate_flat_background(current, context.active_mask, settings.strength)
    else:
        background = _rolling_background(current, context.active_mask, settings.rolling_radius)

    if background is None:
        logger.warning("Preprocessing: background mode '%s' unavailable; skipping", settings.mode)
        return

    if cv2:
        # Use OpenCV's subtraction which handles underflow by clipping to 0
        adjusted = cv2.subtract(current, background)
    else:
        # Fallback to numpy if OpenCV is not available
        adjusted = np.clip(current.astype(np.int16) - background.astype(np.int16), 0, 255).astype(np.uint8)

    adjusted = _apply_mask(current, adjusted, context.active_mask)
    context.update_working(adjusted)
    context.push_history("background", settings.model_dump())


def normalize_intensity(context: PreprocessingContext) -> None:
    """Normalize intensities (contrast stretch / CLAHE)."""

    settings = context.settings.normalization
    if not settings.enabled:
        return

    current = context.state.working_roi
    if current is None:
        return

    if settings.method == "clahe":
        normalized = _apply_clahe(current, settings.clip_limit, settings.grid_size)
    elif settings.method == "otsu":
        normalized = _apply_otsu_threshold(current)
    else:
        normalized = _histogram_stretch(current)

    if normalized is None:
        logger.warning("Preprocessing: normalization '%s' unavailable; skipping", settings.method)
        return

    normalized = _apply_mask(current, normalized, context.active_mask)
    context.update_working(normalized)
    context.set_normalized(normalized.copy())
    context.push_history("normalize", {"method": settings.method})


def detect_contact_line(context: PreprocessingContext) -> None:
    """Update contact line presence flag based on context markers."""

    if context.state.contact_line_presence:
        return
    markers = context.state.markers
    if markers and markers.contact_line_anchors:
        context.state.contact_line_presence = True
        context.state.metadata["contact_line_anchors"] = markers.contact_line_anchors
    else:
        context.state.contact_line_presence = False


def fill_holes(context: PreprocessingContext) -> None:
    """Fill small interior holes in the ROI mask and remove small spurious objects.

    Uses skimage.morphology.remove_small_holes/remove_small_objects when available; falls
    back to OpenCV-based contour filling and connected-component filtering if needed.
    Updates `context.state.roi_mask` and appends a history record.
    """
    settings = getattr(context.settings, "fill_holes", None)
    if settings is None or not getattr(settings, "enabled", False):
        return

    mask = context.state.roi_mask
    if mask is None:
        logger.debug("fill_holes: no ROI mask available; skipping")
        return

    bin_mask = (mask > 0)
    max_area = int(getattr(settings, "max_hole_area", 500) or 0)

    out_mask = None
    # Prefer skimage morphology utilities
    if morphology is not None:
        try:
            logger.debug("fill_holes: running skimage morphology-based cleanup")
            # remove_small_holes expects boolean array
            if max_area > 0:
                filled = morphology.remove_small_holes(bin_mask, area_threshold=max_area)
            else:
                filled = bin_mask

            # Optionally remove small objects (spurious) near contact line
            if getattr(settings, "remove_spurious_near_contact", False) and context.state.contact_line_mask is not None:
                contact = (context.state.contact_line_mask > 0)
                if contact.any():
                    # create a proximity mask around contact line
                    try:
                        from scipy import ndimage as _nd

                        dist = _nd.distance_transform_edt(~contact)
                        proximity = dist <= int(getattr(settings, "proximity_px", 5) or 0)
                        # remove small objects that lie within proximity
                        cleaned = morphology.remove_small_objects(filled & ~proximity, min_size=1)
                        # Keep objects inside proximity as-is; merge
                        final = cleaned | (filled & proximity)
                    except Exception:
                        final = filled
                else:
                    final = filled
            else:
                # remove very small objects globally (no size threshold besides default)
                final = morphology.remove_small_objects(filled, min_size=1)

            out_mask = (final.astype(np.uint8) * 255)
        except Exception:
            logger.debug("fill_holes: skimage path failed, falling back to OpenCV", exc_info=True)

    # Fallback to OpenCV based processing
    if out_mask is None:
        out_mask = mask.copy()
        if cv2 is not None:
            logger.debug("fill_holes: using OpenCV fallback for filling/removal")
            inv = (~bin_mask).astype('uint8') * 255
            contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 0 < max_area and area <= max_area:
                    cv2.drawContours(out_mask, [cnt], -1, color=255, thickness=-1)

            if getattr(settings, "remove_spurious_near_contact", False) and context.state.contact_line_mask is not None:
                contact = (context.state.contact_line_mask > 0).astype('uint8')
                if contact.any():
                    k = max(1, int(getattr(settings, "proximity_px", 5) or 0))
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                    prox = cv2.dilate(contact, kernel, iterations=1).astype(bool)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((out_mask > 0).astype('uint8'), connectivity=8)
                    for lab in range(1, num_labels):
                        area = int(stats[lab, cv2.CC_STAT_AREA])
                        if 0 < max_area and area <= max_area:
                            comp = (labels == lab)
                            overlap = np.logical_and(comp, prox).sum()
                            if overlap > 0:
                                out_mask[comp] = 0
        else:
            logger.debug("fill_holes: OpenCV not available; cannot run fallback operations")

    # Persist mask and history
    context.state.roi_mask = (out_mask > 0).astype(np.uint8) * 255
    try:
        params = settings.model_dump() if hasattr(settings, "model_dump") else {"max_hole_area": max_area}
    except Exception:
        params = {"max_hole_area": max_area}
    context.push_history("fill_holes", params)


# ---------------------------------------------------------------------------
# Low-level utilities (mostly pure functions)
# ---------------------------------------------------------------------------

def _clamp_roi(roi: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return int(x), int(y), int(w), int(h)


def _embed_contact_line_mask(context: PreprocessingContext, mask: np.ndarray) -> None:
    segment = context.contact_line_segment
    if segment is None or mask is None:
        return
    (x1, y1), (x2, y2) = segment
    x0, y0, _, _ = context.roi_bounds or (0, 0, mask.shape[1], mask.shape[0])
    padding = max(1, context.settings.contact_line.dilation)
    xs = sorted(((x1 - x0), (x2 - x0)))
    ys = sorted(((y1 - y0), (y2 - y0)))
    x_start = max(0, min(mask.shape[1], xs[0] - padding))
    x_end = max(0, min(mask.shape[1], xs[1] + padding))
    y_start = max(0, min(mask.shape[0], ys[0] - padding))
    y_end = max(0, min(mask.shape[0], ys[1] + padding))
    if x_start >= x_end or y_start >= y_end:
        return
    contact_mask = np.zeros_like(mask)
    contact_mask[y_start:y_end, x_start:x_end] = 255
    context.state.contact_line_mask = contact_mask


def _cv2_interpolation(name: str) -> int:
    mapping = {
        "nearest": 0,
        "linear": 1,
        "cubic": 2,
        "area": 3,
        "lanczos": 4,
    }
    return mapping.get(name, 1)


def _resize_array(array: np.ndarray, shape: Tuple[int, int], interpolation: int) -> Optional[np.ndarray]:
    new_w, new_h = shape
    if array.shape[1] == new_w and array.shape[0] == new_h:
        return array.copy()
    if cv2 is not None:
        return cv2.resize(array, (new_w, new_h), interpolation=interpolation)
    try:
        from skimage.transform import resize

        result = resize(array, (new_h, new_w, *array.shape[2:]), order=1 if interpolation != 0 else 0, preserve_range=True, anti_aliasing=False)
        return result.astype(array.dtype)
    except Exception:
        if new_w < array.shape[1] or new_h < array.shape[0]:
            return None
        if new_w <= 0 or new_h <= 0:
            return None
        sy = max(1, int(round(new_h / array.shape[0])))
        sx = max(1, int(round(new_w / array.shape[1])))
        resized = np.repeat(np.repeat(array, sy, axis=0), sx, axis=1)
        return resized[:new_h, :new_w]


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    new_w, new_h = shape
    if mask.shape[1] == new_w and mask.shape[0] == new_h:
        return mask.copy()
    if cv2 is not None:
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized = _resize_array(mask, shape, interpolation=0)
    return resized if resized is not None else mask.copy()


def _apply_mask(original: np.ndarray, candidate: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return candidate
    mask_bool = mask.astype(bool)
    out = original.copy()
    if candidate.ndim == 3 and mask_bool.ndim == 2:
        mask_bool = np.stack([mask_bool] * candidate.shape[2], axis=-1)
    out[mask_bool] = candidate[mask_bool]
    return out


def _gaussian_blur(array: np.ndarray, kernel: int, sigma: float) -> Optional[np.ndarray]:
    if kernel % 2 == 0:
        kernel += 1
    if cv2 is not None:
        sigma = max(float(sigma), 0.0)
        return cv2.GaussianBlur(array, (kernel, kernel), sigma)
    try:
        from scipy.ndimage import gaussian_filter

        sigma_tuple = (sigma, sigma, 0) if array.ndim == 3 else sigma
        return gaussian_filter(array, sigma=sigma_tuple).astype(array.dtype)
    except Exception:
        return None


def _median_blur(array: np.ndarray, kernel: int) -> Optional[np.ndarray]:
    if kernel % 2 == 0:
        kernel += 1
    if cv2 is not None:
        if array.ndim == 3:
            channels = [cv2.medianBlur(array[..., i], kernel) for i in range(array.shape[2])]
            return np.stack(channels, axis=2)
        else:
            return cv2.medianBlur(array, kernel)
    try:
        from scipy.ndimage import median_filter

        size = (kernel, kernel, 1) if array.ndim == 3 else (kernel, kernel)
        return median_filter(array, size=size).astype(array.dtype)
    except Exception:
        return None


def _bilateral_filter(array: np.ndarray, kernel: int, sigma_color: float, sigma_space: float) -> Optional[np.ndarray]:
    if cv2 is not None:
        return cv2.bilateralFilter(array, kernel, sigma_color, sigma_space)
    return None


def _estimate_flat_background(array: np.ndarray, mask: Optional[np.ndarray], strength: float) -> np.ndarray:
    if mask is not None and mask.any():
        mask_bool = mask.astype(bool)
        if array.ndim == 3:
            # mask_bool indexes the first two axes; use it to select pixels and keep channels
            # resulting shape will be (N, channels)
            values = array[mask_bool]
            if values.size == 0:
                mean = np.zeros((array.shape[2],), dtype=float)
            else:
                # values.reshape(-1, array.shape[2]) is equivalent if numpy returns flattened channel-last
                mean = values.reshape(-1, array.shape[2]).mean(axis=0)
        else:
            mean = array[mask_bool].mean()
    else:
        mean = array.mean(axis=(0, 1)) if array.ndim == 3 else array.mean()
    return np.full_like(array, (mean * float(strength)).astype(array.dtype))


def _rolling_background(array: np.ndarray, mask: Optional[np.ndarray], radius: int) -> Optional[np.ndarray]:
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        if array.ndim == 3:
            # For color images, apply to the L channel in LAB space to preserve color
            lab = cv2.cvtColor(array, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            l_channel_opened = cv2.morphologyEx(l_channel, cv2.MORPH_OPEN, kernel)
            lab[:, :, 0] = l_channel_opened
            background_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return background_bgr
        else:
            return cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel)

    if morphology is not None:
        try:
            if array.ndim == 3:
                # skimage handles color images with the `channel_axis` argument
                result = morphology.opening(array, morphology.disk(radius), channel_axis=-1)
            else:
                result = morphology.opening(array, morphology.disk(radius))
            return result.astype(array.dtype)
        except Exception:
            return None
    return None


def _apply_clahe(array: np.ndarray, clip_limit: float, grid_size: int) -> Optional[np.ndarray]:
    if array.ndim == 2:
        return _clahe_single(array, clip_limit, grid_size)
    channels = []
    for idx in range(array.shape[2]):
        channel = _clahe_single(array[..., idx], clip_limit, grid_size)
        if channel is None:
            return None
        channels.append(channel)
    return np.stack(channels, axis=2)


def _clahe_single(channel: np.ndarray, clip_limit: float, grid_size: int) -> Optional[np.ndarray]:
    if cv2 is not None:
        clahe = cv2.createCLAHE(clipLimit=float(max(clip_limit, 0.0)), tileGridSize=(grid_size, grid_size))
        return clahe.apply(channel)
    if exposure is not None:
        result = exposure.equalize_adapthist(channel, clip_limit=max(clip_limit, 0.001), kernel_size=grid_size)
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    return None


def _apply_otsu_threshold(array: np.ndarray) -> Optional[np.ndarray]:
    """Apply Otsu's binarization."""
    if cv2 is None:
        return None

    if array.ndim == 3:
        logger.warning("Otsu thresholding requires a grayscale image. Converting.")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def _histogram_stretch(array: np.ndarray) -> Optional[np.ndarray]:
    arr = array.astype(np.float32)
    max_val = float(arr.max())
    min_val = float(arr.min())
    if max_val - min_val < 1e-5:
        return array.copy()
    scaled = (arr - min_val) / (max_val - min_val) * 255.0
    return scaled.astype(np.uint8)
