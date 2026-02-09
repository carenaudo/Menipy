"""Edge detection utilities and pipeline stage logic."""

from __future__ import annotations
from typing import Callable, Optional
from abc import ABC, abstractmethod
import numpy as np
import logging

from menipy.models.config import EdgeDetectionSettings
from menipy.models.geometry import Contour

# Keep your registry import
from .registry import EDGE_DETECTORS
from .image_utils import ensure_gray, edges_to_xy

# OpenCV is optional; code degrades gracefully if missing
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


# -------- plugin discovery (unchanged in spirit) --------
def _load_plugin(name: str) -> Optional[Callable]:
    """
    Try to load a detector from Python entry points (group='menipy.edge_detection').
    The callable is expected to accept an image (gray or BGR) and return either:
      - an (N,2) float array of (x,y) contour points in image coords, or
      - a 2D edges mask (uint8/bool), from which we'll extract the largest contour.
    """
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="menipy.edge_detection"):
            if ep.name == name:
                return ep.load()
    except Exception:
        pass
    return None


# -------- helpers --------
# Helpers _ensure_gray and _edges_to_xy are kept for potential backward 
# compatibility but we prefer using the ones from detection_helpers.
_ensure_gray = ensure_gray
_edges_to_xy = edges_to_xy


def _fallback_canny(img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
    """Default detector: try registry 'canny' or fallback to simple threshold."""
    # Try getting 'canny' from registry
    canny_fn = EDGE_DETECTORS.get("canny")
    if canny_fn:
        return canny_fn(img, settings)
    
    # Minimal hardcoded fallback if registry is empty
    g = ensure_gray(img)
    # Simple threshold around median
    v = float(np.median(g))
    edges = np.asarray((g > v).astype(np.uint8) * 255, dtype=np.uint8)
    
    return edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


# -------- strategies --------
# Strategies (CannyDetector, ThresholdDetector, etc.) have been moved to plugins/edge_detectors.py
# and are no longer defined here. We rely on the registry.


def get_contour_detector(
    name: str = "canny",
) -> Callable[[np.ndarray, EdgeDetectionSettings], np.ndarray]:
    """
    Resolution order:
      1) Registry (EDGE_DETECTORS) which includes built-ins
      2) Plugin entry point 'menipy.edge_detection'
      3) Built-in fallback Canny
    """
    if name in EDGE_DETECTORS:
        return EDGE_DETECTORS[name]

    plugin_fn = _load_plugin(name)
    if plugin_fn:
        return plugin_fn

    return lambda img, settings: _fallback_canny(img, settings)


def extract_external_contour(image: np.ndarray) -> np.ndarray:
    """Extracts the largest external contour from a binary image."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for contour extraction.")

    # Ensure image is binary
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=float)

    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour.squeeze(1).astype(float)


# -------- public API --------
def run(ctx, settings: EdgeDetectionSettings):
    """Run a contour detector on the current image and write to ctx.contour.xy."""
    if not settings.enabled:
        logger.info("Edge detection is disabled. Skipping.")
        return

    # 1) pick an image from Context
    img = getattr(ctx, "preprocessed", None)
    if img is None:
        img = getattr(ctx, "gray", None)
    if img is None:
        img = getattr(ctx, "frame", None)
    if img is None:
        frames = getattr(ctx, "frames", None)
        if isinstance(frames, (list, tuple)) and frames:
            img = frames[0]
        elif frames is not None:
            img = frames
    if img is None:
        img = getattr(ctx, "image", None)
    if img is None:
        img = getattr(ctx, "preview", None)

    # Handle Frame object
    if hasattr(img, "image"):
        img = img.image
    state = getattr(ctx, "preprocessed_state", None)
    roi_bounds = getattr(ctx, "roi", None)
    roi_image_override = None
    if state is not None:
        try:
            state_roi = getattr(state, "roi_bounds", None)
            if state_roi:
                roi_bounds = state_roi
            for key in ("normalized_roi", "filtered_roi", "working_roi", "raw_roi"):
                candidate = getattr(state, key, None)
                if candidate is not None:
                    roi_image_override = candidate
                    break
        except Exception:
            roi_image_override = None

    # use module-level logger

    if img is None and roi_image_override is None:
        # Try a safe acquisition fallback
        try:
            from . import acquisition as acq

            ip = getattr(ctx, "image_path", None) or getattr(ctx, "image", None)
            if ip:
                logger.info("EdgeDetection fallback: loading image from path: %s", ip)
                try:
                    frames = acq.from_file([ip])
                except Exception as e:
                    logger.warning("EdgeDetection fallback: from_file failed: %s", e)
                    frames = None
                if frames:
                    ctx.frames = frames
                    img = frames[0]
                    msg = f"EdgeDetection fallback: loaded {len(frames)} frame(s) from {ip}."
                    logger.info(msg)
                    try:
                        setattr(ctx, "status_message", msg)
                    except Exception:
                        pass

            if img is None:
                cam = getattr(ctx, "camera_id", None)
                n = getattr(ctx, "frames_requested", None) or 1
                if cam is not None:
                    logger.info(
                        "EdgeDetection fallback: capturing from camera %s (n=%s)",
                        cam,
                        n,
                    )
                    try:
                        frames = acq.from_camera(device=cam, n_frames=int(n))
                    except Exception as e:
                        logger.warning(
                            "EdgeDetection fallback: from_camera failed: %s", e
                        )
                        frames = None
                    if frames:
                        ctx.frames = frames
                        img = frames[0]
                        msg = f"EdgeDetection fallback: captured {len(frames)} frame(s) from camera {cam}."
                        logger.info(msg)
                        try:
                            setattr(ctx, "status_message", msg)
                        except Exception:
                            pass
        except Exception:
            logger.debug("EdgeDetection fallback: acquisition helper not available")
            img = None

    if img is None and roi_image_override is None:
        raise RuntimeError(
            "EdgeDetection: no image in Context. "
            "Ensure 'acquisition' (and usually 'preprocessing') ran and set ctx.frame/frames."
        )

    # 2) crop ROI if present
    roi = roi_bounds
    x0 = y0 = 0
    img_roi = None

    def _parse_roi(candidate) -> tuple[int, int, int, int] | None:
        if candidate is None:
            return None
        # Mapping-like (dict, SimpleNamespace) with keys
        try:
            if hasattr(candidate, "get"):
                x = candidate.get("x") or candidate.get("left")
                y = candidate.get("y") or candidate.get("top")
                w = candidate.get("w") or candidate.get("width")
                h = candidate.get("h") or candidate.get("height")
                if x is not None and y is not None and w is not None and h is not None:
                    return (int(x), int(y), int(w), int(h))
        except Exception:
            pass
        # Object with attributes
        try:
            x = getattr(candidate, "x", None) or getattr(candidate, "left", None)
            y = getattr(candidate, "y", None) or getattr(candidate, "top", None)
            w = getattr(candidate, "w", None) or getattr(candidate, "width", None)
            h = getattr(candidate, "h", None) or getattr(candidate, "height", None)
            if x is not None and y is not None and w is not None and h is not None:
                return (int(x), int(y), int(w), int(h))
        except Exception:
            pass
        # Sequence-like
        try:
            seq = list(candidate)
            if len(seq) >= 4:
                return (int(seq[0]), int(seq[1]), int(seq[2]), int(seq[3]))
        except Exception:
            pass
        return None

    parsed = _parse_roi(roi)
    if roi_image_override is not None:
        img_roi = roi_image_override
        if parsed:
            x0, y0 = parsed[0], parsed[1]
    elif parsed:
        x0, y0, w, h = parsed
        img_roi = img[y0 : y0 + h, x0 : x0 + w]
    else:
        img_roi = img

    # Ensure image is grayscale for edge detection
    img_roi_gray = _ensure_gray(img_roi)

    # Apply Gaussian blur if enabled
    if settings.gaussian_blur_before and cv2 is not None:
        ksize = (settings.gaussian_kernel_size, settings.gaussian_kernel_size)
        img_roi_gray = cv2.GaussianBlur(img_roi_gray, ksize, settings.gaussian_sigma_x)

    # 3) select detector (Strategy Pattern) and apply
    detector = get_contour_detector(settings.method)
    xy = detector(img_roi_gray, settings)

    if xy.size == 0:
        logger.warning(
            "EdgeDetection: detector returned no contour for method %s.",
            settings.method,
        )
        # If no contour is found, try a very basic thresholding as a last resort
        if cv2 is not None:
            _, edges = cv2.threshold(img_roi_gray, 127, 255, cv2.THRESH_BINARY)
            xy = edges_to_xy(
                edges, settings.min_contour_length, settings.max_contour_length
            )
        if xy.size == 0:
            # If no contour can be found, don't raise — instead set an empty
            # contour so downstream stages can handle this gracefully.
            logger.warning(
                "EdgeDetection: detector returned no contour even after fallback; setting empty contour."
            )
            xy = np.empty((0, 2), float)
            # write into Context and return early so callers get an empty contour
            if getattr(ctx, "contour", None) is None:
                ctx.contour = Contour(xy=xy)
            else:
                ctx.contour.xy = xy
            ctx.fluid_interface_contour = None
            ctx.solid_interface_contour = None
            return ctx

    # 5) offset back to full-image coords (only if we parsed a valid ROI)
    if parsed:
        xy = xy.copy()
        xy[:, 0] += x0
        xy[:, 1] += y0

    # 6) write into Context
    if getattr(ctx, "contour", None) is None:
        ctx.contour = Contour(xy=xy)
    else:
        ctx.contour.xy = xy

    # Handle interface specific detection
    if settings.detect_fluid_interface:
        # The main detected contour is considered the fluid-droplet interface
        ctx.fluid_interface_contour = ctx.contour
    else:
        ctx.fluid_interface_contour = None

    if (
        settings.detect_solid_interface
        and ctx.contact_line is not None
        and cv2 is not None
    ):
        # Search for solid interface near the contact line
        p1, p2 = ctx.contact_line
        mask = np.zeros_like(img_roi_gray)
        min_x = min(p1[0], p2[0]) - settings.solid_interface_proximity
        max_x = max(p1[0], p2[0]) + settings.solid_interface_proximity
        min_y = min(p1[1], p2[1]) - settings.solid_interface_proximity
        max_y = max(p1[1], p2[1]) + settings.solid_interface_proximity

        h, w = img_roi_gray.shape
        min_x = max(0, min_x)
        max_x = min(w, max_x)
        min_y = max(0, min_y)
        max_y = min(h, max_y)

        if min_x < max_x and min_y < max_y:
            mask[min_y:max_y, min_x:max_x] = 255

            # Re-generate edges mask for this secondary detection
            # Note: ideally we would get the edges image from the strategy,
            # but strategies return xy. For now, we reuse the Canny fallback or similar logic just for the mask
            # This is a limitation of the current interface that returns xy directly.
            # To fix this properly would require changing strategy to return edges image or having separate method.
            # For now, we perform a quick local Canny for solid interface
            edges_for_solid = cv2.Canny(
                img_roi_gray, settings.canny_threshold1, settings.canny_threshold2
            )

            masked_edges = cv2.bitwise_and(edges_for_solid, edges_for_solid, mask=mask)
            solid_interface_xy = edges_to_xy(
                masked_edges, settings.min_contour_length, settings.max_contour_length
            )

            if solid_interface_xy.size > 0:
                if roi:
                    solid_interface_xy[:, 0] += x0
                    solid_interface_xy[:, 1] += y0
                ctx.solid_interface_contour = Contour(xy=solid_interface_xy)
            else:
                ctx.solid_interface_contour = None
        else:
            ctx.solid_interface_contour = None
    else:
        ctx.solid_interface_contour = None

    return ctx
