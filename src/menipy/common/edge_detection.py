from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import logging

# Keep your registry import
from .registry import EDGE_DETECTORS

# OpenCV is optional; code degrades gracefully if missing
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore


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
def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3 and cv2 is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3:
        # simple luminance fallback if OpenCV isn't present
        return (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(np.uint8)
    return img


def _edges_to_xy(edges: np.ndarray) -> np.ndarray:
    """Convert an edges mask to an (N,2) contour (largest external contour)."""
    if edges is None:
        return np.empty((0, 2), float)
    if cv2 is None:
        ys, xs = np.nonzero(edges)
        if xs.size == 0:
            return np.empty((0, 2), float)
        return np.column_stack([xs, ys]).astype(float)
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not cnts:
        return np.empty((0, 2), float)
    c = max(cnts, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(float)


def _fallback_canny(img: np.ndarray) -> np.ndarray:
    """Default detector: auto-thresholded Canny → largest contour (Nx2)."""
    g = _ensure_gray(img)
    if cv2 is None:
        # No OpenCV: threshold around median as a crude edge map
        v = float(np.median(g))
        edges = (g > v).astype(np.uint8) * 255
    else:
        v = float(np.median(g))
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(g, lower, upper)
    return _edges_to_xy(edges)


def get_contour_detector(name: str = "canny") -> Callable[[np.ndarray], np.ndarray]:
    """
    Resolution order:
      1) Built-in registry (EDGE_DETECTORS)
      2) Plugin entry point 'menipy.edge_detection'
      3) Built-in fallback Canny
    """
    return EDGE_DETECTORS.get(name) or _load_plugin(name) or _fallback_canny


# -------- public API --------
def run(ctx, method: str = "canny"):
    """
    Run a contour detector on the current image and write to ctx.contour.xy.

    Image sources checked (first hit wins):
      ctx.preprocessed → ctx.gray → ctx.frame → ctx.frames[0]/ctx.frames → ctx.image → ctx.preview

    If ctx.roi = (x,y,w,h) is present, detection runs on the crop and the contour
    is re-offset back to full image coordinates.
    """
    # 1) pick an image from Context
    img = getattr(ctx, "preprocessed", None)
    if img is None: img = getattr(ctx, "gray", None)
    if img is None: img = getattr(ctx, "frame", None)
    if img is None:
        frames = getattr(ctx, "frames", None)
        if isinstance(frames, (list, tuple)) and frames:
            img = frames[0]
        elif frames is not None:
            img = frames
    if img is None: img = getattr(ctx, "image", None)
    if img is None: img = getattr(ctx, "preview", None)

    # prepare a logger for visible runtime info
    logger = logging.getLogger(__name__)

    if img is None:
        # Try a safe acquisition fallback if the pipeline forgot to populate frames.
        # Use any hints available on ctx (image_path, camera_id, frames_requested).
        try:
            from . import acquisition as acq  # local helper: from_file / from_camera
            # Prefer explicit image path
            ip = getattr(ctx, "image_path", None) or getattr(ctx, "image", None)
            if ip:
                logger.info("EdgeDetection fallback: attempting to load image from path: %s", ip)
                # record a status message on ctx for UI display if desired
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
            # Otherwise try camera hints
            if img is None:
                cam = getattr(ctx, "camera_id", None)
                n = getattr(ctx, "frames_requested", None) or 1
                if cam is not None:
                    logger.info("EdgeDetection fallback: attempting to capture from camera %s (n=%s)", cam, n)
                    try:
                        frames = acq.from_camera(device=cam, n_frames=int(n))
                    except Exception as e:
                        logger.warning("EdgeDetection fallback: from_camera failed: %s", e)
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
            # If acquisition helper isn't available or fails, fall through to raise below.
            logger.debug("EdgeDetection fallback: acquisition helper not available")
            img = None

    if img is None:
        raise RuntimeError(
            "EdgeDetection: no image in Context. "
            "Ensure 'acquisition' (and usually 'preprocessing') ran and set ctx.frame/frames."
        )

    # 2) crop ROI if present
    roi = getattr(ctx, "roi", None)
    x0 = y0 = 0
    if roi:
        x0, y0, w, h = [int(v) for v in roi]
        img_roi = img[y0:y0 + h, x0:x0 + w]
    else:
        img_roi = img

    # 3) select detector (registry → plugin → fallback)
    detector = get_contour_detector(method)
    logger.info("EdgeDetection: using detector '%s'", method)
    try:
        setattr(ctx, "status_message", f"EdgeDetection: running detector '{method}'")
    except Exception:
        pass

    # 4) call detector; accept Nx2 or edges mask
    try:
        out = detector(img_roi)
    except Exception:
        # Some plugins expect grayscale
        out = detector(_ensure_gray(img_roi))

    xy: np.ndarray
    if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] == 2:
        xy = out.astype(float, copy=False)
    else:
        # Treat as edges mask
        xy = _edges_to_xy(out)

    if xy.size == 0:
        raise RuntimeError("EdgeDetection: detector returned no contour.")

    # 5) offset back to full-image coords
    if roi:
        xy = xy.copy()
        xy[:, 0] += x0
        xy[:, 1] += y0

    # 6) write into Context
    if getattr(ctx, "contour", None) is None:
        from types import SimpleNamespace
        ctx.contour = SimpleNamespace()
    ctx.contour.xy = xy
    return ctx
