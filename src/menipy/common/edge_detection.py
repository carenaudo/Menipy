"""
Edge detection utilities and pipeline stage logic.
"""

from __future__ import annotations
from typing import Callable, Optional
from abc import ABC, abstractmethod
import numpy as np
import logging

from menipy.models.config import EdgeDetectionSettings
from menipy.models.geometry import Contour

# Keep your registry import
from .registry import EDGE_DETECTORS

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
def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3 and cv2 is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3:
        # simple luminance fallback if OpenCV isn't present
        return (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(
            np.uint8
        )
    return img


def _edges_to_xy(
    edges: np.ndarray, min_len: int = 0, max_len: int = 100000
) -> np.ndarray:
    """Convert an edges mask to an (N,2) contour (largest external contour)."""
    if edges is None:
        return np.empty((0, 2), float)
    if cv2 is None:
        ys, xs = np.nonzero(edges)
        if xs.size == 0:
            return np.empty((0, 2), float)
        xy = np.column_stack([xs, ys]).astype(float)
    else:
        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if not cnts:
            return np.empty((0, 2), float)
        # Filter by length
        valid_cnts = [c for c in cnts if min_len <= len(c) <= max_len]
        if not valid_cnts:
            return np.empty((0, 2), float)
        c = max(valid_cnts, key=cv2.contourArea)
        xy = c.reshape(-1, 2).astype(float)
    return xy


def _fallback_canny(img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
    """Default detector: auto-thresholded Canny → largest contour (Nx2)."""
    g = _ensure_gray(img)
    if cv2 is None:
        # No OpenCV: threshold around median as a crude edge map
        v = float(np.median(g))
        edges = (g > v).astype(np.uint8) * 255
    else:
        lower = settings.canny_threshold1
        upper = settings.canny_threshold2
        edges = cv2.Canny(
            g,
            lower,
            upper,
            apertureSize=settings.canny_aperture_size,
            L2gradient=settings.canny_L2_gradient,
        )
    return _edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


# -------- strategies --------


class EdgeDetectorStrategy(ABC):
    """Abstract strategy for edge detection algorithms."""

    @abstractmethod
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        """
        Detect contour in the given grayscale image.
        Returns (N, 2) array of contour points.
        """
        pass


class CannyDetector(EdgeDetectorStrategy):
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        if cv2 is None:
            return _fallback_canny(img, settings)
        edges = cv2.Canny(
            img,
            settings.canny_threshold1,
            settings.canny_threshold2,
            apertureSize=settings.canny_aperture_size,
            L2gradient=settings.canny_L2_gradient,
        )
        return _edges_to_xy(
            edges, settings.min_contour_length, settings.max_contour_length
        )


class ThresholdDetector(EdgeDetectorStrategy):
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        if cv2 is None:
            # Simple threshold fallback
            _, edges = (img > settings.threshold_value).astype(np.uint8) * 255
            return _edges_to_xy(
                edges, settings.min_contour_length, settings.max_contour_length
            )

        thresh_type = getattr(
            cv2, f"THRESH_{settings.threshold_type.upper()}", cv2.THRESH_BINARY
        )
        _, edges = cv2.threshold(
            img, settings.threshold_value, settings.threshold_max_value, thresh_type
        )
        return _edges_to_xy(
            edges, settings.min_contour_length, settings.max_contour_length
        )


class SobelDetector(EdgeDetectorStrategy):
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        if cv2 is None:
            return _fallback_canny(img, settings)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=settings.sobel_kernel_size)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=settings.sobel_kernel_size)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, edges = cv2.threshold(
            magnitude,
            settings.threshold_value,
            settings.threshold_max_value,
            cv2.THRESH_BINARY,
        )
        return _edges_to_xy(
            edges, settings.min_contour_length, settings.max_contour_length
        )


class ScharrDetector(EdgeDetectorStrategy):
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        if cv2 is None:
            return _fallback_canny(img, settings)
        grad_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, edges = cv2.threshold(
            magnitude,
            settings.threshold_value,
            settings.threshold_max_value,
            cv2.THRESH_BINARY,
        )
        return _edges_to_xy(
            edges, settings.min_contour_length, settings.max_contour_length
        )


class LaplacianDetector(EdgeDetectorStrategy):
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        if cv2 is None:
            return _fallback_canny(img, settings)
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=settings.laplacian_kernel_size)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, edges = cv2.threshold(
            laplacian,
            settings.threshold_value,
            settings.threshold_max_value,
            cv2.THRESH_BINARY,
        )
        return _edges_to_xy(
            edges, settings.min_contour_length, settings.max_contour_length
        )


class ActiveContourDetector(EdgeDetectorStrategy):
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        # Placeholder for Active Contour implementation
        logger.warning(
            "Active contour method is not fully implemented yet. Using Canny fallback."
        )
        return _fallback_canny(img, settings)


# Register strategies
EDGE_DETECTORS.register("canny", CannyDetector().detect)
EDGE_DETECTORS.register("threshold", ThresholdDetector().detect)
EDGE_DETECTORS.register("sobel", SobelDetector().detect)
EDGE_DETECTORS.register("scharr", ScharrDetector().detect)
EDGE_DETECTORS.register("laplacian", LaplacianDetector().detect)
EDGE_DETECTORS.register("active_contour", ActiveContourDetector().detect)


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
    """
    Run a contour detector on the current image and write to ctx.contour.xy.
    """
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

    # prepare a logger for visible runtime info
    logger = logging.getLogger(__name__)

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
    if roi_image_override is not None:
        img_roi = roi_image_override
        if roi:
            x0, y0 = int(roi[0]), int(roi[1])
    elif roi:
        x0, y0, w, h = [int(v) for v in roi]
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
            xy = _edges_to_xy(
                edges, settings.min_contour_length, settings.max_contour_length
            )
        if xy.size == 0:
            raise RuntimeError(
                "EdgeDetection: detector returned no contour even after fallback."
            )

    # 5) offset back to full-image coords
    if roi:
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
            solid_interface_xy = _edges_to_xy(
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
