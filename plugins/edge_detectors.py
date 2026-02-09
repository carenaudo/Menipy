"""
Edge detection plugins for Menipy.

Provides various edge detection algorithms beyond the core implementations:
- Otsu thresholding
- Adaptive thresholding  
- Laplacian of Gaussian (LoG) with optional zero-crossing
- Improved Snake (Active Contour) with substrate masking

Usage:
    These detectors are automatically registered to EDGE_DETECTORS registry
    when this module is imported.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
# NOTE: Heavy imports (cv2, skimage) are moved inside functions for lazy loading

from pydantic import BaseModel, Field, ConfigDict
from menipy.common.registry import EDGE_DETECTORS
from menipy.models.config import EdgeDetectionSettings
from menipy.common.plugin_settings import register_detector_settings, resolve_plugin_settings
from menipy.common.image_utils import edges_to_xy, ensure_gray
# NOTE: Removed import of _fallback_canny from edge_detection to avoid circular imports.
# Canny fallback is now simple or reimplemented locally if needed.

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# Use edges_to_xy from image_utils
# But keeping _edges_to_xy for backward compatibility if needed, aliased
_edges_to_xy = edges_to_xy


def _simple_threshold_fallback(img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
    """Simple threshold fallback when OpenCV is unavailable."""
    gray = ensure_gray(img)
    v = float(np.median(gray))
    edges = (gray > v).astype(np.uint8) * 255
    return _edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


# -----------------------------------------------------------------------------
# Migrated Core Detectors
# -----------------------------------------------------------------------------

class CannyDetector:
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        # Resolve settings using new CannySettings model
        defaults = {
            "threshold1": settings.canny_threshold1,
            "threshold2": settings.canny_threshold2,
            "aperture_size": settings.canny_aperture_size,
            "L2gradient": settings.canny_L2_gradient,
        }
        raw = resolve_plugin_settings("canny", getattr(settings, "plugin_settings", {}), **defaults)
        cfg = CannySettings(**raw)

        """Detect edges using Canny (with local fallback)."""
        try:
            import cv2
        except Exception:
            cv2 = None

        if cv2 is None:
            return _simple_threshold_fallback(img, settings)

        g = ensure_gray(img)
        edges = cv2.Canny(
            g,
            cfg.threshold1,
            cfg.threshold2,
            apertureSize=cfg.aperture_size,
            L2gradient=cfg.L2gradient,
        )
        return edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


class ThresholdDetector:
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        # Resolve settings
        defaults = {
            "threshold_value": settings.threshold_value,
            "max_value": settings.threshold_max_value,
            "type": settings.threshold_type,
        }
        raw = resolve_plugin_settings("threshold", getattr(settings, "plugin_settings", {}), **defaults)
        cfg = ThresholdSettings(**raw)

        """Detect edges using simple/global threshold or OpenCV threshold."""
        try:
            import cv2
        except Exception:
            cv2 = None

        g = ensure_gray(img)

        if cv2 is None:
            edges = (g > settings.threshold_value).astype(np.uint8) * 255
            return edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)

        thresh_type = getattr(cv2, f"THRESH_{cfg.type.upper()}", cv2.THRESH_BINARY)
        _, edges = cv2.threshold(g, cfg.threshold_value, cfg.max_value, thresh_type)
        return edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


class SobelDetector:
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        """Detect edges using Sobel gradients (or fallback)."""
        defaults = {
            "kernel_size": settings.sobel_kernel_size,
            "threshold_value": settings.threshold_value,
            "max_value": settings.threshold_max_value,
        }
        raw = resolve_plugin_settings("sobel", getattr(settings, "plugin_settings", {}), **defaults)
        cfg = SobelSettings(**raw)

        try:
            import cv2
        except Exception:
            cv2 = None

        g = ensure_gray(img)

        if cv2 is None:
            return CannyDetector().detect(img, settings)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=cfg.kernel_size)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=cfg.kernel_size)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, edges = cv2.threshold(
            magnitude,
            cfg.threshold_value,
            cfg.max_value,
            cv2.THRESH_BINARY,
        )
        return edges_to_xy(
            edges, settings.min_contour_length, settings.max_contour_length
        )


class ScharrDetector:
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        """Detect edges using Scharr operator (or fallback)."""
        defaults = {
            "threshold_value": settings.threshold_value,
            "max_value": settings.threshold_max_value,
        }
        raw = resolve_plugin_settings("scharr", getattr(settings, "plugin_settings", {}), **defaults)
        cfg = ScharrSettings(**raw)

        try:
            import cv2
        except Exception:
            cv2 = None

        g = ensure_gray(img)

        if cv2 is None:
            return CannyDetector().detect(img, settings)

        grad_x = cv2.Scharr(g, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(g, cv2.CV_64F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(magnitude, cfg.threshold_value, cfg.max_value, cv2.THRESH_BINARY)
        return edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


class LaplacianBasicDetector:
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        """Detect edges using the Laplacian operator (or fallback)."""
        defaults = {
            "kernel_size": settings.laplacian_kernel_size,
            "threshold_value": settings.threshold_value,
            "max_value": settings.threshold_max_value,
        }
        raw = resolve_plugin_settings("laplacian", getattr(settings, "plugin_settings", {}), **defaults)
        cfg = LaplacianSettings(**raw)

        try:
            import cv2
        except Exception:
            cv2 = None

        g = ensure_gray(img)

        if cv2 is None:
            return CannyDetector().detect(img, settings)
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=cfg.kernel_size)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, edges = cv2.threshold(
            laplacian,
            cfg.threshold_value,
            cfg.max_value,
            cv2.THRESH_BINARY,
        )
        return _edges_to_xy(
            edges, settings.min_contour_length, settings.max_contour_length
        )


class LegacySnakeDetector:
    """Old 'ActiveContourDetector' from core."""
    def detect(self, img: np.ndarray, settings: EdgeDetectionSettings) -> np.ndarray:
        """Run legacy active-contour (snake) refinement if scikit-image is available."""
        try:
            from skimage.segmentation import active_contour
            from skimage.filters import gaussian
        except Exception:
            logger.error("LegacySnakeDetector requires scikit-image.")
            return CannyDetector().detect(img, settings)
            
        defaults = {
            "iterations": settings.snake_iterations,
            "alpha": settings.snake_alpha,
            "beta": settings.snake_beta,
            "gamma": settings.snake_gamma,
        }
        # Try resolving for 'legacy_snake' or 'active_contour' depending on method name in settings?
        # But here valid method names are passed implicitly.
        # Let's resolve 'legacy_snake' as primary
        raw = resolve_plugin_settings("legacy_snake", getattr(settings, "plugin_settings", {}), **defaults)
        cfg = LegacySnakeSettings(**raw)

        # 1. Initial Contour using Canny/Otsu
        initial_xy = CannyDetector().detect(img, settings)

        if initial_xy.size < 3:
            return initial_xy

        img_smooth = gaussian(img, sigma=getattr(settings, 'gaussian_sigma_x', 1.0))

        # 3. Coordinate Swap (x,y) -> (row,col)
        init_snake_rc = initial_xy[:, ::-1]

        # 4. Run Snake
        snake_rc = active_contour(
            img_smooth,
            init_snake_rc,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            max_num_iter=cfg.iterations,
            convergence=0.01 
        )

        # Convert back to (x,y) coordinates and return
        snake_xy = snake_rc[:, ::-1]
        return snake_xy


# -----------------------------------------------------------------------------
# Core Detector Settings Models
# -----------------------------------------------------------------------------

from typing import Literal

class CannySettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    threshold1: int = Field(50, ge=0, le=255)
    threshold2: int = Field(150, ge=0, le=255)
    aperture_size: Literal[3, 5, 7] = Field(3)
    L2gradient: bool = Field(False)

class ThresholdSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    threshold_value: int = Field(128, ge=0, le=255)
    max_value: int = Field(255, ge=0, le=255)
    type: Literal["binary", "binary_inv", "trunc", "to_zero", "to_zero_inv"] = Field("binary")

class SobelSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    kernel_size: int = Field(3, ge=1, description="Must be odd")
    threshold_value: int = Field(0, ge=0, le=255) # Sobel magnitude often needs thresh
    max_value: int = Field(255, ge=0, le=255)

class ScharrSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    threshold_value: int = Field(0, ge=0, le=255)
    max_value: int = Field(255, ge=0, le=255)

class LaplacianSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    kernel_size: int = Field(1, ge=1, description="Must be odd")
    threshold_value: int = Field(0, ge=0, le=255)
    max_value: int = Field(255, ge=0, le=255)

class LegacySnakeSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    iterations: int = Field(100, ge=1)
    alpha: float = Field(0.01, ge=0.0)
    beta: float = Field(0.1, ge=0.0)
    gamma: float = Field(0.001, ge=0.0)
    
# Register these models immediately so they are available
register_detector_settings("canny", CannySettings)
register_detector_settings("threshold", ThresholdSettings)
register_detector_settings("sobel", SobelSettings)
register_detector_settings("scharr", ScharrSettings)
register_detector_settings("laplacian", LaplacianSettings)
register_detector_settings("active_contour", LegacySnakeSettings)
register_detector_settings("legacy_snake", LegacySnakeSettings)


# -----------------------------------------------------------------------------
# Advanced/Custom Detectors
# -----------------------------------------------------------------------------

def _zero_crossing_detection(
    laplacian: np.ndarray,
    min_gradient: float = 5.0,
) -> np.ndarray:
    """Detect zero crossings in a Laplacian image."""
    abs_lap = np.abs(laplacian)
    sign_img = np.sign(laplacian)
    
    z_c_image = np.zeros(laplacian.shape, dtype=np.uint8)
    
    # Check horizontal zero crossings
    zero_cross_h = (sign_img[:, :-1] * sign_img[:, 1:]) < 0
    grad_h = np.maximum(abs_lap[:, :-1], abs_lap[:, 1:])
    
    # Check vertical zero crossings
    zero_cross_v = (sign_img[:-1, :] * sign_img[1:, :]) < 0
    grad_v = np.maximum(abs_lap[:-1, :], abs_lap[1:, :])
    
    # Apply threshold
    z_c_image[:, :-1] |= ((zero_cross_h & (grad_h > min_gradient)) * 255).astype(np.uint8)
    z_c_image[:-1, :] |= ((zero_cross_v & (grad_v > min_gradient)) * 255).astype(np.uint8)
    
    return z_c_image


# -----------------------------------------------------------------------------
# Otsu Edge Detector
# -----------------------------------------------------------------------------

class OtsuEdgeDetector:
    """Edge detection using Otsu's automatic thresholding method."""
    
    def detect(
        self,
        img: np.ndarray,
        settings: EdgeDetectionSettings,
    ) -> np.ndarray:
        import cv2

        if settings.gaussian_blur_before:
            ksize = settings.gaussian_kernel_size
            if ksize % 2 == 0:
                ksize += 1
            img = cv2.GaussianBlur(img, (ksize, ksize), settings.gaussian_sigma_x)
        
        _, edges = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return _edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


# -----------------------------------------------------------------------------
# Adaptive Threshold Edge Detector
# -----------------------------------------------------------------------------

class AdaptiveSettings(BaseModel):
    """Settings for Adaptive Threshold detector."""
    model_config = ConfigDict(extra='ignore')
    
    adaptive_block_size: int = Field(21, ge=3, description="Block size (must be odd)")
    adaptive_c: int = Field(2, description="Constant subtracted from mean")

    def model_post_init(self, __context):
        """Model_post_init."""
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1


class AdaptiveEdgeDetector:
    """Edge detection using adaptive (local) thresholding."""
    
    def detect(
        self,
        img: np.ndarray,
        settings: EdgeDetectionSettings,
    ) -> np.ndarray:
        import cv2

        if settings.gaussian_blur_before:
            ksize = settings.gaussian_kernel_size
            if ksize % 2 == 0:
                ksize += 1
            img = cv2.GaussianBlur(img, (ksize, ksize), settings.gaussian_sigma_x)
        
        # Resolve settings
        raw_cfg = resolve_plugin_settings("adaptive", getattr(settings, "plugin_settings", {}))
        cfg = AdaptiveSettings(**raw_cfg)
        
        edges = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            cfg.adaptive_block_size,
            cfg.adaptive_c
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return _edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


# -----------------------------------------------------------------------------
# Laplacian of Gaussian (LoG) Edge Detector
# -----------------------------------------------------------------------------

class LoGSettings(BaseModel):
    """Settings for LoG detector."""
    model_config = ConfigDict(extra='ignore')

    log_sigma: float = Field(1.0, ge=0.1)
    log_min_gradient: float = Field(5.0, ge=0.0)
    log_use_zero_crossing: bool = Field(False)


class LoGEdgeDetector:
    """Edge detection using Laplacian of Gaussian (LoG)."""
    
    def detect(
        self,
        img: np.ndarray,
        settings: EdgeDetectionSettings,
    ) -> np.ndarray:
        import cv2

        # Resolve settings
        raw_cfg = resolve_plugin_settings("log", getattr(settings, "plugin_settings", {}))
        cfg = LoGSettings(**raw_cfg)
        
        ksize = int(6 * cfg.log_sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)
        
        blur = cv2.GaussianBlur(img, (ksize, ksize), cfg.log_sigma)
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        
        if cfg.log_use_zero_crossing:
            edges = _zero_crossing_detection(laplacian, cfg.log_min_gradient)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            h, w = edges.shape
            filled = edges.copy()
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
                if filled[seed[1], seed[0]] == 0:
                    cv2.floodFill(filled, mask, seed, 128)
            
            interior = np.where(filled == 0, 255, 0).astype(np.uint8)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(interior, cv2.MORPH_OPEN, kernel_small)
        else:
            edges = cv2.convertScaleAbs(laplacian)
            _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return _edges_to_xy(edges, settings.min_contour_length, settings.max_contour_length)


# -----------------------------------------------------------------------------
# Improved Snake (Active Contour) Detector
# -----------------------------------------------------------------------------

class ImprovedSnakeSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    iterations: int = Field(500, ge=1)
    alpha: float = Field(0.015, ge=0.0)
    beta: float = Field(10.0, ge=0.0)
    gamma: float = Field(0.001, ge=0.0)

register_detector_settings("improved_snake", ImprovedSnakeSettings)


class ImprovedSnakeDetector:
    """
    Enhanced active contour (snake) edge detection.
    """
    
    def detect(
        self,
        img: np.ndarray,
        settings: EdgeDetectionSettings,
        substrate_y: Optional[int] = None,
        needle_rect: Optional[Tuple[int, int, int, int]] = None,
        return_debug: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, list]:
        defaults = {
            "iterations": settings.snake_iterations,
            "alpha": settings.snake_alpha,
            "beta": settings.snake_beta,
            "gamma": settings.snake_gamma,
        }
        raw = resolve_plugin_settings("improved_snake", getattr(settings, "plugin_settings", {}), **defaults)
        cfg = ImprovedSnakeSettings(**raw)

        import cv2

        try:
            from skimage.segmentation import active_contour
            from skimage.filters import gaussian as skimage_gaussian
        except ImportError:
            logger.error("ImprovedSnakeDetector requires scikit-image.")
            res = OtsuEdgeDetector().detect(img, settings)
            return (res, []) if return_debug else res
        
        h, w = img.shape[:2]
        
        # Source 1: Otsu
        _, otsu_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Source 2: Canny
        enhanced = cv2.GaussianBlur(img, (5, 5), 0)
        canny_edges = cv2.Canny(enhanced, 30, 100)
        canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        all_candidates = []
        for c in otsu_contours:
            all_candidates.append((c, "otsu"))
        for c in canny_contours:
            all_candidates.append((c, "canny"))
        
        debug_info = []
        if return_debug:
             # Store raw candidates
             for c, src in all_candidates:
                 area = cv2.contourArea(c)
                 if area > 10:
                     xy = c.reshape(-1, 2).astype(float)
                     debug_info.append((xy, area, f"{src}"))

        if not all_candidates:
            res = np.empty((0, 2), float)
            return (res, debug_info) if return_debug else res
        
        scored = []
        for cnt, src in all_candidates:
            score = self._score_contour(cnt, h, needle_rect, substrate_y)
            if score > 0:
                scored.append((cnt, score, src))
                if return_debug:
                    xy = cnt.reshape(-1, 2).astype(float)
                    debug_info.append((xy, score, f"{src}-scored:{int(score)}"))
        
        if not scored:
            res = np.empty((0, 2), float)
            return (res, debug_info) if return_debug else res
        
        scored.sort(key=lambda x: x[1], reverse=True)
        best_cnt = scored[0][0]
        initial_xy = best_cnt.reshape(-1, 2).astype(float)
        
        snake_img = img.copy()
        if substrate_y is not None and 0 <= substrate_y < h:
            initial_xy[:, 1] = np.minimum(initial_xy[:, 1], substrate_y - 1)
            snake_img[substrate_y:, :] = 255
        
        img_smooth = skimage_gaussian(snake_img.astype(np.float64) / 255.0, sigma=2.0)
        init_rc = initial_xy[:, ::-1]
        
        snake_rc = active_contour(
            img_smooth,
            init_rc,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            w_line=-1.0,
            w_edge=1.0,
            max_px_move=1.0,
            max_num_iter=cfg.iterations,
            convergence=0.01
        )
        
        result_xy = snake_rc[:, ::-1]
        
        if return_debug:
            return result_xy, debug_info
        return result_xy
    
    def _score_contour(
        self,
        cnt: np.ndarray,
        roi_h: int,
        needle_rect: Optional[Tuple[int, int, int, int]],
        substrate_y: Optional[int],
    ) -> float:
        import cv2

        area = cv2.contourArea(cnt)
        if area < 50:
            return -1.0
        
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cy = M["m01"] / M["m00"]
        else:
            cy = y + h / 2
        
        if needle_rect is not None:
            nx, ny, nw, nh = needle_rect
            if cy < ny + nh and cy < roi_h * 0.5:
                return -1.0
        
        position_score = cy / roi_h
        if position_score > 0.3:
            return area * (1 + 2 * position_score)
        else:
            return area * 0.2


# -----------------------------------------------------------------------------
# Registry Registration
# -----------------------------------------------------------------------------

# Register detectors (Core ones registered at top, ensure these are too)
EDGE_DETECTORS.register("otsu", OtsuEdgeDetector().detect)
EDGE_DETECTORS.register("adaptive", AdaptiveEdgeDetector().detect)
EDGE_DETECTORS.register("log", LoGEdgeDetector().detect)
EDGE_DETECTORS.register("improved_snake", ImprovedSnakeDetector().detect)

# Register settings
register_detector_settings("log", LoGSettings)
register_detector_settings("adaptive", AdaptiveSettings)

# Settings for these plugins already registered via calls above or previously.
# Canny/etc registered at top.

logger.info("Registered edge detection plugins: otsu, adaptive, log, improved_snake, canny, etc.")
