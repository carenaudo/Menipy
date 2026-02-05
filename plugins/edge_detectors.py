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

import cv2
import numpy as np

from pydantic import BaseModel, Field, ConfigDict
from menipy.common.registry import EDGE_DETECTORS
from menipy.models.config import EdgeDetectionSettings
from menipy.common.plugin_settings import register_detector_settings, resolve_plugin_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _edges_to_xy(
    edges: np.ndarray,
    min_len: int = 0,
    max_len: int = 100000,
) -> np.ndarray:
    """Convert a binary edges mask to an (N,2) contour array (largest external contour)."""
    if edges is None:
        return np.empty((0, 2), float)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), float)
    
    # Filter by length
    valid_cnts = [c for c in contours if min_len <= len(c) <= max_len]
    if not valid_cnts:
        return np.empty((0, 2), float)
    
    # Select largest by area
    c = max(valid_cnts, key=cv2.contourArea)
    xy = c.reshape(-1, 2).astype(float)
    return xy


def _zero_crossing_detection(
    laplacian: np.ndarray,
    min_gradient: float = 5.0,
) -> np.ndarray:
    """
    Detect zero crossings in a Laplacian image.
    """
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
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1


class AdaptiveEdgeDetector:
    """Edge detection using adaptive (local) thresholding."""
    
    def detect(
        self,
        img: np.ndarray,
        settings: EdgeDetectionSettings,
    ) -> np.ndarray:
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

class ImprovedSnakeDetector:
    """
    Enhanced active contour (snake) edge detection.
    Using standard config settings for snake params (alpha, beta, etc)
    as they are part of core configuration.
    """
    
    def detect(
        self,
        img: np.ndarray,
        settings: EdgeDetectionSettings,
        substrate_y: Optional[int] = None,
        needle_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        try:
            from skimage.segmentation import active_contour
            from skimage.filters import gaussian as skimage_gaussian
        except ImportError:
            logger.error("ImprovedSnakeDetector requires scikit-image.")
            return OtsuEdgeDetector().detect(img, settings)
        
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
        
        if not all_candidates:
            return np.empty((0, 2), float)
        
        scored = []
        for cnt, src in all_candidates:
            score = self._score_contour(cnt, h, needle_rect, substrate_y)
            if score > 0:
                scored.append((cnt, score, src))
        
        if not scored:
            return np.empty((0, 2), float)
        
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
            alpha=settings.snake_alpha,
            beta=settings.snake_beta,
            gamma=settings.snake_gamma,
            w_line=-1.0,
            w_edge=1.0,
            max_px_move=1.0,
            max_num_iter=settings.snake_iterations,
            convergence=0.01
        )
        
        return snake_rc[:, ::-1]
    
    def _score_contour(
        self,
        cnt: np.ndarray,
        roi_h: int,
        needle_rect: Optional[Tuple[int, int, int, int]],
        substrate_y: Optional[int],
    ) -> float:
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

DETECTOR_SETTINGS = {
    "log": LoGSettings,
    "adaptive": AdaptiveSettings,
}

# Register detectors
EDGE_DETECTORS.register("otsu", OtsuEdgeDetector().detect)
EDGE_DETECTORS.register("adaptive", AdaptiveEdgeDetector().detect)
EDGE_DETECTORS.register("log", LoGEdgeDetector().detect)
EDGE_DETECTORS.register("improved_snake", ImprovedSnakeDetector().detect)

# Register settings
register_detector_settings("log", LoGSettings)
register_detector_settings("adaptive", AdaptiveSettings)

logger.info("Registered edge detection plugins: otsu, adaptive, log, improved_snake")
