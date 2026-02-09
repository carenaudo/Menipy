"""
Auto-Adaptive Edge Detection Plugin

Automatically selects the best contour detection method based on image quality metrics:
- Canny: Strong, continuous edges (best for ADSA)
- Otsu: Good bimodal separation with uniform lighting
- Adaptive: Non-uniform lighting or poor separation

Works with all pipeline types: sessile, pendant, captive bubble, oscillating, capillary rise.
"""

import numpy as np
import logging
from typing import Optional, Any

from pydantic import BaseModel, Field, ConfigDict

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from menipy.common.plugin_settings import register_detector_settings, resolve_plugin_settings
except ImportError:
    register_detector_settings = None
    resolve_plugin_settings = None

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Image Quality Metrics
# -----------------------------------------------------------------------------

def calculate_otsu_variance_ratio(gray: np.ndarray) -> tuple[float, int]:
    """Calculate Otsu's inter-class variance ratio for bimodal separation assessment.
    
    Returns:
        variance_ratio: 0-1 ratio (higher = better bimodal separation)
        best_threshold: Optimal Otsu threshold value
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    bins = np.arange(256)
    
    max_variance = 0
    best_threshold = 0
    
    for t in range(1, 256):
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        
        if w0 == 0 or w1 == 0:
            continue
        
        mu0 = np.sum(bins[:t] * hist[:t]) / w0
        mu1 = np.sum(bins[t:] * hist[t:]) / w1
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    total_mean = np.sum(bins * hist)
    total_variance = np.sum(((bins - total_mean) ** 2) * hist)
    
    variance_ratio = max_variance / total_variance if total_variance > 0 else 0
    return variance_ratio, best_threshold


def calculate_illumination_uniformity(gray: np.ndarray, block_size: int = 50) -> float:
    """Calculate coefficient of variation of local mean intensities.
    
    Returns:
        cv: Coefficient of variation (lower = more uniform lighting)
    """
    h, w = gray.shape
    local_means = []
    
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.size > 0:
                local_means.append(np.mean(block))
    
    if len(local_means) > 0 and np.mean(local_means) > 0:
        return np.std(local_means) / np.mean(local_means)
    return 0


def calculate_gradient_strength(gray: np.ndarray) -> float:
    """Calculate average gradient magnitude to assess edge strength."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return float(np.mean(gradient_magnitude))


def calculate_edge_quality(edges: np.ndarray) -> tuple[float, float]:
    """Calculate edge quality metrics.
    
    Returns:
        edge_density: Ratio of edge pixels to total pixels
        edge_continuity: Longest contour length / image perimeter
    """
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / total_pixels
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        edge_continuity = 0.0
    else:
        max_contour_length = max([len(c) for c in contours])
        perimeter = 2 * (edges.shape[0] + edges.shape[1])
        edge_continuity = max_contour_length / perimeter
    
    return edge_density, edge_continuity


# -----------------------------------------------------------------------------
# Contour Extraction
# -----------------------------------------------------------------------------

def _edges_to_contour(edges: np.ndarray, min_len: int = 0, max_len: int = 100000) -> np.ndarray:
    """Convert edge image to largest (N,2) contour array."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=float)
    
    valid_cnts = [c for c in contours if min_len <= len(c) <= max_len]
    if not valid_cnts:
        return np.empty((0, 2), dtype=float)
    
    best = max(valid_cnts, key=cv2.contourArea)
    return best.reshape(-1, 2).astype(float)


def _binary_to_contour(binary: np.ndarray, min_len: int = 0, max_len: int = 100000) -> np.ndarray:
    """Extract largest contour from binary mask."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=float)
    
    valid_cnts = [c for c in contours if min_len <= len(c) <= max_len]
    if not valid_cnts:
        return np.empty((0, 2), dtype=float)
    
    best = max(valid_cnts, key=cv2.contourArea)
    return best.reshape(-1, 2).astype(float)


# -----------------------------------------------------------------------------
# Main Detection Function
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Settings Model
# -----------------------------------------------------------------------------

class AutoAdaptiveSettings(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    otsu_variance_threshold: float = Field(0.5, ge=0.0, le=1.0)
    illumination_cv_threshold: float = Field(0.15, ge=0.0)
    gradient_strength_threshold: float = Field(20.0, ge=0.0)
    canny_low: int = Field(50, ge=0, le=255)
    canny_high: int = Field(150, ge=0, le=255)
    adaptive_block_size: int = Field(11, ge=3, description="Block size (must be odd)")
    adaptive_c: int = Field(2)

    def model_post_init(self, __context):
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1

if register_detector_settings:
    register_detector_settings("auto_adaptive", AutoAdaptiveSettings)


# -----------------------------------------------------------------------------
# Main Detection Function
# -----------------------------------------------------------------------------

def auto_adaptive_detect(
    img: np.ndarray,
    settings=None,
    **kwargs
) -> np.ndarray:
    """Automatically select and apply the best edge detection method.
    
    Decision logic:
    1. If strong gradients and continuous edges -> Canny
    2. If good bimodal separation and uniform lighting -> Otsu
    3. Otherwise -> Adaptive thresholding
    """
    if cv2 is None:
        logger.warning("OpenCV not available - returning empty contour")
        return np.empty((0, 2), dtype=float)

    # 1. Resolve Settings
    defaults = {
        "otsu_variance_threshold": 0.5,
        "illumination_cv_threshold": 0.15,
        "gradient_strength_threshold": 20.0,
        "canny_low": 50,
        "canny_high": 150,
        "adaptive_block_size": 11,
        "adaptive_c": 2
    }
    # Update with kwargs overrides
    defaults.update(kwargs)
    
    plugin_settings = getattr(settings, "plugin_settings", {}) if settings else {}
    
    if resolve_plugin_settings:
        raw_cfg = resolve_plugin_settings("auto_adaptive", plugin_settings, **defaults)
        cfg = AutoAdaptiveSettings(**raw_cfg)
    else:
        # Fallback if resolving not available (standalone)
        cfg = AutoAdaptiveSettings(**defaults)

    if settings is not None:
         min_len = getattr(settings, 'min_contour_length', 50)
         max_len = getattr(settings, 'max_contour_length', 100000)
    else:
         min_len = 50
         max_len = 100000
    
    # Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Calculate quality metrics
    variance_ratio, _ = calculate_otsu_variance_ratio(gray)
    illumination_cv = calculate_illumination_uniformity(gray)
    gradient_strength = calculate_gradient_strength(gray)
    
    # Apply Canny to check edge quality (using cfg params)
    edges_canny = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    _, edge_continuity = calculate_edge_quality(edges_canny)
    
    method_used = "Unknown"
    
    # PRIORITY 1: Canny
    if gradient_strength >= cfg.gradient_strength_threshold and edge_continuity > 0.3:
        method_used = "Canny"
        xy = _edges_to_contour(edges_canny, min_len, max_len)
    
    # PRIORITY 2: Otsu
    elif variance_ratio >= cfg.otsu_variance_threshold and illumination_cv <= cfg.illumination_cv_threshold:
        method_used = "Otsu"
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        xy = _binary_to_contour(binary, min_len, max_len)
    
    # PRIORITY 3: Adaptive
    else:
        method_used = "Adaptive"
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            cfg.adaptive_block_size,
            cfg.adaptive_c
        )
        xy = _binary_to_contour(binary, min_len, max_len)
    
    logger.info(f"auto_adaptive_detect: selected {method_used} "
                f"(grad={gradient_strength:.1f}, cont={edge_continuity:.2f}, "
                f"var={variance_ratio:.2f}, illum={illumination_cv:.2f})")
    
    return xy


# -----------------------------------------------------------------------------
# Plugin Registration
# -----------------------------------------------------------------------------

# Expose for discovery
EDGE_DETECTORS = {"auto_adaptive": auto_adaptive_detect}

# Register with menipy when imported
try:
    from menipy.common.registry import register_edge
    register_edge("auto_adaptive", auto_adaptive_detect)
except ImportError:
    # Registry not available in standalone contexts
    pass
