"""
Image Quality Analysis Utility Plugin

Provides image quality metrics useful for droplet analysis:
- Contrast ratio
- Brightness uniformity
- Edge sharpness
- Noise estimation
- Otsu variance ratio (bimodal separation)
- Overall quality score
"""

import numpy as np
import logging

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


def image_quality(image: np.ndarray) -> dict:
    """Analyze image quality metrics for droplet analysis.
    
    Evaluates various quality metrics that affect droplet detection:
    - contrast: Michelson contrast (higher is better)
    - brightness_uniformity: CV of local means (lower is better)
    - edge_sharpness: Mean gradient magnitude (higher is better)
    - noise_estimate: Laplacian variance (lower is less noise)
    - bimodal_separation: Otsu variance ratio (higher is better)
    - overall_quality: Composite score from 0-100
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        Dictionary with quality metrics
    """
    if cv2 is None:
        return {"error": "OpenCV not available"}
    
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    results = {}
    
    # 1. Contrast (Michelson)
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    if max_val + min_val > 0:
        contrast = (max_val - min_val) / (max_val + min_val)
    else:
        contrast = 0.0
    results["contrast"] = contrast
    
    # 2. Brightness uniformity (coefficient of variation of local means)
    block_size = 50
    h, w = gray.shape
    local_means = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.size > 0:
                local_means.append(np.mean(block))
    
    if len(local_means) > 0 and np.mean(local_means) > 0:
        brightness_cv = np.std(local_means) / np.mean(local_means)
    else:
        brightness_cv = 0.0
    results["brightness_uniformity"] = 1.0 - min(brightness_cv, 1.0)  # Higher is better
    
    # 3. Edge sharpness (gradient magnitude)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_sharpness = float(np.mean(gradient_magnitude))
    results["edge_sharpness"] = edge_sharpness
    
    # 4. Noise estimation (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = float(laplacian.var())
    results["noise_estimate"] = noise_estimate
    
    # 5. Bimodal separation (Otsu variance ratio)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    bins = np.arange(256)
    
    max_variance = 0
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
    
    total_mean = np.sum(bins * hist)
    total_variance = np.sum(((bins - total_mean) ** 2) * hist)
    bimodal_separation = max_variance / total_variance if total_variance > 0 else 0
    results["bimodal_separation"] = bimodal_separation
    
    # 6. Overall quality score (0-100)
    # Weighted combination of metrics
    score = 0.0
    score += contrast * 25  # 0-25 points
    score += results["brightness_uniformity"] * 25  # 0-25 points
    score += min(edge_sharpness / 50, 1.0) * 25  # 0-25 points (normalized)
    score += bimodal_separation * 25  # 0-25 points
    
    results["overall_quality"] = f"{score:.0f}/100"
    results["quality_score"] = score
    
    # Quality rating
    if score >= 80:
        results["rating"] = "Excellent"
    elif score >= 60:
        results["rating"] = "Good"
    elif score >= 40:
        results["rating"] = "Fair"
    else:
        results["rating"] = "Poor"
    
    logger.info(f"Image quality analysis: {results['rating']} ({score:.0f}/100)")
    
    return results


def edge_comparison(image: np.ndarray) -> dict:
    """Compare edge detection methods on the image.
    
    Runs Canny, Otsu, and Adaptive thresholding and compares results.
    Recommends the best method based on contour quality.
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        Dictionary with comparison results and recommendation
    """
    if cv2 is None:
        return {"error": "OpenCV not available"}
    
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    results = {}
    
    # Apply each method
    methods = {}
    
    # Canny
    canny = cv2.Canny(gray, 50, 150)
    contours_canny, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_length_canny = max([len(c) for c in contours_canny]) if contours_canny else 0
    methods["canny"] = {"contours": len(contours_canny), "max_length": max_length_canny}
    
    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_otsu, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_length_otsu = max([len(c) for c in contours_otsu]) if contours_otsu else 0
    methods["otsu"] = {"contours": len(contours_otsu), "max_length": max_length_otsu}
    
    # Adaptive
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    contours_adaptive, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_length_adaptive = max([len(c) for c in contours_adaptive]) if contours_adaptive else 0
    methods["adaptive"] = {"contours": len(contours_adaptive), "max_length": max_length_adaptive}
    
    results["methods"] = methods
    
    # Compare and recommend
    # Prefer method with longest single contour (likely the droplet)
    best_method = max(methods.keys(), key=lambda m: methods[m]["max_length"])
    results["recommended_method"] = best_method
    results["recommendation_reason"] = f"Longest contour: {methods[best_method]['max_length']} points"
    
    # Add summary
    results["canny_contours"] = methods["canny"]["max_length"]
    results["otsu_contours"] = methods["otsu"]["max_length"]
    results["adaptive_contours"] = methods["adaptive"]["max_length"]
    
    logger.info(f"Edge comparison: {best_method} recommended")
    
    return results


# Register utilities
UTILITIES = {
    "image_quality": image_quality,
    "edge_comparison": edge_comparison,
}

# Auto-register with menipy
try:
    from menipy.common.registry import register_utility
    register_utility("image_quality", image_quality)
    register_utility("edge_comparison", edge_comparison)
except ImportError:
    pass
