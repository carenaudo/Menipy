"""Sessile Detection Method2.

Experimental implementation."""


import cv2
import numpy as np

def calculate_otsu_variance_ratio(image):
    """Calculate Otsu's inter-class variance ratio."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
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


def calculate_illumination_uniformity(image, block_size=50):
    """Calculate coefficient of variation of local mean intensities."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    local_means = []
    
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.size > 0:
                local_means.append(np.mean(block))
    
    cv = np.std(local_means) / np.mean(local_means) if len(local_means) > 0 and np.mean(local_means) > 0 else 0
    return cv


def calculate_edge_quality(edges):
    """
    Calculate edge quality metrics for Canny edges.
    
    Returns:
        edge_density: Ratio of edge pixels to total pixels
        edge_continuity: Measure of how continuous the edges are
    """
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / total_pixels
    
    # Find contours to measure continuity
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        edge_continuity = 0
    else:
        # Longest contour length as a measure of continuity
        max_contour_length = max([len(c) for c in contours])
        # Normalize by image perimeter
        perimeter = 2 * (edges.shape[0] + edges.shape[1])
        edge_continuity = max_contour_length / perimeter
    
    return edge_density, edge_continuity


def calculate_gradient_strength(image):
    """
    Calculate average gradient magnitude to assess edge strength.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    return avg_gradient


def auto_select_method_with_canny(image,
                                  otsu_variance_threshold=0.5,
                                  illumination_cv_threshold=0.15,
                                  gradient_strength_threshold=20.0,
                                  block_size=50,
                                  canny_low=50,
                                  canny_high=150,
                                  adaptive_block_size=11,
                                  adaptive_c=2,
                                  verbose=True):
    """
    Automatically select between Otsu, Adaptive, and Canny based on image characteristics.
    
    Decision logic:
    1. If strong gradients and good edges → Canny (best for ADSA)
    2. If good bimodal separation and uniform lighting → Otsu
    3. Otherwise → Adaptive
    
    Returns:
        result_image: Processed image (binary or edges)
        method_used: String indicating which method was used
        metrics: Dictionary with all diagnostic metrics
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate all metrics
    variance_ratio, otsu_thresh = calculate_otsu_variance_ratio(gray)
    illumination_cv = calculate_illumination_uniformity(gray, block_size)
    gradient_strength = calculate_gradient_strength(gray)
    
    # Apply Canny to check edge quality
    edges_canny = cv2.Canny(gray, canny_low, canny_high)
    edge_density, edge_continuity = calculate_edge_quality(edges_canny)
    
    # Store all metrics
    metrics = {
        'variance_ratio': variance_ratio,
        'illumination_cv': illumination_cv,
        'gradient_strength': gradient_strength,
        'edge_density': edge_density,
        'edge_continuity': edge_continuity,
        'otsu_threshold': otsu_thresh
    }
    
    if verbose:
        print("=" * 70)
        print("AUTOMATIC METHOD SELECTION (Including Canny)")
        print("=" * 70)
        print(f"1. Gradient Strength:    {gradient_strength:.2f} (threshold: {gradient_strength_threshold})")
        print(f"   → {'Strong edges detected' if gradient_strength >= gradient_strength_threshold else 'Weak edges'}")
        print(f"2. Edge Continuity:      {edge_continuity:.4f}")
        print(f"   → {'Good continuous edges' if edge_continuity > 0.3 else 'Fragmented edges'}")
        print()
        print(f"3. Otsu Variance Ratio:  {variance_ratio:.4f} (threshold: {otsu_variance_threshold})")
        print(f"   → {'Good bimodal separation' if variance_ratio >= otsu_variance_threshold else 'Poor separation'}")
        print()
        print(f"4. Illumination CV:      {illumination_cv:.4f} (threshold: {illumination_cv_threshold})")
        print(f"   → {'Uniform lighting' if illumination_cv <= illumination_cv_threshold else 'Non-uniform lighting'}")
        print()
        print("-" * 70)
    
    # Decision logic (prioritized)
    
    # PRIORITY 1: Canny if strong gradients and continuous edges
    if gradient_strength >= gradient_strength_threshold and edge_continuity > 0.3:
        if verbose:
            print("✓ DECISION: Using CANNY edge detection")
            print("  Reason: Strong, continuous edges detected")
            print("  → BEST for ADSA droplet contour extraction")
        result = edges_canny
        method_used = "Canny"
    
    # PRIORITY 2: Otsu if good separation and uniform lighting
    elif variance_ratio >= otsu_variance_threshold and illumination_cv <= illumination_cv_threshold:
        if verbose:
            print("✓ DECISION: Using OTSU thresholding")
            print("  Reason: Good bimodal separation & uniform lighting")
            print("  → Extract contours from binary mask for ADSA")
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        method_used = "Otsu"
    
    # PRIORITY 3: Adaptive for everything else
    else:
        if verbose:
            print("✓ DECISION: Using ADAPTIVE thresholding")
            reasons = []
            if gradient_strength < gradient_strength_threshold:
                reasons.append("weak gradients")
            if edge_continuity <= 0.3:
                reasons.append("fragmented edges")
            if variance_ratio < otsu_variance_threshold:
                reasons.append("poor bimodal separation")
            if illumination_cv > illumination_cv_threshold:
                reasons.append("non-uniform lighting")
            print(f"  Reason(s): {', '.join(reasons)}")
            print("  → Extract contours from binary mask for ADSA")
        
        result = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       adaptive_block_size,
                                       adaptive_c)
        method_used = "Adaptive"
    
    if verbose:
        print("=" * 70)
        print()
    
    return result, method_used, metrics


def compare_all_methods(image, canny_low=50, canny_high=150):
    """
    Apply all three methods and display side-by-side for comparison.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply all methods
    _, otsu_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
    canny_result = cv2.Canny(gray, canny_low, canny_high)
    
    # Create comparison display
    h, w = gray.shape
    comparison = np.zeros((h * 2, w * 2), dtype=np.uint8)
    
    comparison[0:h, 0:w] = gray
    comparison[0:h, w:2*w] = otsu_result
    comparison[h:2*h, 0:w] = adaptive_result
    comparison[h:2*h, w:2*w] = canny_result
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'Otsu', (w + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'Adaptive', (10, h + 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, 'Canny', (w + 10, h + 30), font, 0.7, (255, 255, 255), 2)
    
    return comparison


# Example usage
if __name__ == "__main__":
    # Load image
    #image = cv2.imread('./data/samples/prueba sesil 2.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('./data/samples/gota depositada 1.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('./data/samples/gota pendiente 1.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('./data/samples/prueba pend 1.png', cv2.IMREAD_GRAYSCALE)
    #image = None
    if image is None:
        print("Error: Could not load image. Using synthetic example.")
        # Create synthetic droplet with good edges
        image = np.ones((400, 400), dtype=np.uint8) * 200
        cv2.circle(image, (200, 200), 100, 50, -1)
        # Add some gradient for realistic edge
        for r in range(100, 110):
            cv2.circle(image, (200, 200), r, 50 + (r-100)*15, 2)
    
    print("\n" + "="*70)
    print("AUTOMATIC METHOD SELECTION WITH CANNY")
    print("="*70 + "\n")
    
    # Automatic selection
    result, method, metrics = auto_select_method_with_canny(
        image,
        otsu_variance_threshold=0.5,
        illumination_cv_threshold=0.15,
        gradient_strength_threshold=20.0,
        canny_low=50,
        canny_high=150,
        verbose=True
    )
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    outsubinary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptivebinary = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 
                                   11, 
                                   3)
    cannybinary = cv2.Canny(gray, 50, 150)
    
    # Display automatic selection result
    result_display = np.hstack([image, result, outsubinary[1], adaptivebinary, cannybinary])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_display, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result_display, f'{method} (Auto)', (image.shape[1] + 10, 30),
                font, 1, (255, 255, 255), 2)
    
    # Display comparison of all methods
    comparison = compare_all_methods(image)
    
    cv2.imshow('Automatic Selection Result', result_display)
    cv2.imshow('Method Comparison (All)', comparison)
    
    print("\nMetrics Summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION FOR ADSA:")
    print("="*70)
    if method == "Canny":
        print("✓ Canny is ideal - use cv2.findContours() to extract droplet profile")
        print("  Then fit Young-Laplace equation to the contour points")
    elif method == "Otsu":
        print("✓ Otsu binary mask - use cv2.findContours() with RETR_EXTERNAL")
        print("  to extract the outer droplet boundary")
    else:
        print("✓ Adaptive binary mask - use cv2.findContours() with RETR_EXTERNAL")
        print("  to extract the outer droplet boundary")
    print("="*70 + "\n")
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()