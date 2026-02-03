import cv2
import numpy as np

def calculate_otsu_variance_ratio(image):
    """
    Calculate the normalized inter-class variance from Otsu's method.
    
    Returns:
        variance_ratio: Value between 0 and 1. Higher values indicate 
                       better separation between background and foreground.
        best_threshold: The optimal threshold value found by Otsu.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Calculate cumulative sums and means
    bins = np.arange(256)
    
    # For each possible threshold
    max_variance = 0
    best_threshold = 0
    
    for t in range(1, 256):
        # Weight of background class
        w0 = np.sum(hist[:t])
        # Weight of foreground class
        w1 = np.sum(hist[t:])
        
        if w0 == 0 or w1 == 0:
            continue
        
        # Mean of background class
        mu0 = np.sum(bins[:t] * hist[:t]) / w0
        # Mean of foreground class
        mu1 = np.sum(bins[t:] * hist[t:]) / w1
        
        # Inter-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    # Total variance of the image
    total_mean = np.sum(bins * hist)
    total_variance = np.sum(((bins - total_mean) ** 2) * hist)
    
    # Normalized ratio (0 to 1)
    if total_variance == 0:
        variance_ratio = 0
    else:
        variance_ratio = max_variance / total_variance
    
    return variance_ratio, best_threshold


def calculate_illumination_uniformity(image, block_size=50):
    """
    Calculate the coefficient of variation of local mean intensities.
    
    Returns:
        cv: Coefficient of variation. Lower values indicate more uniform illumination.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    local_means = []
    
    # Divide image into blocks and calculate local means
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.size > 0:
                local_means.append(np.mean(block))
    
    # Calculate coefficient of variation
    if len(local_means) > 0 and np.mean(local_means) > 0:
        cv = np.std(local_means) / np.mean(local_means)
    else:
        cv = 0
    
    return cv


def auto_select_threshold_method(image, 
                                 otsu_variance_threshold=0.5,
                                 illumination_cv_threshold=0.15,
                                 block_size=50,
                                 adaptive_block_size=11,
                                 adaptive_c=2,
                                 verbose=True):
    """
    Automatically select between Otsu and Adaptive thresholding based on:
    1. Otsu's inter-class variance ratio
    2. Illumination uniformity (coefficient of variation)
    
    Parameters:
        image: Input grayscale image
        otsu_variance_threshold: Minimum variance ratio to use Otsu (default 0.5)
        illumination_cv_threshold: Maximum CV to use Otsu (default 0.15)
        block_size: Block size for illumination uniformity calculation
        adaptive_block_size: Block size for adaptive thresholding (must be odd)
        adaptive_c: Constant subtracted from mean in adaptive thresholding
        verbose: Print decision details
    
    Returns:
        binary_image: Thresholded binary image
        method_used: String indicating which method was used
        metrics: Dictionary with diagnostic metrics
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate both metrics
    variance_ratio, otsu_thresh = calculate_otsu_variance_ratio(gray)
    illumination_cv = calculate_illumination_uniformity(gray, block_size)
    
    # Store metrics
    metrics = {
        'variance_ratio': variance_ratio,
        'illumination_cv': illumination_cv,
        'otsu_threshold': otsu_thresh
    }
    
    # Decision logic: BOTH conditions must be met for Otsu
    use_otsu = (variance_ratio >= otsu_variance_threshold and 
                illumination_cv <= illumination_cv_threshold)
    
    if verbose:
        print("=" * 60)
        print("AUTOMATIC THRESHOLD METHOD SELECTION")
        print("=" * 60)
        print(f"Otsu variance ratio:     {variance_ratio:.4f} (threshold: {otsu_variance_threshold})")
        print(f"  → {'PASS' if variance_ratio >= otsu_variance_threshold else 'FAIL'}: "
              f"{'Good' if variance_ratio >= otsu_variance_threshold else 'Poor'} bimodal separation")
        print()
        print(f"Illumination CV:         {illumination_cv:.4f} (threshold: {illumination_cv_threshold})")
        print(f"  → {'PASS' if illumination_cv <= illumination_cv_threshold else 'FAIL'}: "
              f"{'Uniform' if illumination_cv <= illumination_cv_threshold else 'Non-uniform'} lighting")
        print()
        print("-" * 60)
        
    if use_otsu:
        if verbose:
            print("✓ DECISION: Using OTSU thresholding")
            print(f"  Both conditions met - good separation & uniform lighting")
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        method_used = "Otsu"
    else:
        if verbose:
            print("✓ DECISION: Using ADAPTIVE thresholding")
            reasons = []
            if variance_ratio < otsu_variance_threshold:
                reasons.append("poor bimodal separation")
            if illumination_cv > illumination_cv_threshold:
                reasons.append("non-uniform lighting")
            print(f"  Reason(s): {', '.join(reasons)}")
        
        binary = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 
                                       adaptive_block_size, 
                                       adaptive_c)
        method_used = "Adaptive"
    
    if verbose:
        print("=" * 60)
        print()
    
    return binary, method_used, metrics


# Example usage and visualization
if __name__ == "__main__":
    # Load image
    image = cv2.imread('./data/samples/prueba sesil 2.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('./data/samples/gota depositada 1.png', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image. Using synthetic example.")
        # Create a synthetic droplet image for demonstration
        image = np.ones((400, 400), dtype=np.uint8) * 200
        cv2.circle(image, (200, 200), 100, 50, -1)
    
    # Apply automatic selection
    binary, method, metrics = auto_select_threshold_method(
        image,
        otsu_variance_threshold=0.5,
        illumination_cv_threshold=0.15,
        block_size=50,
        adaptive_block_size=11,
        adaptive_c=2,
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

    # Display results
    result_display = np.hstack([image, binary, outsubinary[1], adaptivebinary])
    
    # Add text to show which method was used
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_display, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result_display, f'{method} Method', (image.shape[1] + 10, 30), 
                font, 1, (255, 255, 255), 2)
    
    cv2.imshow('Automatic Threshold Selection', result_display)
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # You can also access the metrics
    print("\nMetrics Dictionary:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")