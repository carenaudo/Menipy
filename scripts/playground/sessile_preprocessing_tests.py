"""Sessile Preprocessing Tests.

Test module."""


import cv2
import numpy as np
from scipy import stats

class ImageQualityAnalyzer:
    """Analyze image quality and recommend preprocessing steps for ADSA."""
    
    def __init__(self, image):
        """Initialize with grayscale image."""
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image.copy()
        
        self.h, self.w = self.gray.shape
        self.preprocessing_needed = []
        self.metrics = {}
    
    def analyze_noise(self):
        """
        Detect noise level in the image.
        High noise requires Gaussian blur or median filtering.
        """
        # Method 1: Standard deviation in homogeneous regions
        # Sample multiple small patches and calculate variance
        patch_size = 20
        variances = []
        
        for i in range(10):  # Sample 10 random patches
            y = np.random.randint(0, self.h - patch_size)
            x = np.random.randint(0, self.w - patch_size)
            patch = self.gray[y:y+patch_size, x:x+patch_size]
            variances.append(np.var(patch))
        
        noise_estimate = np.median(variances)
        
        # Method 2: Laplacian variance (edge sharpness vs noise)
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        self.metrics['noise_estimate'] = noise_estimate
        self.metrics['laplacian_variance'] = laplacian_var
        
        # Decision thresholds
        if noise_estimate > 100:  # High noise
            self.preprocessing_needed.append({
                'step': 'Gaussian Blur or Median Filter',
                'reason': f'High noise detected (variance={noise_estimate:.1f})',
                'severity': 'HIGH',
                'recommended_params': 'cv2.GaussianBlur(image, (5,5), 0) or cv2.medianBlur(image, 5)'
            })
            return 'HIGH'
        elif noise_estimate > 50:  # Moderate noise
            self.preprocessing_needed.append({
                'step': 'Light Gaussian Blur',
                'reason': f'Moderate noise detected (variance={noise_estimate:.1f})',
                'severity': 'MEDIUM',
                'recommended_params': 'cv2.GaussianBlur(image, (3,3), 0)'
            })
            return 'MEDIUM'
        
        return 'LOW'
    
    def analyze_contrast(self):
        """
        Detect low contrast that might need histogram equalization or CLAHE.
        """
        # Calculate histogram
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Metrics for contrast
        mean_intensity = np.mean(self.gray)
        std_intensity = np.std(self.gray)
        intensity_range = np.max(self.gray) - np.min(self.gray)
        
        # Calculate effective dynamic range (5th to 95th percentile)
        p5 = np.percentile(self.gray, 5)
        p95 = np.percentile(self.gray, 95)
        effective_range = p95 - p5
        
        self.metrics['mean_intensity'] = mean_intensity
        self.metrics['std_intensity'] = std_intensity
        self.metrics['intensity_range'] = intensity_range
        self.metrics['effective_range'] = effective_range
        
        # Low contrast detection
        if effective_range < 80:  # Very low contrast
            self.preprocessing_needed.append({
                'step': 'CLAHE (Contrast Limited Adaptive Histogram Equalization)',
                'reason': f'Very low contrast (effective range={effective_range:.1f})',
                'severity': 'HIGH',
                'recommended_params': 'clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); clahe.apply(image)'
            })
            return 'LOW'
        elif effective_range < 120:  # Moderate contrast
            self.preprocessing_needed.append({
                'step': 'Light CLAHE or Histogram Equalization',
                'reason': f'Moderate contrast (effective range={effective_range:.1f})',
                'severity': 'MEDIUM',
                'recommended_params': 'clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)); clahe.apply(image)'
            })
            return 'MODERATE'
        
        return 'GOOD'
    
    def analyze_brightness(self):
        """
        Detect if image is too dark or too bright.
        """
        mean_intensity = np.mean(self.gray)
        
        self.metrics['brightness_mean'] = mean_intensity
        
        if mean_intensity < 60:  # Too dark
            self.preprocessing_needed.append({
                'step': 'Brightness Correction',
                'reason': f'Image too dark (mean={mean_intensity:.1f})',
                'severity': 'MEDIUM',
                'recommended_params': f'cv2.convertScaleAbs(image, alpha=1.0, beta={int(127-mean_intensity)})'
            })
            return 'DARK'
        elif mean_intensity > 195:  # Too bright
            self.preprocessing_needed.append({
                'step': 'Brightness Correction',
                'reason': f'Image too bright (mean={mean_intensity:.1f})',
                'severity': 'MEDIUM',
                'recommended_params': f'cv2.convertScaleAbs(image, alpha=1.0, beta={int(127-mean_intensity)})'
            })
            return 'BRIGHT'
        
        return 'GOOD'
    
    def analyze_sharpness(self):
        """
        Detect blur using Laplacian variance.
        Blurry images might need sharpening (but careful with ADSA).
        """
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        self.metrics['sharpness'] = sharpness
        
        if sharpness < 100:  # Very blurry
            self.preprocessing_needed.append({
                'step': 'Sharpening (use with caution)',
                'reason': f'Image appears blurry (sharpness={sharpness:.1f})',
                'severity': 'LOW',
                'recommended_params': 'kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]); cv2.filter2D(image, -1, kernel)'
            })
            return 'BLURRY'
        
        return 'SHARP'
    
    def analyze_illumination_uniformity(self, block_size=50):
        """
        Detect non-uniform illumination (vignetting, shadows, gradients).
        """
        local_means = []
        
        for i in range(0, self.h - block_size + 1, block_size):
            for j in range(0, self.w - block_size + 1, block_size):
                block = self.gray[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    local_means.append(np.mean(block))
        
        cv = np.std(local_means) / np.mean(local_means) if np.mean(local_means) > 0 else 0
        
        self.metrics['illumination_cv'] = cv
        
        if cv > 0.25:  # Severe non-uniformity
            self.preprocessing_needed.append({
                'step': 'Illumination Correction',
                'reason': f'Severe non-uniform illumination (CV={cv:.3f})',
                'severity': 'HIGH',
                'recommended_params': 'Use morphological tophat or estimate background with Gaussian blur'
            })
            return 'NON_UNIFORM'
        elif cv > 0.15:  # Moderate non-uniformity
            self.preprocessing_needed.append({
                'step': 'Light Illumination Correction',
                'reason': f'Moderate non-uniform illumination (CV={cv:.3f})',
                'severity': 'MEDIUM',
                'recommended_params': 'Consider background subtraction or adaptive thresholding'
            })
            return 'MODERATE'
        
        return 'UNIFORM'
    
    def detect_artifacts(self):
        """
        Detect common artifacts: hot pixels, dead pixels, scratches, dust.
        """
        artifacts = []
        
        # Hot pixels (very bright isolated pixels)
        _, hot_pixels = cv2.threshold(self.gray, 250, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        hot_pixels_eroded = cv2.erode(hot_pixels, kernel, iterations=1)
        num_hot_pixels = np.sum(hot_pixels_eroded > 0)
        
        # Dead pixels (very dark isolated pixels)
        _, dead_pixels = cv2.threshold(self.gray, 5, 255, cv2.THRESH_BINARY_INV)
        dead_pixels_eroded = cv2.erode(dead_pixels, kernel, iterations=1)
        num_dead_pixels = np.sum(dead_pixels_eroded > 0)
        
        self.metrics['hot_pixels'] = num_hot_pixels
        self.metrics['dead_pixels'] = num_dead_pixels
        
        total_artifacts = num_hot_pixels + num_dead_pixels
        
        if total_artifacts > 50:  # Significant artifacts
            self.preprocessing_needed.append({
                'step': 'Median Filter',
                'reason': f'Detected {total_artifacts} artifact pixels (hot/dead pixels)',
                'severity': 'MEDIUM',
                'recommended_params': 'cv2.medianBlur(image, 3)'
            })
            return 'HIGH'
        
        return 'LOW'
    
    def analyze_background_complexity(self):
        """
        Detect if background is complex (multiple objects, textures).
        Important for ADSA to ensure clean droplet isolation.
        """
        # Edge density
        edges = cv2.Canny(self.gray, 50, 150)
        edge_density = np.sum(edges > 0) / (self.h * self.w)
        
        self.metrics['edge_density'] = edge_density
        
        if edge_density > 0.15:  # Very busy image
            self.preprocessing_needed.append({
                'step': 'Region of Interest (ROI) Selection',
                'reason': f'Complex background detected (edge density={edge_density:.3f})',
                'severity': 'HIGH',
                'recommended_params': 'Crop to droplet region or use morphological operations to isolate droplet'
            })
            return 'COMPLEX'
        
        return 'SIMPLE'
    
    def detect_reflections_glare(self):
        """
        Detect specular reflections or glare on droplet surface.
        """
        # Very bright regions (top 1% of pixels)
        threshold = np.percentile(self.gray, 99)
        bright_regions = self.gray > threshold
        
        # Count connected bright regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright_regions.astype(np.uint8), connectivity=8
        )
        
        # Filter small bright spots (likely reflections)
        reflection_count = 0
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if 5 < area < 500:  # Typical reflection size
                reflection_count += 1
        
        self.metrics['reflection_count'] = reflection_count
        
        if reflection_count > 3:
            self.preprocessing_needed.append({
                'step': 'Reflection Removal',
                'reason': f'Detected {reflection_count} potential reflections/glare spots',
                'severity': 'MEDIUM',
                'recommended_params': 'cv2.inpaint() or morphological closing to fill bright spots'
            })
            return 'HIGH'
        
        return 'LOW'
    
    def analyze_all(self):
        """
        Run all analyses and generate comprehensive report.
        """
        print("=" * 80)
        print("IMAGE QUALITY ANALYSIS FOR ADSA PREPROCESSING")
        print("=" * 80)
        print()
        
        # Run all analyses
        noise = self.analyze_noise()
        contrast = self.analyze_contrast()
        brightness = self.analyze_brightness()
        sharpness = self.analyze_sharpness()
        illumination = self.analyze_illumination_uniformity()
        artifacts = self.detect_artifacts()
        background = self.analyze_background_complexity()
        reflections = self.detect_reflections_glare()
        
        # Print metrics
        print("IMAGE METRICS:")
        print("-" * 80)
        for key, value in self.metrics.items():
            print(f"  {key:.<40} {value:.2f}")
        print()
        
        # Print preprocessing recommendations
        if len(self.preprocessing_needed) == 0:
            print("✓ IMAGE QUALITY: EXCELLENT")
            print("  No preprocessing needed - proceed directly to contour detection")
            print()
        else:
            print(f"⚠ PREPROCESSING RECOMMENDED: {len(self.preprocessing_needed)} issue(s) detected")
            print()
            
            # Sort by severity
            severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            sorted_steps = sorted(self.preprocessing_needed, 
                                key=lambda x: severity_order[x['severity']])
            
            for i, step in enumerate(sorted_steps, 1):
                severity_symbol = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }
                
                print(f"{i}. [{step['severity']}] {step['step']}")
                print(f"   Reason: {step['reason']}")
                print(f"   Code: {step['recommended_params']}")
                print()
        
        print("=" * 80)
        print()
        
        return self.preprocessing_needed, self.metrics
    
    def apply_recommended_preprocessing(self):
        """
        Automatically apply recommended preprocessing steps.
        Returns preprocessed image.
        """
        processed = self.gray.copy()
        
        print("Applying preprocessing steps...")
        print("-" * 80)
        
        # Sort by severity and apply in order
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_steps = sorted(self.preprocessing_needed, 
                            key=lambda x: severity_order[x['severity']])
        
        for step in sorted_steps:
            step_name = step['step']
            
            # Apply based on step type
            if 'Median Filter' in step_name:
                print(f"✓ Applying: {step_name}")
                processed = cv2.medianBlur(processed, 5)
            
            elif 'Gaussian Blur' in step_name:
                print(f"✓ Applying: {step_name}")
                if 'Light' in step_name:
                    processed = cv2.GaussianBlur(processed, (3, 3), 0)
                else:
                    processed = cv2.GaussianBlur(processed, (5, 5), 0)
            
            elif 'CLAHE' in step_name:
                print(f"✓ Applying: {step_name}")
                if step['severity'] == 'HIGH':
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                else:
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
            
            elif 'Brightness Correction' in step_name:
                print(f"✓ Applying: {step_name}")
                mean_val = np.mean(processed)
                beta = int(127 - mean_val)
                processed = cv2.convertScaleAbs(processed, alpha=1.0, beta=beta)
            
            elif 'Illumination Correction' in step_name:
                print(f"✓ Applying: {step_name}")
                # Estimate background with large Gaussian blur
                background = cv2.GaussianBlur(processed, (51, 51), 0)
                processed = cv2.subtract(processed, background)
                processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
            
            else:
                print(f"⚠ Skipping (manual intervention needed): {step_name}")
        
        print("-" * 80)
        print()
        
        return processed


def visualize_preprocessing(original, processed, analyzer):
    """
    Create side-by-side comparison with metrics overlay.
    """
    h, w = original.shape
    
    # Create comparison image
    comparison = np.zeros((h, w * 2), dtype=np.uint8)
    comparison[:, 0:w] = original
    comparison[:, w:2*w] = processed
    
    # Add labels and key metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Preprocessed', (w + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Add key metrics
    y_offset = 60
    key_metrics = ['noise_estimate', 'effective_range', 'illumination_cv']
    for metric in key_metrics:
        if metric in analyzer.metrics:
            text = f"{metric}: {analyzer.metrics[metric]:.2f}"
            cv2.putText(comparison, text, (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    return comparison


# Main usage example
if __name__ == "__main__":
    # Load image
    #image = cv2.imread('./data/samples/prueba sesil 2.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('./data/samples/gota depositada 1.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('./data/samples/gota pendiente 1.png', cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread('./data/samples/prueba pend 1.png', cv2.IMREAD_GRAYSCALE)
    image = None
    if image is None:
        print("Creating synthetic test image...")
        # Create synthetic droplet with some issues
        image = np.ones((400, 400), dtype=np.uint8) * 120  # Low contrast
        cv2.circle(image, (200, 200), 100, 80, -1)  # Droplet
        
        # Add noise
        noise = np.random.normal(0, 15, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Add vignetting (non-uniform illumination)
        rows, cols = image.shape
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        image = (image * mask).astype(np.uint8)
    
    # Analyze image
    analyzer = ImageQualityAnalyzer(image)
    preprocessing_steps, metrics = analyzer.analyze_all()
    
    # Apply preprocessing if needed
    if len(preprocessing_steps) > 0:
        print("APPLYING AUTOMATIC PREPROCESSING")
        print("=" * 80)
        processed = analyzer.apply_recommended_preprocessing()
        
        # Visualize
        comparison = visualize_preprocessing(image, processed, analyzer)
        cv2.imshow('Preprocessing Comparison', comparison)
        
        print("✓ Preprocessing complete!")
        print("  Now ready for contour detection")
        print()
    else:
        print("✓ No preprocessing needed - image quality is good!")
        processed = image
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()