"""
Droplet Shape Analysis using scikit-image Active Contours
This is a simpler implementation using scikit-image's built-in active_contour
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.segmentation import active_contour
from skimage.filters import gaussian


def load_or_create_droplet():
    """Create a synthetic droplet image"""
    img = np.zeros((500, 500), dtype=np.uint8)
    
    # Create multiple droplets with different shapes
    # Droplet 1: Circular
    cv2.circle(img, (150, 150), 60, 255, -1)
    
    # Droplet 2: Elliptical
    cv2.ellipse(img, (350, 150), (50, 70), 0, 0, 360, 255, -1)
    
    # Droplet 3: Slightly deformed
    cv2.ellipse(img, (150, 350), (65, 55), 30, 0, 360, 255, -1)
    
    # Droplet 4: Pear-shaped (using polygon)
    pts = np.array([[350, 300], [380, 320], [390, 360], [370, 400], 
                    [330, 400], [310, 360], [320, 320]], np.int32)
    cv2.fillPoly(img, [pts], 255)
    
    # Add noise and blur
    noise = np.random.normal(0, 15, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img


def detect_droplets_simple(image, min_area=1000):
    """Simple droplet detection using thresholding and contours"""
    # Threshold
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    droplet_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            droplet_contours.append(cnt)
    
    return droplet_contours


def init_snake_from_contour(contour, expand=1.2):
    """Initialize snake from detected contour"""
    # Get center
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = contour[0][0]
    
    # Expand contour outward
    contour = contour.reshape(-1, 2)
    center = np.array([cx, cy])
    expanded = center + expand * (contour - center)
    
    return expanded


def analyze_droplet_comprehensive(snake_contour, image_shape):
    """Comprehensive droplet analysis"""
    # Convert to OpenCV format
    contour_cv = snake_contour.astype(np.float32).reshape(-1, 1, 2)
    
    # Basic measurements
    area = cv2.contourArea(contour_cv)
    perimeter = cv2.arcLength(contour_cv, True)
    
    # Fit ellipse
    if len(snake_contour) >= 5:
        ellipse = cv2.fitEllipse(contour_cv)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        
        # Shape descriptors
        circularity = 4 * np.pi * area / (perimeter ** 2)
        aspect_ratio = major_axis / minor_axis
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        
        # Solidity (convexity)
        hull = cv2.convexHull(contour_cv)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Equivalent diameter
        equiv_diameter = np.sqrt(4 * area / np.pi)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour_cv)
        extent = area / (w * h) if w * h > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'center': center,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'equiv_diameter': equiv_diameter,
            'extent': extent,
            'ellipse': ellipse,
            'convex_hull': hull
        }
    
    return None


def process_image_with_active_contours(image_path=None):
    """Main processing pipeline"""
    # Load or create image
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = load_or_create_droplet()
    
    # Smooth image for better edge detection
    img_smooth = gaussian(img / 255.0, sigma=2)
    
    # Detect droplets using simple method
    initial_contours = detect_droplets_simple(img, min_area=1000)
    
    print(f"Found {len(initial_contours)} droplet(s)")
    
    # Process each droplet
    results = []
    
    fig, axes = plt.subplots(2, len(initial_contours), 
                             figsize=(5 * len(initial_contours), 10))
    if len(initial_contours) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, contour in enumerate(initial_contours):
        print(f"\nProcessing droplet {idx + 1}...")
        
        # Initialize snake from contour
        init_snake = init_snake_from_contour(contour, expand=1.15)
        
        # Apply active contour (snake)
        snake = active_contour(
            img_smooth,
            init_snake,
            alpha=0.015,      # Length weight
            beta=10,          # Smoothness weight
            gamma=0.001,      # Time step
            max_num_iter=500,
            convergence=0.1
        )
        
        # Analyze
        analysis = analyze_droplet_comprehensive(snake, img.shape)
        
        if analysis:
            results.append(analysis)
            
            # Visualization - Initial
            axes[0, idx].imshow(img, cmap='gray')
            axes[0, idx].plot(init_snake[:, 0], init_snake[:, 1], 
                            'r-', linewidth=2, label='Initial')
            axes[0, idx].set_title(f'Droplet {idx + 1} - Initial')
            axes[0, idx].legend()
            axes[0, idx].axis('off')
            
            # Visualization - Final with analysis
            axes[1, idx].imshow(img, cmap='gray')
            axes[1, idx].plot(snake[:, 0], snake[:, 1], 
                            'g-', linewidth=2, label='Snake')
            
            # Draw fitted ellipse
            ellipse_img = np.zeros_like(img)
            center = tuple(map(int, analysis['ellipse'][0]))
            axes_len = tuple(map(int, [x/2 for x in analysis['ellipse'][1]]))
            angle = int(analysis['ellipse'][2])
            cv2.ellipse(ellipse_img, center, axes_len, angle, 0, 360, 255, 2)
            axes[1, idx].contour(ellipse_img, colors='cyan', linewidths=1.5)
            
            # Draw convex hull
            hull = analysis['convex_hull'].reshape(-1, 2)
            axes[1, idx].plot(hull[:, 0], hull[:, 1], 'y--', 
                            linewidth=1.5, alpha=0.6, label='Convex Hull')
            
            # Add measurements text
            text = f"Area: {analysis['area']:.0f} px²\n"
            text += f"Circ: {analysis['circularity']:.3f}\n"
            text += f"AR: {analysis['aspect_ratio']:.2f}\n"
            text += f"Ecc: {analysis['eccentricity']:.3f}\n"
            text += f"Sol: {analysis['solidity']:.3f}"
            
            axes[1, idx].text(10, 30, text, fontsize=8, color='yellow',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            axes[1, idx].set_title(f'Droplet {idx + 1} - Analysis')
            axes[1, idx].legend(fontsize=8)
            axes[1, idx].axis('off')
            
            # Print detailed results
            print(f"  Area: {analysis['area']:.2f} px²")
            print(f"  Perimeter: {analysis['perimeter']:.2f} px")
            print(f"  Circularity: {analysis['circularity']:.4f}")
            print(f"  Aspect Ratio: {analysis['aspect_ratio']:.4f}")
            print(f"  Eccentricity: {analysis['eccentricity']:.4f}")
            print(f"  Solidity: {analysis['solidity']:.4f}")
            print(f"  Equivalent Diameter: {analysis['equiv_diameter']:.2f} px")
    
    plt.tight_layout()
    plt.savefig('/home/claude/droplet_analysis_multiple.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to droplet_analysis_multiple.png")
    
    return results


def classify_droplet_shape(analysis):
    """Classify droplet shape based on measurements"""
    if analysis is None:
        return "Unknown"
    
    circ = analysis['circularity']
    aspect = analysis['aspect_ratio']
    solidity = analysis['solidity']
    
    if circ > 0.85 and aspect < 1.2:
        return "Circular/Spherical"
    elif circ > 0.7 and 1.2 <= aspect < 1.5:
        return "Slightly Elliptical"
    elif aspect >= 1.5 and aspect < 2.5:
        return "Elliptical"
    elif aspect >= 2.5:
        return "Elongated"
    elif solidity < 0.85:
        return "Irregular/Deformed"
    else:
        return "Moderately Circular"


if __name__ == "__main__":
    print("=== Droplet Shape Analysis with Active Contours ===\n")
    
    # Process image
    results = process_image_with_active_contours()
    
    # Classify each droplet
    print("\n=== Shape Classification ===")
    for i, analysis in enumerate(results):
        shape_class = classify_droplet_shape(analysis)
        print(f"Droplet {i + 1}: {shape_class}")
    
    print("\n✓ Analysis complete!")
