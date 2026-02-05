"""
Droplet Shape Analysis using Active Contours (Snakes) with OpenCV
This script demonstrates contour detection and analysis for droplet shapes
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def create_synthetic_droplet(size=400):
    """Create a synthetic droplet image for demonstration"""
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Create an elliptical droplet shape
    center = (size // 2, size // 2)
    axes = (80, 100)  # (width, height)
    cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Apply slight blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img


def initialize_snake_circle(center, radius, num_points=100):
    """Initialize snake as a circle"""
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.array([x, y]).T


def compute_external_energy(image):
    """Compute external energy (edge-based) using gradient magnitude"""
    # Compute gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize and invert (we want edges to have low energy)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    external_energy = -gradient_magnitude
    
    # Apply Gaussian blur for smoother energy landscape
    external_energy = cv2.GaussianBlur(external_energy, (5, 5), 1)
    
    return external_energy


def active_contour_evolution(image, snake, alpha=0.01, beta=0.1, gamma=0.1, iterations=100):
    """
    Evolve active contour using greedy algorithm
    
    Parameters:
    - alpha: elasticity (prevents stretching)
    - beta: rigidity (prevents bending)
    - gamma: step size
    - iterations: number of iterations
    """
    # Compute external energy
    external_energy = compute_external_energy(image)
    
    # Create coordinate grids for interpolation
    h, w = image.shape
    
    for iteration in range(iterations):
        snake_prev = snake.copy()
        
        for i in range(len(snake)):
            # Get neighboring points
            prev_point = snake[(i - 1) % len(snake)]
            curr_point = snake[i]
            next_point = snake[(i + 1) % len(snake)]
            
            # Internal energy (continuity and curvature)
            # Continuity: prefers equal spacing
            continuity = prev_point + next_point - 2 * curr_point
            
            # Curvature: prefers smoothness
            curvature = prev_point - 2 * curr_point + next_point
            
            # Search in neighborhood for best position
            best_energy = float('inf')
            best_pos = curr_point
            
            search_range = 3
            for dx in range(-search_range, search_range + 1):
                for dy in range(-search_range, search_range + 1):
                    new_x = int(curr_point[0] + dx)
                    new_y = int(curr_point[1] + dy)
                    
                    # Check bounds
                    if 0 <= new_x < w and 0 <= new_y < h:
                        new_point = np.array([new_x, new_y])
                        
                        # Compute total energy
                        cont_energy = alpha * np.sum(continuity**2)
                        curv_energy = beta * np.sum(curvature**2)
                        ext_energy = external_energy[new_y, new_x]
                        
                        total_energy = cont_energy + curv_energy + ext_energy
                        
                        if total_energy < best_energy:
                            best_energy = total_energy
                            best_pos = new_point
            
            snake[i] = best_pos
        
        # Check convergence
        if np.allclose(snake, snake_prev, atol=0.1):
            print(f"Converged at iteration {iteration}")
            break
    
    return snake


def analyze_droplet(contour):
    """Analyze droplet properties from contour"""
    # Convert contour to the format expected by OpenCV
    contour_cv = contour.astype(np.float32).reshape(-1, 1, 2)
    
    # Fit ellipse
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour_cv)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        
        # Calculate properties
        area = cv2.contourArea(contour_cv)
        perimeter = cv2.arcLength(contour_cv, True)
        
        # Circularity (4π*area/perimeter²)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Aspect ratio
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        
        # Eccentricity
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        
        return {
            'center': center,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'eccentricity': eccentricity,
            'ellipse': ellipse
        }
    
    return None


def visualize_results(image, initial_snake, final_snake, analysis):
    """Visualize the snake evolution and analysis results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with initial snake
    axes[0].imshow(image, cmap='gray')
    axes[0].plot(initial_snake[:, 0], initial_snake[:, 1], 'r-', linewidth=2, label='Initial')
    axes[0].set_title('Initial Snake')
    axes[0].legend()
    axes[0].axis('off')
    
    # Image with final snake
    axes[1].imshow(image, cmap='gray')
    axes[1].plot(final_snake[:, 0], final_snake[:, 1], 'g-', linewidth=2, label='Final')
    axes[1].set_title('Final Snake (Active Contour)')
    axes[1].legend()
    axes[1].axis('off')
    
    # Analysis visualization
    axes[2].imshow(image, cmap='gray')
    axes[2].plot(final_snake[:, 0], final_snake[:, 1], 'g-', linewidth=2)
    
    if analysis:
        # Draw fitted ellipse
        ellipse = analysis['ellipse']
        center = tuple(map(int, ellipse[0]))
        axes_len = tuple(map(int, [x/2 for x in ellipse[1]]))
        angle = int(ellipse[2])
        cv2.ellipse(np.zeros_like(image), center, axes_len, angle, 0, 360, 255, 2)
        
        # Add text with measurements
        text_str = f"Area: {analysis['area']:.1f} px²\n"
        text_str += f"Perimeter: {analysis['perimeter']:.1f} px\n"
        text_str += f"Circularity: {analysis['circularity']:.3f}\n"
        text_str += f"Aspect Ratio: {analysis['aspect_ratio']:.3f}\n"
        text_str += f"Eccentricity: {analysis['eccentricity']:.3f}"
        
        axes[2].text(10, 30, text_str, fontsize=9, color='yellow', 
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    axes[2].set_title('Analysis Results')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('./droplet_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to droplet_analysis.png")


def main():
    # Create or load droplet image
    print("Creating synthetic droplet image...")
    image = create_synthetic_droplet(size=400)
    
    # Initialize snake (circular initialization around estimated droplet)
    print("Initializing snake contour...")
    center = (200, 200)
    radius = 120  # Start with larger radius
    initial_snake = initialize_snake_circle(center, radius, num_points=100)
    
    # Evolve active contour
    print("Evolving active contour...")
    final_snake = active_contour_evolution(
        image, 
        initial_snake.copy(),
        alpha=0.015,  # Elasticity
        beta=0.1,     # Rigidity
        gamma=0.1,    # Step size
        iterations=200
    )
    
    # Analyze droplet
    print("Analyzing droplet properties...")
    analysis = analyze_droplet(final_snake)
    
    if analysis:
        print("\n=== Droplet Analysis Results ===")
        print(f"Center: ({analysis['center'][0]:.1f}, {analysis['center'][1]:.1f})")
        print(f"Major Axis: {analysis['major_axis']:.2f} px")
        print(f"Minor Axis: {analysis['minor_axis']:.2f} px")
        print(f"Area: {analysis['area']:.2f} px²")
        print(f"Perimeter: {analysis['perimeter']:.2f} px")
        print(f"Circularity: {analysis['circularity']:.4f}")
        print(f"Aspect Ratio: {analysis['aspect_ratio']:.4f}")
        print(f"Eccentricity: {analysis['eccentricity']:.4f}")
        print(f"Orientation: {analysis['angle']:.2f}°")
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(image, initial_snake, final_snake, analysis)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
