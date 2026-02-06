"""Sessile Calculations.

Experimental implementation."""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calculate_contact_angle(contour_points, contact_point, substrate_y, side='left', fit_points=20):
    """
    Calculate contact angle using polynomial fitting near the contact point.
    
    Args:
        contour_points: Array of contour points
        contact_point: [x, y] coordinates of contact point
        substrate_y: Y-coordinate of substrate line
        side: 'left' or 'right' contact point
        fit_points: Number of points to use for tangent fitting
    
    Returns:
        angle: Contact angle in degrees
    """
    cp_x, cp_y = contact_point
    
    # Get points near the contact point for fitting
    # Sort by distance from contact point
    distances = np.sqrt((contour_points[:, 0] - cp_x)**2 + 
                       (contour_points[:, 1] - cp_y)**2)
    
    # Get nearby points above the substrate
    nearby_mask = (distances < 50) & (contour_points[:, 1] < substrate_y - 3)
    nearby_points = contour_points[nearby_mask]
    
    if len(nearby_points) < 5:
        return None
    
    # Sort by x-coordinate
    nearby_points = nearby_points[np.argsort(nearby_points[:, 0])]
    
    # Select appropriate points based on side
    if side == 'left':
        # Take points to the right of contact point
        fit_pts = nearby_points[nearby_points[:, 0] >= cp_x][:fit_points]
    else:  # right side
        # Take points to the left of contact point
        fit_pts = nearby_points[nearby_points[:, 0] <= cp_x][-fit_points:]
    
    if len(fit_pts) < 5:
        return None
    
    # Fit a 2nd order polynomial
    try:
        coeffs = np.polyfit(fit_pts[:, 0], fit_pts[:, 1], 2)
        # Get derivative at contact point: dy/dx = 2*a*x + b
        slope = 2 * coeffs[0] * cp_x + coeffs[1]
        
        # Calculate angle with horizontal (substrate)
        angle_rad = np.arctan(abs(slope))
        angle_deg = np.degrees(angle_rad)
        
        # For sessile drops, we want the angle measured through the liquid
        # If slope is positive on left or negative on right, adjust
        if side == 'left' and slope > 0:
            angle_deg = 180 - angle_deg
        elif side == 'right' and slope < 0:
            angle_deg = 180 - angle_deg
            
        return angle_deg
    except:
        return None

def calculate_drop_metrics(contour, contact_left, contact_right, substrate_y):
    """
    Calculate additional drop metrics.
    
    Returns:
        dict: Dictionary containing various measurements
    """
    points = contour[:, 0, :]
    
    # Base width (contact line length)
    base_width = abs(contact_right[0] - contact_left[0])
    
    # Drop height (highest point to substrate)
    highest_point = points[np.argmin(points[:, 1])]
    drop_height = substrate_y - highest_point[1]
    
    # Contact area (2D projection)
    area = cv2.contourArea(contour)
    
    # Aspect ratio
    aspect_ratio = drop_height / base_width if base_width > 0 else 0
    
    # Perimeter
    perimeter = cv2.arcLength(contour, closed=True)
    
    # Volume estimation (assuming rotational symmetry)
    # Using disk integration method
    volume = 0
    y_coords = sorted(set(points[:, 1]))
    for y in y_coords:
        if y >= substrate_y:
            continue
        row_points = points[points[:, 1] == y]
        if len(row_points) >= 2:
            width = max(row_points[:, 0]) - min(row_points[:, 0])
            radius = width / 2
            # Volume of thin disk: π * r² * dy
            volume += np.pi * radius**2
    
    return {
        'base_width': base_width,
        'height': drop_height,
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'volume_estimate': volume,
        'apex_position': highest_point
    }

def sessile_drop_adaptive(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    output_img = img.copy()
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- STEP 1: CONTRAST ENHANCEMENT (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # --- STEP 2: ROBUST SUBSTRATE DETECTION ---
    margin_px = min(50, width // 10)
    left_strip = enhanced_gray[:, 0:margin_px]
    right_strip = enhanced_gray[:, width-margin_px:width]

    def find_horizon_median(strip_gray):
    """Find horizon median.

    Parameters
    ----------
    strip_gray : type
        Description.

    Returns
    -------
    type
        Description.
    """
        detected_ys = []
        h, w = strip_gray.shape
        min_limit, max_limit = int(h * 0.05), int(h * 0.95)
        for col in range(w):
            col_data = strip_gray[:, col].astype(float)
            grad = np.diff(col_data)
            valid_grad = grad[min_limit:max_limit]
            if len(valid_grad) == 0: continue
            best_y = np.argmin(valid_grad) + min_limit
            detected_ys.append(best_y)
        if not detected_ys: return None
        return int(np.median(detected_ys))

    y_left = find_horizon_median(left_strip)
    y_right = find_horizon_median(right_strip)
    
    if y_left is None or y_right is None:
        print("Error: Substrate line lost in low contrast.")
        substrate_y = int(height * 0.8)
    else:
        substrate_y = int((y_left + y_right) / 2)

    # --- STEP 3: ADAPTIVE SEGMENTATION ---
    blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 2)

    # CLEANUP:
    binary[substrate_y-2:, :] = 0
    kernel = np.ones((3,3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- STEP 4: CONTOUR & HULL ---
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found.")
        return

    # Filter Contours
    valid_contours = []
    center_x = width // 2
    needle_cnt = None
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        cnt_center_x = x + w//2 
        cnt_center_y = y + h//2
        max_y = y + h
        min_y = y
        min_x = x
        max_x = x + w
        
        if y < 5:
            if needle_cnt is None: 
                needle_cnt = cnt
                needle_y = cnt_center_y
                needle_x = max_x
        else:
            if area > (width*height)*0.005 and x > 5 and (x+w) < (width-5):
                valid_contours.append(cnt)
    
    if not valid_contours:
        print("Error: No valid drop contours found.")
        drop_cnt = max(contours, key=cv2.contourArea)
    else:
        drop_cnt = max(valid_contours, key=cv2.contourArea)

    # Apply Convex Hull
    hull = cv2.convexHull(drop_cnt)

    # --- STEP 5: RECONSTRUCT FLAT BASE ---
    points = hull[:, 0, :]
    dome_points = [pt for pt in points if pt[1] < (substrate_y - 5)]
    
    if not dome_points:
        print("Error: Hull collapsed.")
        return
        
    dome_points = sorted(dome_points, key=lambda p: p[0])
    x_left = dome_points[0][0]
    x_right = dome_points[-1][0]
    
    cp_left = [x_left, substrate_y]
    cp_right = [x_right, substrate_y]
    
    final_polygon = np.array([cp_left] + dome_points + [cp_right], dtype=np.int32)
    final_cnt = final_polygon.reshape((-1, 1, 2))
    
    x, y, w, h = cv2.boundingRect(final_cnt)
    drop_y = y
    drop_x = x + w//2

    # --- STEP 6: CALCULATE CONTACT ANGLES ---
    dome_points_array = np.array(dome_points)
    
    angle_left = calculate_contact_angle(dome_points_array, cp_left, substrate_y, 'left')
    angle_right = calculate_contact_angle(dome_points_array, cp_right, substrate_y, 'right')
    
    # Calculate additional metrics
    metrics = calculate_drop_metrics(final_cnt, cp_left, cp_right, substrate_y)
    
    # --- STEP 7: VISUALIZATION ---
    cv2.putText(output_img, "Substrate Baseline", (10, substrate_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.line(output_img, (0, substrate_y), (width, substrate_y), (255, 0, 255), 2)
    
    overlay = output_img.copy()
    cv2.drawContours(overlay, [final_cnt], -1, (0, 255, 0), -1)
    cv2.putText(output_img, "Drop", (drop_x - 5, drop_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 40, 0), 1)
    
    if needle_cnt is not None:
        cv2.drawContours(overlay, [needle_cnt], -1, (0, 0, 255), -1)
        cv2.putText(output_img, "Needle", (needle_x + 5, needle_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)
    cv2.drawContours(output_img, [final_cnt], -1, (0, 255, 0), 2)
    
    # Draw contact points
    cv2.circle(output_img, tuple(cp_left), 5, (0, 0, 255), -1)
    cv2.circle(output_img, tuple(cp_right), 5, (0, 0, 255), -1)
    
    # Draw contact angle labels
    if angle_left is not None:
        cv2.putText(output_img, f"L: {angle_left:.1f}°", 
                   (cp_left[0] - 60, cp_left[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if angle_right is not None:
        cv2.putText(output_img, f"R: {angle_right:.1f}°", 
                   (cp_right[0] + 10, cp_right[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ROI
    pad = 20
    roi_coords = (max(0, x_left - pad), 
                  max(0, min([p[1] for p in dome_points]) - pad), 
                  min(width, x_right + pad), 
                  min(height, substrate_y + pad))
 
    final_roi = img[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    cv2.rectangle(output_img, (roi_coords[0], roi_coords[1]), 
                  (roi_coords[2], roi_coords[3]), (0, 255, 255), 2)
    cv2.putText(output_img, "ROI", (roi_coords[0] + 5, roi_coords[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 15, 15), 1)
    
    # --- PRINT RESULTS ---
    print(f"\n{'='*50}")
    print(f"Analysis Results for: {image_path}")
    print(f"{'='*50}")
    if angle_left is not None:
        print(f"Left Contact Angle:  {angle_left:.2f}°")
    else:
        print(f"Left Contact Angle:  Unable to calculate")
    if angle_right is not None:
        print(f"Right Contact Angle: {angle_right:.2f}°")
    else:
        print(f"Right Contact Angle: Unable to calculate")
    
    if angle_left and angle_right:
        avg_angle = (angle_left + angle_right) / 2
        print(f"Average Angle:       {avg_angle:.2f}°")
    
    print(f"\nDrop Metrics:")
    print(f"  Base Width:        {metrics['base_width']:.2f} px")
    print(f"  Height:            {metrics['height']:.2f} px")
    print(f"  Aspect Ratio:      {metrics['aspect_ratio']:.3f}")
    print(f"  Contact Area:      {metrics['area']:.2f} px²")
    print(f"  Perimeter:         {metrics['perimeter']:.2f} px")
    print(f"  Volume (estimate): {metrics['volume_estimate']:.2f} px³")
    print(f"{'='*50}\n")
    
    # --- Plot Comparison ---
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 3, 1)
    plt.title("Step 1: Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Step 2: Contrast Enhanced (CLAHE)")
    plt.imshow(enhanced_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Step 3: Binary Segmentation")
    plt.imshow(binary_clean, cmap='gray')
    plt.axis('off')

    if final_roi is not None:
        plt.subplot(2, 3, 4)
        plt.title("ROI (For Calculations)")
        plt.imshow(cv2.cvtColor(final_roi, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Step 4: Final Detection")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Results summary plot
    plt.subplot(2, 3, 6)
    plt.axis('off')
    results_text = f"CONTACT ANGLES\n"
    if angle_left:
        results_text += f"Left:  {angle_left:.2f}°\n"
    if angle_right:
        results_text += f"Right: {angle_right:.2f}°\n"
    if angle_left and angle_right:
        results_text += f"Avg:   {(angle_left+angle_right)/2:.2f}°\n"
    results_text += f"\nMETRICS\n"
    results_text += f"Width: {metrics['base_width']:.1f} px\n"
    results_text += f"Height: {metrics['height']:.1f} px\n"
    results_text += f"Ratio: {metrics['aspect_ratio']:.3f}\n"
    results_text += f"Area: {metrics['area']:.1f} px²"
    
    plt.text(0.1, 0.5, results_text, fontsize=11, family='monospace',
             verticalalignment='center')
    plt.title("Measurements")

    plt.tight_layout()
    plt.show()

# Run
sessile_drop_adaptive("./data/samples/prueba sesil 2.png")
sessile_drop_adaptive("./data/samples/gota depositada 1.png")
sessile_drop_adaptive("./data/samples/gota pendiente 1.png")