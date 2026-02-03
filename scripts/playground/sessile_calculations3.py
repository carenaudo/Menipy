import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import odeint

def calculate_contact_angle_tangent(contour_points, contact_point, substrate_y, side='left', fit_points=20):
    """
    Calculate contact angle using polynomial fitting near the contact point (original method).
    """
    cp_x, cp_y = contact_point
    
    distances = np.sqrt((contour_points[:, 0] - cp_x)**2 + 
                       (contour_points[:, 1] - cp_y)**2)
    
    nearby_mask = (distances < 50) & (contour_points[:, 1] < substrate_y - 3)
    nearby_points = contour_points[nearby_mask]
    
    if len(nearby_points) < 5:
        return None
    
    nearby_points = nearby_points[np.argsort(nearby_points[:, 0])]
    
    if side == 'left':
        fit_pts = nearby_points[nearby_points[:, 0] >= cp_x][:fit_points]
    else:
        fit_pts = nearby_points[nearby_points[:, 0] <= cp_x][-fit_points:]
    
    if len(fit_pts) < 5:
        return None
    
    try:
        coeffs = np.polyfit(fit_pts[:, 0], fit_pts[:, 1], 2)
        slope = 2 * coeffs[0] * cp_x + coeffs[1]
        angle_rad = np.arctan(abs(slope))
        angle_deg = np.degrees(angle_rad)
        
        if side == 'left' and slope > 0:
            angle_deg = 180 - angle_deg
        elif side == 'right' and slope < 0:
            angle_deg = 180 - angle_deg
            
        return angle_deg
    except:
        return None

def fit_spherical_cap(contour_points, contact_left, contact_right, substrate_y):
    """
    Fit a spherical cap model to the drop profile.
    Returns: contact angle, radius of curvature, volume
    """
    points = contour_points.copy()
    
    # Center the drop
    base_center_x = (contact_left[0] + contact_right[0]) / 2
    base_width = abs(contact_right[0] - contact_left[0])
    
    # Get drop height
    min_y = np.min(points[:, 1])
    height = substrate_y - min_y
    
    if height <= 0 or base_width <= 0:
        return None, None, None
    
    # For a spherical cap: R² = (h/2)² + (b/2)²
    # where R is radius, h is height, b is base width
    R = (height**2 + (base_width/2)**2) / (2 * height)
    
    # Contact angle from geometry: sin(θ) = (b/2) / R
    sin_theta = (base_width / 2) / R
    if sin_theta > 1:
        sin_theta = 1
    theta_rad = np.arcsin(sin_theta)
    theta_deg = np.degrees(theta_rad)
    
    # Volume of spherical cap: V = πh²(3R - h)/3
    volume = np.pi * height**2 * (3*R - height) / 3
    
    return theta_deg, R, volume

def fit_elliptical(contour_points, contact_left, contact_right, substrate_y):
    """
    Fit an ellipse to the drop profile.
    Returns: contact angles (left, right), semi-axes, volume
    """
    points = contour_points.copy()
    
    # Fit ellipse using OpenCV
    if len(points) < 5:
        return None, None, None, None, None
    
    try:
        ellipse = cv2.fitEllipse(points)
        (center_x, center_y), (width, height), angle = ellipse
        
        # Semi-axes
        a = max(width, height) / 2  # semi-major axis
        b = min(width, height) / 2  # semi-minor axis
        
        # Contact points relative to ellipse center
        dx_left = contact_left[0] - center_x
        dx_right = contact_right[0] - center_x
        dy = substrate_y - center_y
        
        # Calculate slope at contact points using ellipse equation
        # For ellipse: x²/a² + y²/b² = 1
        # dy/dx = -(b²x)/(a²y)
        
        # Left contact angle
        if dy != 0:
            slope_left = -(b**2 * dx_left) / (a**2 * dy)
            angle_left = np.degrees(np.arctan(abs(slope_left)))
            if slope_left > 0:
                angle_left = 180 - angle_left
        else:
            angle_left = 90
        
        # Right contact angle
        if dy != 0:
            slope_right = -(b**2 * dx_right) / (a**2 * dy)
            angle_right = np.degrees(np.arctan(abs(slope_right)))
            if slope_right < 0:
                angle_right = 180 - angle_right
        else:
            angle_right = 90
        
        # Volume approximation (oblate ellipsoid)
        volume = (4/3) * np.pi * a * a * b
        
        return angle_left, angle_right, a, b, volume
    except:
        return None, None, None, None, None

def young_laplace_profile(z, y, b):
    """
    Simplified Young-Laplace differential equations (no gravity).
    y = [r, phi] where r is radius, phi is angle from vertical
    """
    r, phi = y
    
    if abs(r) < 1e-6:  # Near apex singularity
        dr_dz = np.sin(phi)
        dphi_dz = b
    else:
        dr_dz = np.sin(phi)
        dphi_dz = b - np.cos(phi) / r
    
    return [dr_dz, dphi_dz]

def fit_young_laplace(contour_points, contact_left, contact_right, substrate_y):
    """
    Fit Young-Laplace equation to drop profile using simplified model.
    This uses a gravity-free approximation suitable for small drops.
    
    Returns: contact angles, radius of curvature at apex
    """
    points = contour_points.copy()
    
    # Get drop dimensions
    base_center_x = (contact_left[0] + contact_right[0]) / 2
    apex_y = np.min(points[:, 1])
    height = substrate_y - apex_y
    base_width = abs(contact_right[0] - contact_left[0])
    
    if height <= 0 or base_width <= 0:
        return None, None, None
    
    # Normalize coordinates (apex at origin, scale by height)
    points_norm = points.copy().astype(float)
    points_norm[:, 0] = (points_norm[:, 0] - base_center_x) / height
    points_norm[:, 1] = (substrate_y - points_norm[:, 1]) / height
    
    # Filter points above substrate and sort
    valid_mask = points_norm[:, 1] > 0.02  # Avoid substrate noise
    points_norm = points_norm[valid_mask]
    
    if len(points_norm) < 10:
        return None, None, None
    
    # Sort by height (z-coordinate)
    idx = np.argsort(points_norm[:, 1])
    points_norm = points_norm[idx]
    
    # Take right half (assuming symmetry)
    right_mask = points_norm[:, 0] >= -0.05  # Small tolerance for center
    right_side = points_norm[right_mask]
    
    if len(right_side) < 10:
        return None, None, None
    
    # Remove duplicate z values by averaging
    z_unique = []
    r_unique = []
    current_z = right_side[0, 1]
    current_rs = [right_side[0, 0]]
    
    for i in range(1, len(right_side)):
        if abs(right_side[i, 1] - current_z) < 0.01:
            current_rs.append(right_side[i, 0])
        else:
            z_unique.append(current_z)
            r_unique.append(np.mean(current_rs))
            current_z = right_side[i, 1]
            current_rs = [right_side[i, 0]]
    
    z_unique.append(current_z)
    r_unique.append(np.mean(current_rs))
    
    z_data = np.array(z_unique)
    r_data = np.array(r_unique)
    
    if len(z_data) < 5:
        return None, None, None
    
    def fit_error(b_val):
        """Calculate fitting error for given curvature parameter"""
        try:
            # Initial conditions at apex
            y0 = [0.0, 0.0]  # r=0, phi=0 (vertical at apex)
            
            # Solve ODE
            sol = odeint(young_laplace_profile, y0, z_data, args=(b_val,))
            r_fit = sol[:, 0]
            
            # Calculate mean squared error
            error = np.mean((r_fit - r_data)**2)
            return error
        except:
            return 1e10
    
    # Grid search for best b parameter
    b_range = np.linspace(0.5, 5.0, 50)
    errors = [fit_error(b) for b in b_range]
    best_idx = np.argmin(errors)
    b_opt = b_range[best_idx]
    
    # Fine tune around best value
    b_fine = np.linspace(max(0.5, b_opt - 0.5), min(5.0, b_opt + 0.5), 30)
    errors_fine = [fit_error(b) for b in b_fine]
    best_idx_fine = np.argmin(errors_fine)
    b_final = b_fine[best_idx_fine]
    
    if errors_fine[best_idx_fine] > 0.1:  # Poor fit
        return None, None, None
    
    try:
        # Calculate final profile with optimized parameter
        y0 = [0.0, 0.0]
        sol = odeint(young_laplace_profile, y0, z_data, args=(b_final,))
        
        # Contact angle from final phi value
        phi_contact = sol[-1, 1]
        angle_deg = np.degrees(phi_contact)
        
        # Ensure angle is in valid range
        if angle_deg < 0:
            angle_deg = 0
        elif angle_deg > 180:
            angle_deg = 180
        
        # Radius of curvature at apex: R = 1/b (in normalized units)
        R_apex_norm = 1.0 / b_final if b_final > 0 else None
        R_apex_pixels = R_apex_norm * height if R_apex_norm else None
        
        # Assuming symmetric drop
        angle_left = angle_deg
        angle_right = angle_deg
        
        return angle_left, angle_right, R_apex_pixels
    except:
        return None, None, None

def calculate_drop_metrics(contour, contact_left, contact_right, substrate_y):
    """Calculate additional drop metrics."""
    points = contour[:, 0, :]
    
    base_width = abs(contact_right[0] - contact_left[0])
    highest_point = points[np.argmin(points[:, 1])]
    drop_height = substrate_y - highest_point[1]
    area = cv2.contourArea(contour)
    aspect_ratio = drop_height / base_width if base_width > 0 else 0
    perimeter = cv2.arcLength(contour, closed=True)
    
    # Volume estimation (disk integration)
    volume = 0
    y_coords = sorted(set(points[:, 1]))
    for y in y_coords:
        if y >= substrate_y:
            continue
        row_points = points[points[:, 1] == y]
        if len(row_points) >= 2:
            width = max(row_points[:, 0]) - min(row_points[:, 0])
            radius = width / 2
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

    binary[substrate_y-2:, :] = 0
    kernel = np.ones((3,3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- STEP 4: CONTOUR & HULL ---
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found.")
        return

    valid_contours = []
    center_x = width // 2
    needle_cnt = None
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        cnt_center_x = x + w//2 
        cnt_center_y = y + h//2
        
        if y < 5:
            if needle_cnt is None: 
                needle_cnt = cnt
                needle_y = cnt_center_y
                needle_x = x + w
        else:
            if area > (width*height)*0.005 and x > 5 and (x+w) < (width-5):
                valid_contours.append(cnt)
    
    if not valid_contours:
        print("Error: No valid drop contours found.")
        drop_cnt = max(contours, key=cv2.contourArea)
    else:
        drop_cnt = max(valid_contours, key=cv2.contourArea)

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

    # --- STEP 6: CALCULATE CONTACT ANGLES (ALL METHODS) ---
    dome_points_array = np.array(dome_points)
    
    # Method 1: Polynomial tangent (original)
    angle_left_tan = calculate_contact_angle_tangent(dome_points_array, cp_left, substrate_y, 'left')
    angle_right_tan = calculate_contact_angle_tangent(dome_points_array, cp_right, substrate_y, 'right')
    
    # Method 2: Spherical cap
    angle_sphere, R_sphere, vol_sphere = fit_spherical_cap(dome_points_array, cp_left, cp_right, substrate_y)
    
    # Method 3: Elliptical fit
    angle_left_ellipse, angle_right_ellipse, a_ellipse, b_ellipse, vol_ellipse = \
        fit_elliptical(dome_points_array, cp_left, cp_right, substrate_y)
    
    # Method 4: Young-Laplace
    angle_left_yl, angle_right_yl, R_apex = \
        fit_young_laplace(dome_points_array, cp_left, cp_right, substrate_y)
    
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
    
    cv2.circle(output_img, tuple(cp_left), 5, (0, 0, 255), -1)
    cv2.circle(output_img, tuple(cp_right), 5, (0, 0, 255), -1)
    
    # Display tangent method angles on image
    if angle_left_tan is not None:
        cv2.putText(output_img, f"{angle_left_tan:.1f}°", 
                   (cp_left[0] - 50, cp_left[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if angle_right_tan is not None:
        cv2.putText(output_img, f"{angle_right_tan:.1f}°", 
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
    print(f"\n{'='*60}")
    print(f"Analysis Results for: {image_path}")
    print(f"{'='*60}")
    
    print(f"\n--- METHOD 1: POLYNOMIAL TANGENT FIT ---")
    if angle_left_tan is not None:
        print(f"  Left Contact Angle:  {angle_left_tan:.2f}°")
    if angle_right_tan is not None:
        print(f"  Right Contact Angle: {angle_right_tan:.2f}°")
    if angle_left_tan and angle_right_tan:
        print(f"  Average Angle:       {(angle_left_tan + angle_right_tan)/2:.2f}°")
    
    print(f"\n--- METHOD 2: SPHERICAL CAP MODEL ---")
    if angle_sphere is not None:
        print(f"  Contact Angle:       {angle_sphere:.2f}° (symmetric)")
        print(f"  Radius of Curvature: {R_sphere:.2f} px")
        print(f"  Volume:              {vol_sphere:.2f} px³")
    else:
        print(f"  Unable to fit spherical cap model")
    
    print(f"\n--- METHOD 3: ELLIPTICAL FIT ---")
    if angle_left_ellipse is not None:
        print(f"  Left Contact Angle:  {angle_left_ellipse:.2f}°")
        print(f"  Right Contact Angle: {angle_right_ellipse:.2f}°")
        print(f"  Average Angle:       {(angle_left_ellipse + angle_right_ellipse)/2:.2f}°")
        print(f"  Semi-major axis:     {a_ellipse:.2f} px")
        print(f"  Semi-minor axis:     {b_ellipse:.2f} px")
        print(f"  Volume:              {vol_ellipse:.2f} px³")
    else:
        print(f"  Unable to fit ellipse")
    
    print(f"\n--- METHOD 4: YOUNG-LAPLACE EQUATION ---")
    if angle_left_yl is not None:
        print(f"  Left Contact Angle:  {angle_left_yl:.2f}°")
        print(f"  Right Contact Angle: {angle_right_yl:.2f}°")
        print(f"  Average Angle:       {(angle_left_yl + angle_right_yl)/2:.2f}°")
        print(f"  Apex Radius:         {R_apex:.2f} px")
    else:
        print(f"  Unable to fit Young-Laplace profile")
    
    print(f"\n--- GEOMETRIC METRICS ---")
    print(f"  Base Width:          {metrics['base_width']:.2f} px")
    print(f"  Height:              {metrics['height']:.2f} px")
    print(f"  Aspect Ratio:        {metrics['aspect_ratio']:.3f}")
    print(f"  Contact Area:        {metrics['area']:.2f} px²")
    print(f"  Perimeter:           {metrics['perimeter']:.2f} px")
    print(f"  Volume (disk int.):  {metrics['volume_estimate']:.2f} px³")
    print(f"{'='*60}\n")
    
    # --- PLOT COMPARISON ---
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 4, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.title("Enhanced (CLAHE)")
    plt.imshow(enhanced_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.title("Binary Segmentation")
    plt.imshow(binary_clean, cmap='gray')
    plt.axis('off')

    if final_roi is not None:
        plt.subplot(2, 4, 4)
        plt.title("ROI")
        plt.imshow(cv2.cvtColor(final_roi, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.title("Final Detection")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Results comparison table
    plt.subplot(2, 4, 6)
    plt.axis('off')
    plt.title("Contact Angles Comparison")
    
    methods_text = "METHOD          | LEFT   | RIGHT  | AVG\n"
    methods_text += "="*45 + "\n"
    
    if angle_left_tan:
        avg_tan = (angle_left_tan + angle_right_tan)/2 if angle_right_tan else angle_left_tan
        methods_text += f"Tangent Fit     | {angle_left_tan:5.1f}° | {angle_right_tan:5.1f}° | {avg_tan:5.1f}°\n"
    
    if angle_sphere:
        methods_text += f"Spherical Cap   | {angle_sphere:5.1f}° | {angle_sphere:5.1f}° | {angle_sphere:5.1f}°\n"
    
    if angle_left_ellipse:
        avg_ell = (angle_left_ellipse + angle_right_ellipse)/2
        methods_text += f"Elliptical      | {angle_left_ellipse:5.1f}° | {angle_right_ellipse:5.1f}° | {avg_ell:5.1f}°\n"
    
    if angle_left_yl:
        avg_yl = (angle_left_yl + angle_right_yl)/2
        methods_text += f"Young-Laplace   | {angle_left_yl:5.1f}° | {angle_right_yl:5.1f}° | {avg_yl:5.1f}°\n"
    
    plt.text(0.05, 0.5, methods_text, fontsize=9, family='monospace',
             verticalalignment='center')
    
    # Metrics summary
    plt.subplot(2, 4, 7)
    plt.axis('off')
    plt.title("Geometric Metrics")
    
    metrics_text = f"Width:  {metrics['base_width']:.1f} px\n"
    metrics_text += f"Height: {metrics['height']:.1f} px\n"
    metrics_text += f"Ratio:  {metrics['aspect_ratio']:.3f}\n"
    metrics_text += f"Area:   {metrics['area']:.1f} px²\n"
    metrics_text += f"Perim:  {metrics['perimeter']:.1f} px\n"
    
    plt.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    # Volume comparison
    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.title("Volume Estimates")
    
    vol_text = "METHOD          | VOLUME\n"
    vol_text += "="*30 + "\n"
    vol_text += f"Disk Integration| {metrics['volume_estimate']:.1f} px³\n"
    if vol_sphere:
        vol_text += f"Spherical Cap   | {vol_sphere:.1f} px³\n"
    if vol_ellipse:
        vol_text += f"Ellipsoid       | {vol_ellipse:.1f} px³\n"
    
    plt.text(0.05, 0.5, vol_text, fontsize=9, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.show()

# Run
sessile_drop_adaptive("./data/samples/prueba sesil 2.png")
sessile_drop_adaptive("./data/samples/gota depositada 1.png")
#sessile_drop_adaptive("./data/samples/gota pendiente 1.png")