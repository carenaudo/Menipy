import cv2
import numpy as np
import matplotlib.pyplot as plt
# Example Usage
def analyze_pendant_contact_points(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return

    output_img = img.copy()
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- STEP 1: PRECISE BINARIZATION ---
    # Otsu is still the best choice for this high-contrast silhouette
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Cleanup
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- STEP 2: DROP CONTOUR ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found.")
        return

    # Filter for the main drop (Largest Central)
    valid_contours = []
    img_center_x = width // 2
    for cnt in contours:
        if cv2.contourArea(cnt) > (width * height) * 0.05:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                if abs(cx - img_center_x) < (width * 0.3):
                    valid_contours.append(cnt)

    if not valid_contours:
        print("Error: Drop not found.")
        return
        
    drop_cnt = max(valid_contours, key=cv2.contourArea)
    
    # --- STEP 3: FIND CONTACT POINTS (The Logic Fix) ---
    
    # A. Get Bounding Box & Points
    x, y, w, h = cv2.boundingRect(drop_cnt)
    pts = drop_cnt.reshape(-1, 2)
    
    # B. Define Needle Shaft Reference (Top 20 px)
    # We take the median X of the points at the very top to define "Vertical"
    top_limit = y + 20
    
    # Left Shaft Line
    left_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] < (x + w/2))]
    if len(left_shaft_pts) == 0: return # Safety
    ref_x_left = np.median(left_shaft_pts[:, 0])
    
    # Right Shaft Line
    right_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] > (x + w/2))]
    if len(right_shaft_pts) == 0: return # Safety
    ref_x_right = np.median(right_shaft_pts[:, 0])

    # C. Walk Down to find deviation (The Contact Point)
    # We iterate Y from top to bottom.
    # We look for the first Y where the contour moves AWAY from the shaft line.
    
    contact_y_left = y
    contact_x_left = int(ref_x_left)
    
    contact_y_right = y
    contact_x_right = int(ref_x_right)
    
    # Tolerance: How many pixels "out" counts as the drop starting?
    # 3 pixels is usually enough to distinguish noise from the drop curve.
    tolerance = 3 
    
    # Scan logic: We check the mask row by row for robustness
    # (Contour points are not always ordered by Y, so mask scanning is easier)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [drop_cnt], -1, 255, 1) # Draw outline only
    
    # 1. Find Left Contact
    # Scan down the left side
    for cy in range(y, y + h):
        # Find the left-most white pixel in this row
        row = mask[cy, 0:int(x + w/2)] # Left half
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            current_x = indices[0] # First pixel
            # Check deviation
            if current_x < (ref_x_left - tolerance):
                contact_y_left = cy
                contact_x_left = current_x
                break # FOUND IT

    # 2. Find Right Contact
    for cy in range(y, y + h):
        # Find the right-most white pixel in this row
        row = mask[cy, int(x + w/2):width] # Right half
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            current_x = indices[-1] + int(x + w/2) # Last pixel + offset
            # Check deviation
            if current_x > (ref_x_right + tolerance):
                contact_y_right = cy
                contact_x_right = current_x
                break # FOUND IT

    # Define Needle Box bottom as the HIGHER of the two contact points (to be safe)
    needle_bottom = min(contact_y_left, contact_y_right)

    # --- STEP 4: APEX DETECTION ---
    apex_idx = np.argmax(pts[:, 1])
    apex_pt = pts[apex_idx]
    apex_x, apex_y = apex_pt

    # --- VISUALIZATION ---
    
    # 1. Draw Green Contour
    cv2.drawContours(output_img, [drop_cnt], -1, (0, 255, 0), 2)
    
    # 2. Draw Needle Shaft (Blue Filled Box with transparency)
    overlay = output_img.copy()
    cv2.rectangle(overlay, (int(ref_x_left), y), (int(ref_x_right), needle_bottom), (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, output_img, 0.7, 0, output_img)
    
    # Draw outline of needle
    cv2.rectangle(output_img, (int(ref_x_left), y), (int(ref_x_right), needle_bottom), (255, 0, 0), 2)
    cv2.putText(output_img, "Needle", (int(ref_x_left), y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 0, 0), 1)
    
    # 3. Draw Contact Points (Yellow Circles)
    cv2.circle(output_img, (contact_x_left, contact_y_left), 5, (0, 255, 255), -1)
    cv2.circle(output_img, (contact_x_right, contact_y_right), 5, (0, 255, 255), -1)
    
    # Label Contact Points
    cv2.putText(output_img, "CPL", (contact_x_left - 25, contact_y_left), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 30, 30), 1)
    cv2.putText(output_img, "CPR", (contact_x_right + 5, contact_y_right), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 30, 30), 1)

    # 4. Draw Apex
    cv2.drawMarker(output_img, (apex_x, apex_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(output_img, "Apex", (apex_x + 10, apex_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    
    # ROI Extraction
    pad = 40
    roi_coords = (max(0, x - pad), 
                  max(0, y), 
                  min(width, x + w + pad), 
                  min(height, apex_y + pad))
    final_roi = img[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    # Draw ROI
    cv2.rectangle(output_img, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 255), 2)
    cv2.putText(output_img, "ROI", (roi_coords[0], roi_coords[3] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 10, 10), 2)

    


    # --- PLOT ---
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.title("Step 1: Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Binary Mask (What computer sees)")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    if final_roi.size > 0:
        plt.subplot(2, 2, 3)
        plt.title("Final ROI")
        plt.imshow(cv2.cvtColor(final_roi, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Detection")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    

    plt.tight_layout()
    plt.show()

# Run
analyze_pendant_contact_points("./data/samples/gota pendiente 1.png")
analyze_pendant_contact_points("./data/samples/prueba pend 1.png")

