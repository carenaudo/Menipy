import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    # This is the magic step for low-contrast images.
    # It boosts local contrast without amplifying noise too much.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # --- STEP 2: ROBUST SUBSTRATE DETECTION ---
    # We use the ENHANCED image for detection
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
        # Fallback: Guess bottom 20%
        substrate_y = int(height * 0.8)
    else:
        substrate_y = int((y_left + y_right) / 2)

    # --- STEP 3: ADAPTIVE SEGMENTATION ---
    # Instead of global Otsu, we use Adaptive Thresholding.
    # It calculates the threshold for every small pixel neighborhood.
    # blockSize=21, C=2 are tuned for soft shadows.
    blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 2)

    # CLEANUP:
    # 1. Mask below substrate
    binary[substrate_y-2:, :] = 0
    
    # 2. Morphological Opening (Erosion followed by Dilation)
    # This removes the "grainy" noise that adaptive thresholding creates in the background
    kernel = np.ones((3,3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Morphological Closing (Dilation followed by Erosion)
    # This fills small holes inside the drop
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- STEP 4: CONTOUR & HULL ---
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    if not contours:
        print("Error: No contours found.")
        return

    # Filter Contours:
    # We want the LARGEST area, but also one that is somewhat CENTERED.
    # This avoids picking up dark corners of the image.
    valid_contours = []
    center_x = width // 2
    needle_cnt = None
    for cnt in contours:
        #contour bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        #auxiliary variables, for now calculed for all contours
        # area of the contour
        area = cv2.contourArea(cnt)
        # half value in the x-axis (horizontal midpoint)
        cnt_center_x = x + w//2 
        # half value in the y-axis (vertical midpoint)
        cnt_center_y = y + h//2
        # max value in the y-axis (bottom-most point)
        # The y-axis in images is inverted, so max y is the lowest point.
        max_y = y + h # lowest point of the contour
        min_y = y # highest point of the contour
        # min value in the x-axis (left-most point)
        min_x = x # left-most point of the contour
        # max value in the x-axis (right-most point)
        max_x = x + w # right-most point of the contour
        # Needle Logic: Touches top border
        if y < 5:
            if needle_cnt is None: 
                needle_cnt = cnt
                #auxiliary variables for needle text placing
                needle_y = cnt_center_y
                needle_x = max_x
        else:
            # Conditions:
            # 1. Area must be significant (> 0.5% of image)
            # 2. Must not be touching the left/right image border (artifacts)
            # 3. Center of contour should be roughly near the image center X
            if area > (width*height)*0.005 and x > 5 and (x+w) < (width-5):
                 valid_contours.append(cnt)
    
            if not valid_contours:
                print("Error: No valid drop contours found (check filtering).")
                # Fallback to largest raw contour
                drop_cnt = max(contours, key=cv2.contourArea)
            else:
                drop_cnt = max(valid_contours, key=cv2.contourArea)

                # Apply Convex Hull to smooth the shape
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
                drop_y = min_y
                drop_x = cnt_center_x

    # --- VISUALIZATION ---
    # Show Enhanced Gray to understand what the algorithm "sees"
    cv2.putText(output_img, "Substrate Baseline", (10, substrate_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.line(output_img, (0, substrate_y), (width, substrate_y), (255, 0, 255), 2)
    
    overlay = output_img.copy()
    cv2.drawContours(overlay, [final_cnt], -1, (0, 255, 0), -1) # Green Drop
    cv2.putText(output_img, "Drop", (drop_x - 5, drop_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 40, 0), 1)
    cv2.drawContours(overlay, [needle_cnt], -1, (0, 0, 255), -1) # Red Needle
    cv2.putText(output_img, "Needle", (needle_x + 5, needle_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)
    cv2.drawContours(output_img, [final_cnt], -1, (0, 255, 0), 2)

    
    cv2.circle(output_img, tuple(cp_left), 5, (0, 0, 255), -1)
    cv2.circle(output_img, tuple(cp_right), 5, (0, 0, 255), -1)

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
    # Plot Comparison
    plt.figure(figsize=(12, 6))
    

    plt.subplot(2, 2, 1)
    plt.title("Step 1: Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    if final_roi is not None:
        plt.subplot(2, 2, 2)
        plt.title("ROI (For Calculations)")
        plt.imshow(cv2.cvtColor(final_roi, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 3)
    plt.title("Step 2: Contrast Enhanced (CLAHE)")
    plt.imshow(enhanced_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Step 3: Final Detection")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')



    plt.show()

# Run
sessile_drop_adaptive("./data/samples/prueba sesil 2.png")
sessile_drop_adaptive("./data/samples/gota depositada 1.png")
sessile_drop_adaptive("./data/samples/gota pendiente 1.png")