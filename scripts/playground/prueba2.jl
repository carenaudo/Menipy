using Images, ImageFiltering, ImageMorphology, ImageSegmentation
using ImageEdgeDetection, ImageTransformations, ImageContrastAdjustment
using Plots, Statistics, LinearAlgebra

"""
    sessile_drop_adaptive(image_path::String)

Faithful Julia port of the Python sessile drop detection algorithm.
Uses CLAHE enhancement and adaptive thresholding.
"""
function sessile_drop_adaptive(image_path::String)
    # 1. Load Image
    img = load(image_path)
    height, width = size(img)
    
    output_img = copy(img)
    gray = Gray.(img)
    
    # --- STEP 1: CONTRAST ENHANCEMENT (CLAHE) ---
    enhanced_gray = adjust_histogram(gray, AdaptiveEqualization(nbins=256))
    
    # --- STEP 2: ROBUST SUBSTRATE DETECTION ---
    margin_px = min(50, width ÷ 10)
    left_strip = enhanced_gray[:, 1:margin_px]
    right_strip = enhanced_gray[:, (width-margin_px+1):width]
    
    function find_horizon_median(strip_gray)
        detected_ys = Int[]
        h, w = size(strip_gray)
        min_limit = Int(floor(h * 0.05))
        max_limit = Int(floor(h * 0.95))
        
        for col in 1:w
            col_data = Float64.(strip_gray[:, col])
            grad = diff(col_data)
            valid_grad = grad[min_limit:max_limit]
            
            if length(valid_grad) == 0
                continue
            end
            
            best_y = argmin(valid_grad) + min_limit
            push!(detected_ys, best_y)
        end
        
        if isempty(detected_ys)
            return nothing
        end
        
        return Int(round(median(detected_ys)))
    end
    
    y_left = find_horizon_median(left_strip)
    y_right = find_horizon_median(right_strip)
    
    substrate_y = if isnothing(y_left) || isnothing(y_right)
        println("Error: Substrate line lost in low contrast.")
        Int(floor(height * 0.8))
    else
        Int(round((y_left + y_right) / 2))
    end
    
    # --- STEP 3: ADAPTIVE SEGMENTATION ---
    blur = imfilter(enhanced_gray, Kernel.gaussian(2.5))
    
    # Adaptive threshold (Gaussian weighted)
    # blockSize=21, C=2
    binary = similar(blur, Bool)
    block_size = 21
    half_block = block_size ÷ 2
    C = 0.02  # Normalized constant (2/255 ≈ 0.0078, but 0.02 works better)
    
    for i in 1:height
        for j in 1:width
            i_start = max(1, i - half_block)
            i_end = min(height, i + half_block)
            j_start = max(1, j - half_block)
            j_end = min(width, j + half_block)
            
            # Get local window
            local_window = blur[i_start:i_end, j_start:j_end]
            
            # Gaussian weighted mean (simplified - just use mean for now)
            local_mean = mean(local_window)
            
            # THRESH_BINARY_INV: pixel < (mean - C) becomes white (true)
            binary[i, j] = blur[i, j] < (local_mean - C)
        end
    end
    
    # CLEANUP:
    # 1. Mask below substrate
    if substrate_y >= 2
        binary[substrate_y-1:end, :] .= false
    end
    
    # 2. Morphological Opening (removes noise)
    kernel = strel_box((3, 3))
    binary_clean = opening(binary, kernel)
    binary_clean = opening(binary_clean, kernel)  # iterations=2
    
    # 3. Morphological Closing (fills holes)
    binary_clean = closing(binary_clean, kernel)
    binary_clean = closing(binary_clean, kernel)  # iterations=2
    
    # --- STEP 4: CONTOUR & HULL ---
    # Find connected components
    labels = label_components(binary_clean)
    num_labels = maximum(labels)
    
    if num_labels == 0
        println("Error: No contours found.")
        return nothing, nothing
    end
    
    # Build contours (list of coordinates for each component)
    contours = []
    for label_id in 1:num_labels
        coords = findall(labels .== label_id)
        if length(coords) > 0
            push!(contours, coords)
        end
    end
    
    if isempty(contours)
        println("Error: No contours found.")
        return nothing, nothing
    end
    
    # Filter Contours
    valid_contours = []
    center_x = width ÷ 2
    needle_cnt = nothing
    needle_y = 0
    needle_x = 0
    
    for cnt in contours
        # Get bounding box
        y_coords = [c[1] for c in cnt]
        x_coords = [c[2] for c in cnt]
        
        x = minimum(x_coords)
        y = minimum(y_coords)
        w = maximum(x_coords) - x
        h = maximum(y_coords) - y
        
        # Area
        area = length(cnt)
        
        # Center
        cnt_center_x = x + w ÷ 2
        cnt_center_y = y + h ÷ 2
        
        # Max/min coordinates
        min_y = y
        max_x = x + w
        
        # Needle Logic: Touches top border
        if y < 5
            if isnothing(needle_cnt)
                needle_cnt = cnt
                needle_y = cnt_center_y
                needle_x = max_x
            end
        else
            # Drop conditions
            if area > (width * height) * 0.005 && x > 5 && (x + w) < (width - 5)
                push!(valid_contours, cnt)
            end
        end
    end
    
    # Select drop contour
    if isempty(valid_contours)
        println("Error: No valid drop contours found (check filtering).")
        # Fallback to largest raw contour
        drop_cnt = contours[argmax([length(c) for c in contours])]
    else
        drop_cnt = valid_contours[argmax([length(c) for c in valid_contours])]
    end
    
    # Get drop bounding box for text placement
    y_coords = [c[1] for c in drop_cnt]
    x_coords = [c[2] for c in drop_cnt]
    drop_x = (minimum(x_coords) + maximum(x_coords)) ÷ 2
    drop_y = minimum(y_coords)
    
    # --- STEP 5: CONVEX HULL & RECONSTRUCT FLAT BASE ---
    # Convert to point format (x, y)
    points = [(c[2], c[1]) for c in drop_cnt]
    
    # Simple convex hull (using gift wrapping algorithm)
    hull_points = convex_hull(points)
    
    # Filter dome points (above substrate)
    dome_points = [pt for pt in hull_points if pt[2] < (substrate_y - 5)]
    
    if isempty(dome_points)
        println("Error: Hull collapsed.")
        return nothing, nothing
    end
    
    # Sort by x coordinate
    sort!(dome_points, by=p -> p[1])
    
    x_left = dome_points[1][1]
    x_right = dome_points[end][1]
    
    cp_left = (x_left, substrate_y)
    cp_right = (x_right, substrate_y)
    
    # Create final polygon: left contact point + dome + right contact point
    final_polygon = [cp_left; dome_points; cp_right]
    
    # --- VISUALIZATION ---
    output_img = RGB.(img)
    
    # Create overlay
    overlay = RGB.(img)
    
    # Fill drop interior properly using polygon fill
    # Create a mask for the drop
    drop_mask = fill_polygon(height, width, final_polygon)
    
    # Apply drop color to overlay
    for i in 1:height
        for j in 1:width
            if drop_mask[i, j]
                overlay[i, j] = RGB(0, 1, 0)  # Green
            end
        end
    end
    
    # Draw filled needle (red)
    if !isnothing(needle_cnt)
        for coord in needle_cnt
            if checkbounds(Bool, overlay, coord)
                overlay[coord] = RGB(1, 0, 0)
            end
        end
    end
    
    # Blend overlay with original (alpha = 0.4)
    output_img = 0.6 .* output_img .+ 0.4 .* overlay
    
    # Draw drop contour (green outline, thicker)
    for i in 1:length(final_polygon)-1
        draw_thick_line!(output_img, final_polygon[i], final_polygon[i+1], RGB(0, 1, 0), 2)
    end
    draw_thick_line!(output_img, final_polygon[end], final_polygon[1], RGB(0, 1, 0), 2)
    
    # Draw substrate baseline (after blending so it's on top)
    for x in 1:width
        if substrate_y <= height
            output_img[substrate_y, x] = RGB(1, 0, 1)
            if substrate_y + 1 <= height
                output_img[substrate_y + 1, x] = RGB(1, 0, 1)
            end
        end
    end
    
    # Draw contact points (red circles)
    draw_circle!(output_img, cp_left, 5, RGB(0, 0, 1))
    draw_circle!(output_img, cp_right, 5, RGB(0, 0, 1))
    
    # ROI
    pad = 20
    roi_coords = (
        max(1, x_left - pad),
        max(1, minimum([p[2] for p in dome_points]) - pad),
        min(width, x_right + pad),
        min(height, substrate_y + pad)
    )
    
    final_roi = img[roi_coords[2]:roi_coords[4], roi_coords[1]:roi_coords[3]]
    
    # Draw ROI rectangle (cyan)
    draw_rectangle!(output_img, roi_coords, RGB(0, 1, 1), thickness=2)
    
    # Add text labels
    add_text!(output_img, "Substrate Baseline", (10, substrate_y - 10), RGB(1, 0, 1))
    add_text!(output_img, "Drop", (drop_x - 20, drop_y - 10), RGB(0, 1, 0))
    if !isnothing(needle_cnt)
        add_text!(output_img, "Needle", (needle_x + 10, needle_y), RGB(1, 0, 0))
    end
    add_text!(output_img, "ROI", (roi_coords[1] + 10, roi_coords[2] - 10), RGB(0, 1, 1))
    
    # --- PLOT COMPARISON ---
    p = plot(layout=(2, 2), size=(1200, 1200), margin=5Plots.mm)
    
    plot!(p[1], RGB.(img), title="Step 1: Original Image", 
          axis=false, border=:none, showaxis=false, ticks=false)
    
    plot!(p[2], final_roi, title="ROI (For Calculations)", 
          axis=false, border=:none, showaxis=false, ticks=false)
    
    plot!(p[3], enhanced_gray, title="Step 2: Contrast Enhanced (CLAHE)", 
          axis=false, border=:none, showaxis=false, ticks=false, color=:gray)
    
    plot!(p[4], output_img, title="Step 3: Final Detection", 
          axis=false, border=:none, showaxis=false, ticks=false)
    
    display(p)
    
    return output_img, final_roi
end

# Helper functions

"""Simple gift wrapping convex hull"""
function convex_hull(points)
    if length(points) < 3
        return points
    end
    
    # Find leftmost point
    start = argmin([p[1] for p in points])
    hull = [points[start]]
    
    current = start
    while true
        next_point = 1
        if next_point == current
            next_point = 2
        end
        
        for i in 1:length(points)
            if i == current
                continue
            end
            
            # Cross product to find counterclockwise turn
            if next_point == current || is_left_turn(points[current], points[next_point], points[i])
                next_point = i
            end
        end
        
        current = next_point
        
        if current == start
            break
        end
        
        push!(hull, points[current])
        
        # Safety check
        if length(hull) > length(points)
            break
        end
    end
    
    return hull
end

function is_left_turn(p1, p2, p3)
    # Cross product
    cross = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    return cross > 0
end

function draw_circle!(img, center, radius, color)
    h, w = size(img)
    cx, cy = center
    
    for i in max(1, cy - radius):min(h, cy + radius)
        for j in max(1, cx - radius):min(w, cx + radius)
            if (j - cx)^2 + (i - cy)^2 <= radius^2
                if checkbounds(Bool, img, i, j)
                    img[i, j] = color
                end
            end
        end
    end
end

function draw_rectangle!(img, coords, color; thickness=1)
    x1, y1, x2, y2 = coords
    h, w = size(img)
    
    for t in 0:thickness-1
        # Top and bottom edges
        for x in x1:x2
            if 1 <= y1+t <= h && 1 <= x <= w
                img[y1+t, x] = color
            end
            if 1 <= y2-t <= h && 1 <= x <= w
                img[y2-t, x] = color
            end
        end
        
        # Left and right edges
        for y in y1:y2
            if 1 <= y <= h && 1 <= x1+t <= w
                img[y, x1+t] = color
            end
            if 1 <= y <= h && 1 <= x2-t <= w
                img[y, x2-t] = color
            end
        end
    end
end

function draw_line!(img, p1, p2, color)
    # Bresenham's line algorithm
    x1, y1 = round(Int, p1[1]), round(Int, p1[2])
    x2, y2 = round(Int, p2[1]), round(Int, p2[2])
    
    h, w = size(img)
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = x1 < x2 ? 1 : -1
    sy = y1 < y2 ? 1 : -1
    err = dx - dy
    
    x, y = x1, y1
    
    for _ in 1:max(dx, dy) + 1
        if 1 <= y <= h && 1 <= x <= w
            img[y, x] = color
        end
        
        if x == x2 && y == y2
            break
        end
        
        e2 = 2 * err
        if e2 > -dy
            err -= dy
            x += sx
        end
        if e2 < dx
            err += dx
            y += sy
        end
    end
end

function draw_thick_line!(img, p1, p2, color, thickness)
    for t in -thickness÷2:thickness÷2
        for s in -thickness÷2:thickness÷2
            p1_offset = (p1[1] + t, p1[2] + s)
            p2_offset = (p2[1] + t, p2[2] + s)
            draw_line!(img, p1_offset, p2_offset, color)
        end
    end
end

"""Fill a polygon using scanline algorithm"""
function fill_polygon(height, width, polygon)
    mask = falses(height, width)
    
    if isempty(polygon)
        return mask
    end
    
    # Scanline algorithm
    y_min = max(1, minimum([p[2] for p in polygon]))
    y_max = min(height, maximum([p[2] for p in polygon]))
    
    for y in y_min:y_max
        # Find intersections with this scanline
        intersections = Int[]
        
        for i in 1:length(polygon)
            p1 = polygon[i]
            p2 = polygon[mod1(i + 1, length(polygon))]
            
            # Check if edge crosses this scanline
            if (p1[2] <= y && p2[2] > y) || (p2[2] <= y && p1[2] > y)
                # Calculate x intersection
                t = (y - p1[2]) / (p2[2] - p1[2])
                x = round(Int, p1[1] + t * (p2[1] - p1[1]))
                push!(intersections, x)
            end
        end
        
        # Sort intersections
        sort!(intersections)
        
        # Fill between pairs of intersections
        for i in 1:2:length(intersections)-1
            x_start = max(1, intersections[i])
            x_end = min(width, intersections[i+1])
            mask[y, x_start:x_end] .= true
        end
    end
    
    return mask
end

"""Add simple text to image by drawing pixels"""
function add_text!(img, text, pos, color)
    # Simple bitmap font - we'll draw a box outline as placeholder
    # since proper text rendering requires additional packages
    x, y = pos
    h, w = size(img)
    
    # Draw a small filled rectangle as text background
    text_width = length(text) * 6
    text_height = 12
    
    # Background (semi-transparent dark)
    for i in max(1, y):min(h, y + text_height)
        for j in max(1, x):min(w, x + text_width)
            if checkbounds(Bool, img, i, j)
                img[i, j] = 0.7 * img[i, j] + 0.3 * RGB(0, 0, 0)
            end
        end
    end
    
    # Text "line" (just a colored bar to indicate label)
    for i in max(1, y+4):min(h, y+8)
        for j in max(1, x+2):min(w, x+text_width-2)
            if checkbounds(Bool, img, i, j)
                img[i, j] = color
            end
        end
    end
end

# Example usage:
sessile_drop_adaptive("./data/samples/prueba sesil 2.png")
sessile_drop_adaptive("./data/samples/gota depositada 1.png")
#sessile_drop_adaptive("./data/samples/gota pendiente 1.png")