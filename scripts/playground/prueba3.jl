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
    
    # --- STEP 2: IMPROVED SUBSTRATE DETECTION ---
    margin_px = min(50, width ÷ 10)
    
    function find_horizon_robust(gray_img)
        h, w = size(gray_img)
        
        # Analyze left and right strips
        left_strip = gray_img[:, 1:min(margin_px, w÷4)]
        right_strip = gray_img[:, max(1, w-margin_px):w]
        
        function detect_baseline(strip)
            h_strip, w_strip = size(strip)
            candidates = Int[]
            
            # Only look in bottom 50% of image
            search_start = Int(floor(h_strip * 0.5))
            search_end = Int(floor(h_strip * 0.95))
            
            for col in 1:w_strip
                col_data = Float64.(strip[:, col])
                grad = diff(col_data)
                
                # Look for strongest negative gradient (dark to bright)
                search_grad = grad[search_start:search_end]
                if !isempty(search_grad)
                    min_val, min_idx = findmin(search_grad)
                    # Only accept strong gradients
                    if min_val < -0.05
                        push!(candidates, min_idx + search_start)
                    end
                end
            end
            
            if isempty(candidates)
                return nothing
            end
            
            # Use median but filter outliers
            med = median(candidates)
            filtered = filter(c -> abs(c - med) < h_strip * 0.1, candidates)
            return isempty(filtered) ? Int(round(med)) : Int(round(median(filtered)))
        end
        
        y_left = detect_baseline(left_strip)
        y_right = detect_baseline(right_strip)
        
        if !isnothing(y_left) && !isnothing(y_right)
            return Int(round((y_left + y_right) / 2))
        elseif !isnothing(y_left)
            return y_left
        elseif !isnothing(y_right)
            return y_right
        else
            # Fallback: analyze horizontal profile in bottom region
            bottom_region = gray_img[Int(h*0.6):end, :]
            profile = [mean(bottom_region[i, :]) for i in 1:size(bottom_region, 1)]
            grad_profile = diff(profile)
            
            if !isempty(grad_profile)
                min_idx = argmin(grad_profile)
                return min_idx + Int(h*0.6)
            else
                return Int(floor(h * 0.8))
            end
        end
    end
    
    substrate_y = find_horizon_robust(enhanced_gray)
    println("Detected substrate at y = $substrate_y (image height = $height)")
    # --- EARLY GEOMETRIC ROI FOR DROP CANDIDATES ---
    # Broad box: horizontally centered, from lower mid-image up to substrate
    roi_x1_early = Int(round(width * 0.15))
    roi_x2_early = Int(round(width * 0.85))
    roi_y1_early = Int(round(height * 0.25))
    roi_y2_early = substrate_y + 5  # small band below baseline for safety

    # --- STEP 3: IMPROVED ADAPTIVE SEGMENTATION ---
    blur = imfilter(enhanced_gray, Kernel.gaussian(2.5))
    
    # First, try Otsu thresholding on the region ABOVE substrate
    roi_above_substrate = blur[1:substrate_y-10, :]
    thresh_otsu = otsu_threshold(roi_above_substrate)
    
    # Use adaptive threshold with adjusted parameters
    binary = similar(blur, Bool)
    block_size = 31  # Larger block for this type of image
    half_block = block_size ÷ 2
    C = 0.01  # Smaller C value
    
    for i in 1:height
        for j in 1:width
            if i > substrate_y - 5
                # Below substrate, just set to false
                binary[i, j] = false
            else
                i_start = max(1, i - half_block)
                i_end = min(height, i + half_block)
                j_start = max(1, j - half_block)
                j_end = min(width, j + half_block)
                
                local_window = blur[i_start:i_end, j_start:j_end]
                local_mean = mean(local_window)
                
                # More aggressive threshold for dark objects
                binary[i, j] = blur[i, j] < (local_mean - C)
            end
        end
    end
    
    # Additional cleanup: remove small isolated regions
    kernel = strel_box((5, 5))
    binary_clean = opening(binary, kernel)
    
    # Then close to fill holes
    kernel_small = strel_box((3, 3))
    binary_clean = closing(binary_clean, kernel_small)
    binary_clean = closing(binary_clean, kernel_small)
    
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
        max_y = y + h
        max_x = x + w
        
        # Needle Logic: Touches top border AND is narrow and tall
        if y < 10 && isnothing(needle_cnt)
            aspect_ratio = h / (w + 1)
            if aspect_ratio > 2.0  # Needle should be tall and thin
                needle_cnt = cnt
                needle_y = cnt_center_y
                needle_x = max_x
            end
            continue  # don't consider this one as a drop
        end
        
        # --- EARLY ROI FILTER FOR DROP CANDIDATES ---
        # Only keep components for which most pixels lie inside the broad ROI
        inside_roi = count(c ->
            roi_x1_early <= c[2] <= roi_x2_early &&
            roi_y1_early <= c[1] <= roi_y2_early, cnt)
        frac_roi = inside_roi / max(area, 1)
        if frac_roi < 0.7
            continue  # discard components mostly outside the candidate region
        end
        
        # Drop Logic: Must be in lower portion and significant area
        if y > height * 0.2 && max_y > substrate_y - 20
            # More stringent filtering
            if area > (width * height) * 0.01 && x > 10 && (x + w) < (width - 10)
                # Check if mostly centered
                center_distance = abs(cnt_center_x - center_x)
                if center_distance < width * 0.4  # Within 40% of center
                    push!(valid_contours, (cnt, area, y))
                end
            end
        end
    end

    
    # Select drop contour - prefer lowest AND largest
    if isempty(valid_contours)
        println("Warning: No valid drop contours found, using fallback")
        # Fallback: largest contour that's not the needle
        non_needle = filter(c -> c != needle_cnt, contours)
        if !isempty(non_needle)
            drop_cnt = non_needle[argmax([length(c) for c in non_needle])]
        else
            println("Error: No drop detected")
            return nothing, nothing
        end
    else
        # Sort by area first, then by y position
        sort!(valid_contours, by=x -> (-x[2], x[3]))
        drop_cnt = valid_contours[1][1]
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
    
    # --- SHAPE PRIOR: enforce circular dome ---
    dome_points = enforce_circle_prior(dome_points, substrate_y)
    sort!(dome_points, by=p -> p[1])  # ensure still sorted
    
    x_left = dome_points[1][1]
    x_right = dome_points[end][1]
    
    # We'll refine the contact points with edges later; for now only x-positions
    cp_left_x = x_left
    cp_right_x = x_right
    
    # --- EDGE-BASED REFINEMENT OF CONTACT POINTS ---
    cp_left, cp_right = refine_contact_points(enhanced_gray, substrate_y, cp_left_x, cp_right_x)
    
    # Create final polygon: refined contact points + (smoothed) dome
    final_polygon = [cp_left; dome_points; cp_right]
    

    # ROI
    pad = 20
    roi_coords = (
        max(1, x_left - pad),
        max(1, minimum([p[2] for p in dome_points]) - pad),
        min(width, x_right + pad),
        min(height, substrate_y + pad)
    )
    
    final_roi = img[roi_coords[2]:roi_coords[4], roi_coords[1]:roi_coords[3]]
    
    # --- NEW: local binarization inside ROI ---

    roi_gray = Gray.(final_roi)
    roi_blur = imfilter(roi_gray, Kernel.gaussian(1.5))

    # You can try Otsu first
    t_roi = otsu_threshold(roi_blur)

    binary_roi = roi_blur .< t_roi

    # Remove tiny blobs & noise
    kernel_roi = strel_box((3,3))
    binary_roi = opening(binary_roi, kernel_roi)
    binary_roi = closing(binary_roi, kernel_roi)

    # Optional: kill anything touching top of ROI to avoid the needle/top noise
    h_roi, w_roi = size(binary_roi)
    for j in 1:w_roi
        if binary_roi[1, j]
            # flood fill from top or just clear a band
            binary_roi[1:min(15, h_roi), j] = false
        end
    end

    labels_roi = label_components(binary_roi)
    num_roi = maximum(labels_roi)

    if num_roi == 0
        println("Warning: no components in ROI; falling back to previous hull")
    else
        # largest component
        areas_roi = [count(labels_roi .== k) for k in 1:num_roi]
        best_k = argmax(areas_roi)
        roi_cnt = findall(labels_roi .== best_k)

        # Map ROI coords (i_roi, j_roi) -> global (y, x)
        drop_cnt = [(c[1] + roi_coords[2] - 1,
                    c[2] + roi_coords[1] - 1) for c in roi_cnt]

        # Rebuild hull & dome points based on this refined contour
        points = [(c[2], c[1]) for c in drop_cnt]  # (x,y)
        hull_points = convex_hull(points)

        dome_points = [pt for pt in hull_points if pt[2] < (substrate_y - 5)]
        if isempty(dome_points)
            println("Warning: ROI hull collapsed; using previous hull")
            return nothing, nothing
        else
            sort!(dome_points, by = p -> p[1])
            x_left = dome_points[1][1]
            x_right = dome_points[end][1]

            cp_left = (x_left, substrate_y)
            cp_right = (x_right, substrate_y)
            final_polygon = [cp_left; dome_points; cp_right]

            # From here you can reuse your drawing / mask code unchanged
            drop_mask = fill_polygon(height, width, final_polygon)
            # ... (same visualization code as before)
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
        end
    end    
    
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
"""
Refine left/right contact points using Canny edges in a band around the substrate.
Returns ((x_left, y_left), (x_right, y_right)).
"""
function refine_contact_points(
    gray_img, substrate_y::Int, x_left::Int, x_right::Int;
    search_half_width::Int = 20, band_above::Int = 15, band_below::Int = 3
)
    h, w = size(gray_img)
    y1 = max(1, substrate_y - band_above)
    y2 = min(h, substrate_y + band_below)
    
    # Edge map
    # Canny edge detection – universal fallback for all versions of ImageEdgeDetection
    edges = nothing
    try
        # Most modern versions
        edges = canny(gray_img)
    catch
        try
            # Some versions require sigma
            edges = canny(gray_img, 1.0)
        catch
            @warn "Canny edge detection is not available; using unrefined contact points"
            return (x_left, substrate_y), (x_right, substrate_y)
        end
    end
    band = edges[y1:y2, :]
    
    function best_point(x0::Int)
        xs = max(1, x0 - search_half_width):min(w, x0 + search_half_width)
        best = nothing
        best_dist = Inf
        for y in y1:y2
            for x in xs
                if band[y - y1 + 1, x]
                    d = abs(y - substrate_y)
                    if d < best_dist
                        best_dist = d
                        best = (x, y)
                    end
                end
            end
        end
        return isnothing(best) ? (x0, substrate_y) : best
    end
    
    cp_left  = best_point(x_left)
    cp_right = best_point(x_right)
    return cp_left, cp_right
end

"""
Enforce a circular shape prior on the dome by least-squares circle fitting.
Returns a new set of (x,y) dome points sampled from the fitted circle.
Falls back to original dome if fit fails.
"""
function enforce_circle_prior(
    dome_points::Vector{Tuple{Int,Int}},
    substrate_y::Int;
    num_samples::Int = 80
)
    n = length(dome_points)
    n < 3 && return dome_points  # cannot fit a circle
    
    xs = [float(p[1]) for p in dome_points]
    ys = [float(p[2]) for p in dome_points]
    
    A = zeros(Float64, n, 3)
    b = zeros(Float64, n)
    for i in 1:n
        x = xs[i]; y = ys[i]
        A[i, 1] = 2x
        A[i, 2] = 2y
        A[i, 3] = 1.0
        b[i] = x^2 + y^2
    end
    
    # Solve A * [cx, cy, c3] = b
    p = A \ b
    cx, cy, c3 = p
    r2 = c3 + cx^2 + cy^2
    if r2 <= 0 || !isfinite(r2)
        return dome_points
    end
    r = sqrt(r2)
    
    x_min = minimum(xs)
    x_max = maximum(xs)
    xs_samp = range(x_min, x_max, length=num_samples)
    
    new_points = Tuple{Int,Int}[]
    for x in xs_samp
        disc = r^2 - (x - cx)^2
        disc <= 0 && continue
        y_upper = cy - sqrt(disc)   # upper arc of the circle
        if y_upper < substrate_y - 2
            push!(new_points, (round(Int, x), round(Int, y_upper)))
        end
    end
    
    return isempty(new_points) ? dome_points : new_points
end

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
sessile_drop_adaptive("./data/samples/gota pendiente 1.png")