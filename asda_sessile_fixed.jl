"""
Sessile Drop Contact Angle Analysis Script - FIXED VERSION
Measures contact angles from sessile drop images with improved drop detection
Fixes: Center-focused detection, dark drop support, better component scoring
Julia 1.11+
"""

using Images, ImageFiltering, ImageMorphology, ImageSegmentation
using Polynomials, Statistics, LinearAlgebra
using Plots

# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================

struct AnalysisConfig
    known_distance_mm::Union{Float64,Nothing}
    blur_sigma::Float64
    threshold_method::Symbol
    manual_threshold::Float64
    baseline_detection::Symbol
    poly_degree::Int
    fit_window_mm::Float64
    min_drop_area::Int
    invert_binary::Bool           # NEW: for dark drops on light background
    center_preference::Float64    # NEW: weight for center proximity (0-2)
end

# Default configuration
default_config() = AnalysisConfig(
    nothing,   # No known distance (use pixel units)
    2.0,       # Blur sigma
    :otsu,     # Automatic thresholding
    0.5,       # Manual threshold (0-1)
    :ransac,   # RANSAC for robust baseline detection
    3,         # Cubic polynomial
    0.5,       # Fit 0.5mm around contact point
    1000,      # Minimum 1000 pixels for drop
    false,     # Don't invert by default
    0.5        # Moderate center preference
)

# ============================================================================
# STEP 1: IMAGE LOADING
# ============================================================================

function load_image(filepath::String)
    println("Loading image: $filepath")
    img = load(filepath)
    println("  Image size: $(size(img))")
    return img
end

# ============================================================================
# STEP 2: IMPROVED IMAGE PREPROCESSING
# ============================================================================

function preprocess_image(img, config::AnalysisConfig)
    println("\nPreprocessing image...")

    # Convert to grayscale
    gray_img = Gray.(img)
    println("  ✓ Converted to grayscale")

    # Apply Gaussian blur to reduce noise
    blurred = imfilter(gray_img, Kernel.gaussian(config.blur_sigma))
    println("  ✓ Applied Gaussian blur (σ=$(config.blur_sigma))")

    # Binarize image
    binary = binarize_image(blurred, config)
    invert_msg = config.invert_binary ? " (inverted for dark drops)" : ""
    println("  ✓ Binarized image$invert_msg")

    # Morphological operations to clean up
    cleaned = clean_binary(binary)
    println("  ✓ Cleaned binary image")

    return gray_img, binary, cleaned
end

function binarize_image(img, config::AnalysisConfig)
    if config.threshold_method == :otsu
        threshold = otsu_threshold(img)
        binary = img .> threshold
    elseif config.threshold_method == :yen
        # Simple Yen-like threshold
        threshold = quantile(vec(img), 0.5)
        binary = img .> threshold
    else
        binary = img .> config.manual_threshold
    end

    # Invert if drop is darker than background
    if config.invert_binary
        binary = .!binary
    end

    return binary
end

function clean_binary(binary)
    # Remove small objects and fill holes
    cleaned = copy(binary)
    cleaned = opening(cleaned, strel_diamond((3, 3)))
    cleaned = closing(cleaned, strel_diamond((5, 5)))
    return cleaned
end

# ============================================================================
# STEP 3: IMPROVED ELEMENT DETECTION
# ============================================================================

struct DetectedElements
    drop_region::Vector{CartesianIndex{2}}
    substrate_line::Vector{Tuple{Float64,Float64}}
    contact_left::Tuple{Float64,Float64}
    contact_right::Tuple{Float64,Float64}
    tilt_angle::Float64
    drop_apex::Tuple{Float64,Float64}
end

function detect_elements(binary_img, config::AnalysisConfig)
    println("\nDetecting elements...")

    # Find drop region (IMPROVED)
    drop_mask, drop_coords = find_drop(binary_img, config)
    println("  ✓ Drop detected ($(length(drop_coords)) pixels)")

    # Detect substrate baseline
    baseline, tilt_angle = detect_substrate(binary_img, drop_mask, config)
    println("  ✓ Substrate detected (tilt: $(round(tilt_angle, digits=2))°)")

    # Find contact points
    contact_left, contact_right = find_contact_points(drop_mask, baseline)
    println("  ✓ Contact points: Left=$contact_left, Right=$contact_right")

    # Find drop apex
    apex = find_apex(drop_coords)
    println("  ✓ Drop apex: $apex")

    return DetectedElements(
        drop_coords, baseline, contact_left, contact_right,
        tilt_angle, apex
    )
end

function find_drop(binary_img, config::AnalysisConfig)
    println("  Analyzing components with center-focused scoring...")

    # Find connected components
    labels = label_components(binary_img)
    component_sizes = component_lengths(labels)

    rows, cols = size(binary_img)
    # Expected drop location: horizontal center, lower 70% vertically
    center_x, center_y = cols / 2, rows * 0.7

    # Evaluate each component
    candidates = []

    for label in keys(component_sizes)
        if label == 0 || component_sizes[label] < config.min_drop_area
            continue
        end

        # Get component coordinates
        coords = findall(labels .== label)
        size_val = component_sizes[label]

        # Calculate centroid
        centroid_x = mean([c[2] for c in coords])
        centroid_y = mean([c[1] for c in coords])

        # Calculate distance from expected drop location
        dist_from_center = sqrt((centroid_x - center_x)^2 +
                                (centroid_y - center_y)^2)

        # Calculate bounding box and aspect ratio
        min_x = minimum([c[2] for c in coords])
        max_x = maximum([c[2] for c in coords])
        min_y = minimum([c[1] for c in coords])
        max_y = maximum([c[1] for c in coords])

        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = height / max(width, 1)

        # Multi-factor scoring:
        # 1. Size score: larger is better (normalized)
        size_score = size_val / (rows * cols)

        # 2. Location score: closer to expected location is better (Gaussian)
        location_score = exp(-dist_from_center^2 / (2 * (cols / 4)^2))

        # 3. Aspect ratio score: typical drops have aspect 0.2-2.0
        aspect_score = (aspect_ratio > 0.2 && aspect_ratio < 2.0) ? 1.0 : 0.3

        # 4. Position score: prefer lower regions (drops sit on substrate)
        position_score = centroid_y / rows  # Higher y = lower in image = better

        # Combined score with configurable center preference
        score = (size_score^0.5) *
                (location_score^config.center_preference) *
                aspect_score *
                (position_score^0.3)

        push!(candidates, (
            label=label,
            score=score,
            size=size_val,
            centroid=(centroid_x, centroid_y),
            aspect=aspect_ratio,
            bbox=(min_x, max_x, min_y, max_y)
        ))
    end

    if isempty(candidates)
        error("No drop detected. Try adjusting: threshold, min_drop_area, or invert_binary.")
    end

    # Sort by score and show top candidates
    sort!(candidates, by=c -> c.score, rev=true)

    println("    Top $(min(3, length(candidates))) candidate(s):")
    for (i, cand) in enumerate(candidates[1:min(3, length(candidates))])
        println("      $i. Size=$(cand.size)px, " *
                "Center=($(round(cand.centroid[1], digits=1)), $(round(cand.centroid[2], digits=1))), " *
                "Aspect=$(round(cand.aspect, digits=2)), " *
                "Score=$(round(cand.score, digits=4))")
    end

    # Select best candidate
    drop_label = candidates[1].label
    drop_mask = labels .== drop_label
    drop_coords = findall(drop_mask)

    println("    → Selected component $drop_label")

    return drop_mask, drop_coords
end

function detect_substrate(binary_img, drop_mask, config::AnalysisConfig)
    if config.baseline_detection == :ransac
        return detect_substrate_ransac(binary_img, drop_mask)
    elseif config.baseline_detection == :hough
        return detect_substrate_hough(binary_img, drop_mask)
    else
        return detect_substrate_bottom(drop_mask)
    end
end

function detect_substrate_ransac(binary_img, drop_mask)
    # Extract bottom edge of drop as potential substrate points
    rows, cols = size(binary_img)
    bottom_points = Tuple{Float64,Float64}[]

    # Find bottom-most pixels of drop for each column
    for col in 1:cols
        column_drop = findall(drop_mask[:, col])
        if !isempty(column_drop)
            bottom_row = maximum(column_drop)
            push!(bottom_points, (Float64(col), Float64(bottom_row)))
        end
    end

    if length(bottom_points) < 10
        error("Not enough points for baseline detection")
    end

    # RANSAC to fit robust line
    best_inliers = Int[]
    best_line = nothing
    n_iterations = 100
    threshold = 5.0  # pixels

    for _ in 1:n_iterations
        # Sample 2 points
        idx = rand(1:length(bottom_points), 2)
        p1, p2 = bottom_points[idx[1]], bottom_points[idx[2]]

        # Fit line through these points
        if abs(p2[1] - p1[1]) < 1e-6
            continue  # Skip vertical lines
        end

        slope = (p2[2] - p1[2]) / (p2[1] - p1[1])
        intercept = p1[2] - slope * p1[1]

        # Count inliers
        inliers = Int[]
        for (i, p) in enumerate(bottom_points)
            predicted_y = slope * p[1] + intercept
            if abs(p[2] - predicted_y) < threshold
                push!(inliers, i)
            end
        end

        if length(inliers) > length(best_inliers)
            best_inliers = inliers
            best_line = (slope, intercept)
        end
    end

    if best_line === nothing || length(best_inliers) < 10
        # Fallback to simple bottom detection
        return detect_substrate_bottom(drop_mask)
    end

    # Refine line fit with all inliers
    inlier_points = bottom_points[best_inliers]
    x_coords = [p[1] for p in inlier_points]
    y_coords = [p[2] for p in inlier_points]

    # Least squares fit
    A = hcat(x_coords, ones(length(x_coords)))
    coeffs = A \ y_coords
    slope, intercept = coeffs[1], coeffs[2]

    # Generate baseline points
    x_min, x_max = extrema(x_coords)
    baseline_x = range(x_min, x_max, length=100)
    baseline = [(x, slope * x + intercept) for x in baseline_x]

    # Calculate tilt angle
    tilt_angle = atand(slope)

    return baseline, tilt_angle
end

function detect_substrate_bottom(drop_mask)
    # Simple method: find bottom of drop and assume horizontal
    drop_coords = findall(drop_mask)
    bottom_y = maximum([c[1] for c in drop_coords])

    cols = size(drop_mask, 2)
    baseline = [(Float64(x), Float64(bottom_y)) for x in 1:cols]

    return baseline, 0.0
end

function detect_substrate_hough(binary_img, drop_mask)
    """
    Detect substrate baseline using Hough Transform for line detection.
    
    The Hough Transform maps each edge point to a sinusoidal curve in (ρ, θ) space,
    where ρ = x*cos(θ) + y*sin(θ). Lines appear as peaks in this accumulator space.
    
    For substrate detection, we focus on near-horizontal lines (θ ≈ 90°) at the
    bottom portion of the drop.
    """
    rows, cols = size(binary_img)

    # Extract bottom edge points from the drop mask
    bottom_points = Tuple{Float64,Float64}[]
    for col in 1:cols
        column_drop = findall(drop_mask[:, col])
        if !isempty(column_drop)
            bottom_row = maximum(column_drop)
            push!(bottom_points, (Float64(col), Float64(bottom_row)))
        end
    end

    if length(bottom_points) < 10
        @warn "Not enough edge points for Hough transform, falling back to bottom detection"
        return detect_substrate_bottom(drop_mask)
    end

    # Hough Transform Parameters
    # θ: angle resolution (focus on near-horizontal lines: 70° to 110°)
    theta_min = deg2rad(70)
    theta_max = deg2rad(110)
    theta_step = deg2rad(0.5)
    thetas = theta_min:theta_step:theta_max

    # ρ: distance from origin, max possible is diagonal of image
    rho_max = sqrt(rows^2 + cols^2)
    rho_step = 1.0
    rhos = -rho_max:rho_step:rho_max
    n_rhos = length(rhos)
    n_thetas = length(thetas)

    # Initialize accumulator array
    accumulator = zeros(Int, n_rhos, n_thetas)

    # Precompute cos and sin for efficiency
    cos_thetas = cos.(thetas)
    sin_thetas = sin.(thetas)

    # Vote in accumulator for each edge point
    for (x, y) in bottom_points
        for (j, (cos_t, sin_t)) in enumerate(zip(cos_thetas, sin_thetas))
            # Calculate ρ for this (x, y, θ) combination
            rho = x * cos_t + y * sin_t

            # Find corresponding bin in accumulator
            rho_idx = round(Int, (rho + rho_max) / rho_step) + 1

            if 1 <= rho_idx <= n_rhos
                accumulator[rho_idx, j] += 1
            end
        end
    end

    # Find peaks in accumulator (potential lines)
    # Apply non-maximum suppression in a local neighborhood
    peak_threshold = length(bottom_points) * 0.3  # At least 30% of points
    neighborhood = 5

    peaks = []
    for j in 1:n_thetas
        for i in 1:n_rhos
            if accumulator[i, j] < peak_threshold
                continue
            end

            # Check if this is a local maximum
            is_max = true
            i_start = max(1, i - neighborhood)
            i_end = min(n_rhos, i + neighborhood)
            j_start = max(1, j - neighborhood)
            j_end = min(n_thetas, j + neighborhood)

            for jj in j_start:j_end
                for ii in i_start:i_end
                    if accumulator[ii, jj] > accumulator[i, j]
                        is_max = false
                        break
                    end
                end
                !is_max && break
            end

            if is_max
                rho = rhos[i]
                theta = thetas[j]
                push!(peaks, (votes=accumulator[i, j], rho=rho, theta=theta))
            end
        end
    end

    if isempty(peaks)
        @warn "No Hough peaks found, falling back to bottom detection"
        return detect_substrate_bottom(drop_mask)
    end

    # Sort peaks by vote count and select the best one
    sort!(peaks, by=p -> p.votes, rev=true)
    best_peak = peaks[1]

    println("    Hough: Found $(length(peaks)) line candidate(s)")
    println("    Best line: ρ=$(round(best_peak.rho, digits=1)), " *
            "θ=$(round(rad2deg(best_peak.theta), digits=1))°, " *
            "votes=$(best_peak.votes)")

    # Convert (ρ, θ) back to line parameters
    # Line equation: x*cos(θ) + y*sin(θ) = ρ
    # Solve for y: y = (ρ - x*cos(θ)) / sin(θ)
    rho = best_peak.rho
    theta = best_peak.theta

    if abs(sin(theta)) < 1e-6
        # Near-vertical line (shouldn't happen for substrate)
        @warn "Hough detected near-vertical line, falling back to bottom detection"
        return detect_substrate_bottom(drop_mask)
    end

    # Refine the line using points close to the detected Hough line
    inlier_threshold = 5.0  # pixels
    inliers = Tuple{Float64,Float64}[]

    for (x, y) in bottom_points
        # Distance from point to line: |x*cos(θ) + y*sin(θ) - ρ|
        dist = abs(x * cos(theta) + y * sin(theta) - rho)
        if dist < inlier_threshold
            push!(inliers, (x, y))
        end
    end

    if length(inliers) < 10
        # Use the Hough line directly
        x_min = minimum(p[1] for p in bottom_points)
        x_max = maximum(p[1] for p in bottom_points)
        baseline_x = range(x_min, x_max, length=100)
        baseline = [(x, (rho - x * cos(theta)) / sin(theta)) for x in baseline_x]
        tilt_angle = rad2deg(theta) - 90.0
    else
        # Refine with least squares fit on inliers
        x_coords = [p[1] for p in inliers]
        y_coords = [p[2] for p in inliers]

        A = hcat(x_coords, ones(length(x_coords)))
        coeffs = A \ y_coords
        slope, intercept = coeffs[1], coeffs[2]

        x_min, x_max = extrema(x_coords)
        baseline_x = range(x_min, x_max, length=100)
        baseline = [(x, slope * x + intercept) for x in baseline_x]

        tilt_angle = atand(slope)
    end

    return baseline, tilt_angle
end

function find_contact_points(drop_mask, baseline)
    # Find intersection of drop with baseline
    baseline_y_mean = mean([p[2] for p in baseline])
    baseline_tolerance = 5.0  # pixels

    # Find drop pixels near baseline
    drop_coords = findall(drop_mask)
    contact_pixels = filter(c -> abs(c[1] - baseline_y_mean) < baseline_tolerance,
        drop_coords)

    if isempty(contact_pixels)
        error("No contact points found")
    end

    # Extract column indices
    contact_cols = [c[2] for c in contact_pixels]

    # Left and right contact points
    left_col = minimum(contact_cols)
    right_col = maximum(contact_cols)

    # Get y-coordinate from baseline
    left_y = baseline_y_mean
    right_y = baseline_y_mean

    return (Float64(left_col), left_y), (Float64(right_col), right_y)
end

function find_apex(drop_coords)
    # Apex is the top-most point
    top_row = minimum([c[1] for c in drop_coords])
    apex_pixels = filter(c -> c[1] == top_row, drop_coords)

    # Average column position at apex
    apex_col = mean([c[2] for c in apex_pixels])

    return (Float64(apex_col), Float64(top_row))
end

# ============================================================================
# STEP 4: TILT CORRECTION
# ============================================================================

function correct_tilt(elements::DetectedElements, img_size)
    println("\nCorrecting tilt...")

    if abs(elements.tilt_angle) < 0.5
        println("  ✓ Tilt negligible, no correction needed")
        return elements
    end

    # Rotate elements to make baseline horizontal
    angle_rad = -deg2rad(elements.tilt_angle)

    # Rotation center (middle of baseline)
    cx = img_size[2] / 2
    cy = mean([p[2] for p in elements.substrate_line])

    # Rotate all points
    rotated_drop = [rotate_point(c[2], c[1], cx, cy, angle_rad)
                    for c in elements.drop_region]
    rotated_baseline = [rotate_point(p[1], p[2], cx, cy, angle_rad)
                        for p in elements.substrate_line]
    rotated_left = rotate_point(elements.contact_left[1], elements.contact_left[2],
        cx, cy, angle_rad)
    rotated_right = rotate_point(elements.contact_right[1], elements.contact_right[2],
        cx, cy, angle_rad)
    rotated_apex = rotate_point(elements.drop_apex[1], elements.drop_apex[2],
        cx, cy, angle_rad)

    # Create corrected elements
    corrected = DetectedElements(
        CartesianIndex{2}[],
        rotated_baseline,
        rotated_left,
        rotated_right,
        0.0,
        rotated_apex
    )

    println("  ✓ Tilt corrected (was $(round(elements.tilt_angle, digits=2))°)")

    return corrected, rotated_drop
end

function rotate_point(x, y, cx, cy, angle)
    x_shifted = x - cx
    y_shifted = y - cy

    x_rot = x_shifted * cos(angle) - y_shifted * sin(angle)
    y_rot = x_shifted * sin(angle) + y_shifted * cos(angle)

    return (x_rot + cx, y_rot + cy)
end

# ============================================================================
# STEP 5: DROP CONTOUR EXTRACTION
# ============================================================================

struct DropContour
    left_side::Vector{Tuple{Float64,Float64}}
    right_side::Vector{Tuple{Float64,Float64}}
    baseline_y::Float64
end

function extract_contour(drop_coords, elements::DetectedElements, config::AnalysisConfig)
    println("\nExtracting drop contour...")

    baseline_y = mean([p[2] for p in elements.substrate_line])
    apex_x = elements.drop_apex[1]

    # Split into left and right sides
    left_points = Tuple{Float64,Float64}[]
    right_points = Tuple{Float64,Float64}[]

    for coord in drop_coords
        x, y = Float64(coord[2]), Float64(coord[1])

        if x < apex_x
            push!(left_points, (x, y))
        else
            push!(right_points, (x, y))
        end
    end

    # Sort by y-coordinate (top to bottom)
    sort!(left_points, by=p -> p[2])
    sort!(right_points, by=p -> p[2])

    # Extract edges
    left_edge = extract_edge_points(left_points, true)
    right_edge = extract_edge_points(right_points, false)

    println("  ✓ Contour extracted: $(length(left_edge)) (left), $(length(right_edge)) (right)")

    return DropContour(left_edge, right_edge, baseline_y)
end

function extract_edge_points(points, is_left::Bool)
    # Group by y-coordinate and take leftmost/rightmost x for each y
    y_dict = Dict{Int,Vector{Float64}}()

    for (x, y) in points
        y_int = round(Int, y)
        if !haskey(y_dict, y_int)
            y_dict[y_int] = Float64[]
        end
        push!(y_dict[y_int], x)
    end

    # Extract edge
    edge = Tuple{Float64,Float64}[]
    for y in sort(collect(keys(y_dict)))
        x_values = y_dict[y]
        x_edge = is_left ? minimum(x_values) : maximum(x_values)
        push!(edge, (x_edge, Float64(y)))
    end

    return edge
end

# ============================================================================
# STEP 6: CONTACT ANGLE CALCULATION
# ============================================================================

struct ContactAngle
    angle::Float64
    contact_point::Tuple{Float64,Float64}
    tangent_slope::Float64
    fit_points::Vector{Tuple{Float64,Float64}}
end

function calculate_contact_angles(contour::DropContour, elements::DetectedElements,
    config::AnalysisConfig, px_per_mm::Float64)
    println("\nCalculating contact angles...")

    # Calculate left contact angle
    left_angle = calculate_contact_angle_side(
        contour.left_side, elements.contact_left,
        config, px_per_mm, true
    )
    println("  ✓ Left contact angle: $(round(left_angle.angle, digits=2))°")

    # Calculate right contact angle
    right_angle = calculate_contact_angle_side(
        contour.right_side, elements.contact_right,
        config, px_per_mm, false
    )
    println("  ✓ Right contact angle: $(round(right_angle.angle, digits=2))°")

    # Average contact angle
    avg_angle = (left_angle.angle + right_angle.angle) / 2
    println("  ✓ Average contact angle: $(round(avg_angle, digits=2))°")

    return left_angle, right_angle, avg_angle
end

function calculate_contact_angle_side(contour_side, contact_point,
    config::AnalysisConfig, px_per_mm::Float64,
    is_left::Bool)
    # Select points near contact point for fitting
    fit_window_px = config.fit_window_mm * px_per_mm

    fit_points = filter(p -> abs(p[2] - contact_point[2]) < fit_window_px, contour_side)

    if length(fit_points) < config.poly_degree + 2
        @warn "Not enough points for polynomial fitting. Using more points."
        distances = [abs(p[2] - contact_point[2]) for p in contour_side]
        sorted_idx = sortperm(distances)
        fit_points = contour_side[sorted_idx[1:min(20, length(contour_side))]]
    end

    # Extract x and y coordinates
    x_fit = [p[1] for p in fit_points]
    y_fit = [p[2] for p in fit_points]

    # Fit polynomial: x = f(y) for better stability near vertical
    poly = fit(y_fit, x_fit, config.poly_degree)

    # Calculate derivative at contact point
    poly_deriv = derivative(poly)
    dx_dy = poly_deriv(contact_point[2])

    # Convert to angle
    if abs(dx_dy) < 1e-6
        tangent_slope = Inf
        angle = 90.0
    else
        tangent_slope = 1.0 / dx_dy
        angle_rad = atan(abs(tangent_slope))
        angle = rad2deg(angle_rad)
    end

    # Adjust angle based on side
    if !is_left
        if tangent_slope > 0
            angle = 180.0 - angle
        end
    else
        if tangent_slope < 0
            angle = 180.0 - angle
        end
    end

    return ContactAngle(angle, contact_point, tangent_slope, fit_points)
end

# ============================================================================
# STEP 7: SCALE CALIBRATION
# ============================================================================

function calibrate_scale(elements::DetectedElements, config::AnalysisConfig)
    if config.known_distance_mm !== nothing
        substrate_width_px = abs(elements.contact_right[1] - elements.contact_left[1])
        px_per_mm = substrate_width_px / config.known_distance_mm
        println("\nCalibration: $(round(px_per_mm, digits=2)) pixels/mm")
        return px_per_mm
    else
        println("\nNo calibration provided, using pixel units")
        return 1.0
    end
end

# ============================================================================
# STEP 8: RESULTS VISUALIZATION AND OUTPUT
# ============================================================================

struct AnalysisResults
    left_contact_angle::Float64
    right_contact_angle::Float64
    average_contact_angle::Float64
    tilt_angle::Float64
    drop_height::Float64
    drop_width::Float64
    scale_factor::Float64
    left_angle_details::ContactAngle
    right_angle_details::ContactAngle
end

function display_results(results::AnalysisResults, gray_img, binary_img,
    contour::DropContour, elements::DetectedElements,
    left_angle::ContactAngle, right_angle::ContactAngle)
    println("\n" * "="^60)
    println("SESSILE DROP ANALYSIS RESULTS")
    println("="^60)
    println("Left Contact Angle:    $(round(results.left_contact_angle, digits=2))°")
    println("Right Contact Angle:   $(round(results.right_contact_angle, digits=2))°")
    println("Average Contact Angle: $(round(results.average_contact_angle, digits=2))°")
    println("Substrate Tilt:        $(round(results.tilt_angle, digits=2))°")
    println("Drop Width:            $(round(results.drop_width, digits=2)) px")
    println("Drop Height:           $(round(results.drop_height, digits=2)) px")
    if results.scale_factor != 1.0
        println("Scale Factor:          $(round(results.scale_factor, digits=2)) px/mm")
    end
    println("="^60)

    # Create visualization
    plot_results(gray_img, binary_img, contour, elements, left_angle, right_angle)
end

function plot_results(gray_img, binary_img, contour::DropContour,
    elements::DetectedElements, left_angle::ContactAngle,
    right_angle::ContactAngle)
    # Plot 1: Original image
    p1 = plot(gray_img, title="Original Image", axis=false, legend=false)

    # Plot 2: Binary image with detections
    p2 = plot(binary_img, title="Binary + Detections", axis=false, legend=false)

    baseline_x = [p[1] for p in elements.substrate_line]
    baseline_y = [p[2] for p in elements.substrate_line]
    plot!(p2, baseline_x, baseline_y, color=:red, linewidth=2, label="Baseline")

    scatter!(p2, [elements.contact_left[1]], [elements.contact_left[2]],
        color=:green, markersize=8, label="Contact Points")
    scatter!(p2, [elements.contact_right[1]], [elements.contact_right[2]],
        color=:green, markersize=8, label="")

    # Plot 3: Contact angle measurement WITH IMAGE BACKGROUND
    p3 = plot(gray_img, title="Contact Angle Measurement",
        yflip=true, aspect_ratio=:equal, legend=:topright)

    # Overlay the contour edges
    left_x = [p[1] for p in contour.left_side]
    left_y = [p[2] for p in contour.left_side]
    right_x = [p[1] for p in contour.right_side]
    right_y = [p[2] for p in contour.right_side]

    plot!(p3, left_x, left_y, label="Left Edge", linewidth=3, color=:cyan, alpha=0.8)
    plot!(p3, right_x, right_y, label="Right Edge", linewidth=3, color=:yellow, alpha=0.8)

    plot!(p3, baseline_x, baseline_y, color=:red, linewidth=3,
        linestyle=:dash, label="Substrate", alpha=0.9)

    plot_tangent_line!(p3, left_angle, 50, :cyan)
    plot_tangent_line!(p3, right_angle, 50, :yellow)

    # Add contact points
    scatter!(p3, [elements.contact_left[1]], [elements.contact_left[2]],
        color=:lime, markersize=10, markerstrokewidth=2, markerstrokecolor=:white,
        label="Contact Points", alpha=0.9)
    scatter!(p3, [elements.contact_right[1]], [elements.contact_right[2]],
        color=:lime, markersize=10, markerstrokewidth=2, markerstrokecolor=:white,
        label="", alpha=0.9)

    # Add angle annotations with background boxes for better visibility
    annotate!(p3, elements.contact_left[1] - 50, elements.contact_left[2] - 30,
        text("θ_L = $(round(left_angle.angle, digits=1))°", 11, :left, :white, :bold))
    annotate!(p3, elements.contact_right[1] + 50, elements.contact_right[2] - 30,
        text("θ_R = $(round(right_angle.angle, digits=1))°", 11, :right, :white, :bold))

    plot(p1, p2, p3, layout=(3, 1), size=(800, 1500))
end

function plot_tangent_line!(p, contact_angle::ContactAngle, length_px, color)
    x0, y0 = contact_angle.contact_point

    if isinf(contact_angle.tangent_slope)
        x_line = [x0, x0]
        y_line = [y0 - length_px, y0]
    else
        dx = length_px / sqrt(1 + contact_angle.tangent_slope^2)
        dy = dx * contact_angle.tangent_slope

        x_line = [x0 - dx, x0]
        y_line = [y0 - dy, y0]
    end

    plot!(p, x_line, y_line, color=color, linewidth=3,
        linestyle=:dot, label="", alpha=0.7)
end

# ============================================================================
# MAIN PIPELINE
# ============================================================================

function analyze_sessile_drop(image_path::String, config::AnalysisConfig=default_config())
    println("\n" * "="^60)
    println("SESSILE DROP CONTACT ANGLE ANALYSIS")
    println("="^60)

    try
        # Step 1: Load image
        img = load_image(image_path)

        # Step 2: Preprocess
        gray_img, binary_img, cleaned_img = preprocess_image(img, config)

        # Step 3: Detect elements
        elements = detect_elements(cleaned_img, config)

        # Step 4: Correct tilt
        if abs(elements.tilt_angle) > 0.5
            corrected_elements, rotated_coords = correct_tilt(elements, size(img))
            contour = extract_contour(rotated_coords, corrected_elements, config)
        else
            contour = extract_contour(elements.drop_region, elements, config)
            corrected_elements = elements
        end

        # Step 5: Calibrate scale
        px_per_mm = calibrate_scale(elements, config)

        # Step 6: Calculate contact angles
        left_angle, right_angle, avg_angle = calculate_contact_angles(
            contour, corrected_elements, config, px_per_mm
        )

        # Calculate drop dimensions
        drop_width = abs(elements.contact_right[1] - elements.contact_left[1])
        drop_height = abs(elements.drop_apex[2] -
                          mean([elements.contact_left[2], elements.contact_right[2]]))

        # Step 7: Compile results
        results = AnalysisResults(
            left_angle.angle,
            right_angle.angle,
            avg_angle,
            elements.tilt_angle,
            drop_height,
            drop_width,
            px_per_mm,
            left_angle,
            right_angle
        )

        # Step 8: Display results
        display_results(results, gray_img, binary_img, contour,
            elements, left_angle, right_angle)

        return results, gray_img, binary_img, contour, elements, left_angle, right_angle

    catch e
        println("\n❌ Error during analysis: $e")
        rethrow(e)
    end
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

function example_usage()
    println("\n" * "="^60)
    println("IMPROVED SESSILE DROP ANALYZER - USAGE GUIDE")
    println("="^60)

    println("\n📌 FOR DARK DROPS ON LIGHT BACKGROUND (your case!):")
    println("="^60)
    println("config = AnalysisConfig(")
    println("    nothing,     # No calibration")
    println("    2.0,         # Blur sigma")
    println("    :manual,     # Manual threshold")
    println("    0.4,         # Try values 0.3-0.6")
    println("    :ransac,     # Robust baseline detection")
    println("    3,           # Cubic polynomial")
    println("    0.5,         # Fit window (mm)")
    println("    500,         # Lower min area for small drops")
    println("    true,        # ← INVERT for dark drops!")
    println("    1.0          # ← Strong center preference")
    println(")")
    println("results = analyze_sessile_drop(\"your_image.png\", config)")

    println("\n📌 FOR LIGHT DROPS ON DARK BACKGROUND:")
    println("="^60)
    println("config = AnalysisConfig(")
    println("    nothing, 2.0, :otsu, 0.5, :ransac, 3, 0.5, 1000,")
    println("    false,       # Don't invert")
    println("    0.8          # Center preference")
    println(")")

    println("\n📌 WITH SCALE CALIBRATION:")
    println("="^60)
    println("config = AnalysisConfig(")
    println("    5.0,         # Known substrate width = 5 mm")
    println("    2.0, :otsu, 0.5, :ransac, 3, 0.5, 1000,")
    println("    true, 1.0    # Dark drop, strong center focus")
    println(")")

    println("\n📌 TROUBLESHOOTING:")
    println("="^60)
    println("If wrong component is selected:")
    println("  1. Adjust manual_threshold (try 0.3, 0.35, 0.4, 0.45, 0.5)")
    println("  2. Increase center_preference (0.5 → 1.0 → 1.5)")
    println("  3. Toggle invert_binary (true ↔ false)")
    println("  4. Lower min_drop_area for small drops (1000 → 500 → 250)")
    println("\nCheck 'Binary + Detections' plot:")
    println("  • Drop should be WHITE on dark background")
    println("  • If inverted, set invert_binary=true")

    println("\n" * "="^60)
end

# Show examples
example_usage()

println("\n✅ Script loaded! Ready to analyze your sessile drop images.")
println("📝 Quick start for your image:")
println("   filename = \"prueba sesil 2.png\"")
println("   image_path = joinpath(@__DIR__, \"data\", \"samples\", filename)")
println("   config = AnalysisConfig(nothing, 2.0, :manual, 0.4, :ransac, 3, 0.5, 500, true, 1.0)")
println("   results = analyze_sessile_drop(image_path, config)")
# Descomentar y modificar con tu ruta de imagen:
#filename = "gota depositada 1.png"
filename = "prueba sesil 2.png"
image_path = joinpath(@__DIR__, "data", "samples", filename)
# Basic analysis
results = analyze_sessile_drop(image_path)

# With scale calibration (if you know a reference distance)
config = AnalysisConfig(
    nothing,
    2.0,         # Blur sigma
    :manual,     # Manual threshold
    0.2,         # Try values 0.3-0.6
    :hough,     # Robust baseline detection
    3,           # Cubic polynomial
    0.5,         # Fit window (mm)
    500,         # Lower min area for small drops
    true,        # ← INVERT for dark drops!
    1.0          # ← Strong center preference
)
results, gray_img, binary_img, contour, elements, left_angle, right_angle = analyze_sessile_drop(image_path, config)

# Access results
println("Average contact angle: $(results.average_contact_angle)°")
println("Left angle: $(results.left_contact_angle)°")
println("Right angle: $(results.right_contact_angle)°")

# Plot results again if desired
plot_results(gray_img, binary_img, contour, elements, left_angle, right_angle)