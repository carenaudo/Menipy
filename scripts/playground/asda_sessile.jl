"""
Sessile Drop Contact Angle Analysis Script
Measures contact angles from sessile drop images with tilt correction
Julia 1.11+
"""

using Images, ImageFiltering, ImageMorphology, ImageSegmentation
using Polynomials, Statistics, LinearAlgebra
using Plots

# ============================================================================
# CONFIGURATION
# ============================================================================

struct AnalysisConfig
    known_distance_mm::Union{Float64,Nothing}  # Known distance for calibration (optional)
    blur_sigma::Float64                          # Gaussian blur sigma
    threshold_method::Symbol                     # :otsu, :yen, or :manual
    manual_threshold::Float64                    # Used if threshold_method == :manual
    baseline_detection::Symbol                   # :ransac, :hough, :bottom
    poly_degree::Int                             # Polynomial degree for tangent fitting
    fit_window_mm::Float64                       # Window around contact point for fitting (mm)
    min_drop_area::Int                           # Minimum drop area in pixels
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
    1000       # Minimum 1000 pixels for drop
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
# STEP 2: IMAGE PREPROCESSING
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
    println("  ✓ Binarized image")

    # Morphological operations to clean up
    cleaned = clean_binary(binary)
    println("  ✓ Cleaned binary image")

    return gray_img, binary, cleaned
end

function binarize_image(img, config::AnalysisConfig)
    if config.threshold_method == :otsu
        threshold = otsu_threshold(img)
        return img .> threshold
    elseif config.threshold_method == :yen
        # Simple Yen-like threshold
        threshold = quantile(vec(img), 0.5)
        return img .> threshold
    else
        return img .> config.manual_threshold
    end
end

function clean_binary(binary)
    # Remove small objects and fill holes
    cleaned = copy(binary)
    cleaned = opening(cleaned, strel_diamond((3, 3)))
    cleaned = closing(cleaned, strel_diamond((5, 5)))
    return cleaned
end

# ============================================================================
# STEP 3: ELEMENT DETECTION
# ============================================================================

struct DetectedElements
    drop_region::Vector{CartesianIndex{2}}
    substrate_line::Vector{Tuple{Float64,Float64}}  # (x, y) points
    contact_left::Tuple{Float64,Float64}
    contact_right::Tuple{Float64,Float64}
    tilt_angle::Float64  # Angle in degrees
    drop_apex::Tuple{Float64,Float64}
end

function detect_elements(binary_img, config::AnalysisConfig)
    println("\nDetecting elements...")

    # Find drop region
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
    # Find connected components
    labels = label_components(binary_img)

    # Find components and their sizes
    # component_lengths returns an OffsetArray where index = label number
    component_sizes = component_lengths(labels)

    # Filter out background (label 0) and small components
    valid_labels = Int[]
    for label in keys(component_sizes)
        if label != 0 && component_sizes[label] >= config.min_drop_area
            push!(valid_labels, label)
        end
    end

    if isempty(valid_labels)
        error("No drop detected. Try adjusting threshold or min_drop_area.")
    end

    # Find largest component (assumed to be the drop)
    drop_label = valid_labels[argmax([component_sizes[l] for l in valid_labels])]
    drop_mask = labels .== drop_label
    drop_coords = findall(drop_mask)

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
    points_matrix = hcat([[p[1], p[2]] for p in bottom_points]...)'

    # Simple RANSAC implementation for line fitting
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
    # Placeholder for Hough transform method
    # Would use ImageFeatures.jl if available
    return detect_substrate_ransac(binary_img, drop_mask)
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

    # Create corrected elements (drop_region as empty for now, we only need coords)
    corrected = DetectedElements(
        CartesianIndex{2}[],  # Will use coordinates directly
        rotated_baseline,
        rotated_left,
        rotated_right,
        0.0,  # Tilt corrected
        rotated_apex
    )

    println("  ✓ Tilt corrected (was $(round(elements.tilt_angle, digits=2))°)")

    return corrected, rotated_drop
end

function rotate_point(x, y, cx, cy, angle)
    # Rotate point (x, y) around center (cx, cy) by angle
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
    left_side::Vector{Tuple{Float64,Float64}}   # (x, y) points
    right_side::Vector{Tuple{Float64,Float64}}  # (x, y) points
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

    # Extract edges (one point per y-level)
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
        # Take closest points
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

    # Convert to angle: tangent slope dy/dx = 1/(dx/dy)
    if abs(dx_dy) < 1e-6
        tangent_slope = Inf
        angle = 90.0
    else
        tangent_slope = 1.0 / dx_dy
        angle_rad = atan(abs(tangent_slope))
        angle = rad2deg(angle_rad)
    end

    # For left side, angle is measured from left
    # For right side, angle is measured from right
    # Both should give contact angle with substrate
    if !is_left
        # Right side: if slope is negative, angle is correct
        # if slope is positive, angle = 180 - angle
        if tangent_slope > 0
            angle = 180.0 - angle
        end
    else
        # Left side: if slope is positive, angle is correct
        # if slope is negative, angle = 180 - angle
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
        # Use known distance (e.g., substrate width)
        substrate_width_px = abs(elements.contact_right[1] - elements.contact_left[1])
        px_per_mm = substrate_width_px / config.known_distance_mm
        println("\nCalibration: $(round(px_per_mm, digits=2)) pixels/mm")
        return px_per_mm
    else
        # Use pixels as units
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

    # Add baseline
    baseline_x = [p[1] for p in elements.substrate_line]
    baseline_y = [p[2] for p in elements.substrate_line]
    plot!(p2, baseline_x, baseline_y, color=:red, linewidth=2, label="Baseline")

    # Add contact points
    scatter!(p2, [elements.contact_left[1]], [elements.contact_left[2]],
        color=:green, markersize=8, label="Contact Points")
    scatter!(p2, [elements.contact_right[1]], [elements.contact_right[2]],
        color=:green, markersize=8, label="")

    # Plot 3: Contour with tangent lines
    p3 = plot(title="Contact Angle Measurement", xlabel="x (px)", ylabel="y (px)",
        legend=:best, yflip=true, aspect_ratio=:equal)

    # Plot contours
    left_x = [p[1] for p in contour.left_side]
    left_y = [p[2] for p in contour.left_side]
    right_x = [p[1] for p in contour.right_side]
    right_y = [p[2] for p in contour.right_side]

    plot!(p3, left_x, left_y, label="Left Edge", linewidth=2, color=:blue)
    plot!(p3, right_x, right_y, label="Right Edge", linewidth=2, color=:red)

    # Plot baseline
    plot!(p3, baseline_x, baseline_y, color=:black, linewidth=2,
        linestyle=:dash, label="Substrate")

    # Plot tangent lines at contact points
    plot_tangent_line!(p3, left_angle, 50, :blue)
    plot_tangent_line!(p3, right_angle, 50, :red)

    # Add angle annotations
    annotate!(p3, elements.contact_left[1] - 50, elements.contact_left[2] - 30,
        text("θ_L = $(round(left_angle.angle, digits=1))°", 10, :left))
    annotate!(p3, elements.contact_right[1] + 50, elements.contact_right[2] - 30,
        text("θ_R = $(round(right_angle.angle, digits=1))°", 10, :right))

    plot(p1, p2, p3, layout=(1, 3), size=(1500, 400))
end

function plot_tangent_line!(p, contact_angle::ContactAngle, length_px, color)
    x0, y0 = contact_angle.contact_point

    # Calculate tangent line endpoints
    if isinf(contact_angle.tangent_slope)
        # Vertical line
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

        # Step 4: Correct tilt (optional, but recommended)
        if abs(elements.tilt_angle) > 0.5
            corrected_elements, rotated_coords = correct_tilt(elements, size(img))
            # Use original coordinates for visualization, corrected for calculation
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

        # Return results and data needed for plotting
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
    println("USAGE EXAMPLES")
    println("="^60)

    # Example 1: Basic analysis
    println("\n1. Basic analysis (no scale calibration):")
    println("   results = analyze_sessile_drop(\"sessile_drop.jpg\")")

    # Example 2: With scale calibration
    println("\n2. With known distance for calibration:")
    println("   config = AnalysisConfig(")
    println("       5.0,        # Known substrate width = 5 mm")
    println("       2.0,        # Blur sigma")
    println("       :otsu,      # Threshold method")
    println("       0.5,        # Manual threshold")
    println("       :ransac,    # Baseline detection method")
    println("       3,          # Polynomial degree")
    println("       0.5,        # Fit window (mm)")
    println("       1000        # Min drop area (px)")
    println("   )")
    println("   results = analyze_sessile_drop(\"image.jpg\", config)")

    # Example 3: Advanced customization
    println("\n3. For noisy images or tilted substrate:")
    println("   config = AnalysisConfig(")
    println("       nothing,    # No calibration")
    println("       3.0,        # Higher blur for noisy images")
    println("       :manual,    # Manual threshold")
    println("       0.45,       # Threshold value")
    println("       :ransac,    # RANSAC for robust baseline")
    println("       4,          # Higher order polynomial")
    println("       0.3,        # Smaller fit window")
    println("       500         # Lower min area threshold")
    println("   )")

    println("\n" * "="^60)
    println("\nIMPORTANT:")
    println("• Image should show clear drop on substrate")
    println("• Drop should be well-separated from other objects")
    println("• For best results, substrate should be horizontal or nearly horizontal")
    println("• RANSAC baseline detection is robust to tilt and outliers")
    println("="^60)
end

# Run example
example_usage()
# Descomentar y modificar con tu ruta de imagen:
filename = "gota depositada 1.png"
image_path = joinpath(@__DIR__, "data", "samples", filename)
# Basic analysis
results = analyze_sessile_drop(image_path)

# With scale calibration (if you know a reference distance)
config = AnalysisConfig(
    5.0,       # Known substrate width = 5 mm
    1.5,       # Blur sigma
    :otsu,     # Threshold method
    0.3,       # Manual threshold
    :ransac,   # RANSAC baseline detection (most robust)
    3,         # Cubic polynomial for tangent
    0.5,       # Fit 0.5mm around contact point
    1000       # Minimum drop area in pixels
)
results, gray_img, binary_img, contour, elements, left_angle, right_angle = analyze_sessile_drop(image_path, config)

# Access results
println("Average contact angle: $(results.average_contact_angle)°")
println("Left angle: $(results.left_contact_angle)°")
println("Right angle: $(results.right_contact_angle)°")

# Plot results again if desired
plot_results(gray_img, binary_img, contour, elements, left_angle, right_angle)