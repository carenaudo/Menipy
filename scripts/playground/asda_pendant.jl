"""
Pendant Drop Shape Analysis Script
Calculates surface tension from pendant drop images using Young-Laplace equation
Julia 1.11+
"""

using Images, ImageFiltering, ImageMorphology, ImageSegmentation
using DifferentialEquations, Optimization, OptimizationOptimJL
using Plots, Statistics, LinearAlgebra

# ============================================================================
# CONFIGURATION
# ============================================================================

struct AnalysisConfig
    needle_diameter_mm::Float64  # Known needle outer diameter in mm
    density_diff::Float64        # Density difference (ρ_liquid - ρ_gas) in kg/m³
    gravity::Float64             # Gravitational acceleration in m/s²
    blur_sigma::Float64          # Gaussian blur sigma
    threshold_method::Symbol     # :otsu, :yen, or :manual
    manual_threshold::Float64    # Used if threshold_method == :manual
end

# Default configuration
default_config() = AnalysisConfig(
    1.65,      # 1.65 mm needle diameter (example)
    1000.0,    # Water-air: ~1000 kg/m³
    9.81,      # Standard gravity
    2.0,       # Blur sigma
    :otsu,     # Automatic thresholding
    0.5        # Manual threshold (0-1)
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
        return img .> otsu_threshold(img)
    elseif config.threshold_method == :yen
        threshold = sum(img .* (1:length(img))) / sum(img)  # Simple Yen-like
        return img .> threshold
    else
        return img .> config.manual_threshold
    end
end

function clean_binary(binary)
    # Remove small objects
    cleaned = copy(binary)
    cleaned = opening(cleaned, strel_diamond((3, 3)))
    cleaned = closing(cleaned, strel_diamond((3, 3)))
    return cleaned
end

# ============================================================================
# STEP 3: ELEMENT DETECTION
# ============================================================================

struct DetectedElements
    needle_region::CartesianIndices
    drop_region::CartesianIndices
    contact_point::Tuple{Int,Int}
    needle_width_px::Float64
end

function detect_elements(binary_img)
    println("\nDetecting elements...")

    # Find connected components
    labels = label_components(binary_img)

    # Find the largest component (assumed to be the drop)
    component_sizes = component_lengths(labels)[2:end]
    if isempty(component_sizes)
        error("No objects detected in image")
    end

    drop_label = argmax(component_sizes) + 1
    drop_mask = labels .== drop_label

    # Find drop region
    drop_coords = findall(drop_mask)
    println("  ✓ Drop detected ($(length(drop_coords)) pixels)")

    # Detect needle (top portion of drop with consistent width)
    needle_region, needle_width = detect_needle(drop_mask)
    println("  ✓ Needle detected (width: $(round(needle_width, digits=2)) px)")

    # Find contact point (transition from needle to drop)
    contact_point = find_contact_point(drop_mask, needle_region)
    println("  ✓ Contact point: $contact_point")

    return DetectedElements(needle_region, drop_coords, contact_point, needle_width)
end

function detect_needle(drop_mask)
    # Find top region with relatively constant width
    rows, cols = size(drop_mask)
    top_region = drop_mask[1:min(rows ÷ 3, 100), :]

    widths = [count(top_region[r, :]) for r in 1:size(top_region, 1)]
    avg_width = median(filter(x -> x > 0, widths))

    # Find consistent width region
    needle_rows = findall(abs.(widths .- avg_width) .< avg_width * 0.1)
    if isempty(needle_rows)
        needle_rows = 1:10
    end

    needle_region = CartesianIndices((needle_rows[1]:needle_rows[end], 1:cols))

    return needle_region, avg_width
end

function find_contact_point(drop_mask, needle_region)
    # Contact point is where needle ends
    needle_bottom = maximum(needle_region.indices[1])

    # Find center column
    center_col = size(drop_mask, 2) ÷ 2

    return (needle_bottom, center_col)
end

# ============================================================================
# STEP 4: PIXEL TO MM CALIBRATION
# ============================================================================

function calibrate_scale(elements::DetectedElements, config::AnalysisConfig)
    px_per_mm = elements.needle_width_px / config.needle_diameter_mm
    println("\nCalibration: $(round(px_per_mm, digits=2)) pixels/mm")
    return px_per_mm
end

# ============================================================================
# STEP 5: DROP CONTOUR DETECTION
# ============================================================================

struct DropContour
    left_x::Vector{Float64}
    left_y::Vector{Float64}
    right_x::Vector{Float64}
    right_y::Vector{Float64}
    apex_y::Float64
end

function extract_contour(binary_img, contact_point, px_per_mm)
    println("\nExtracting drop contour...")

    contact_row, contact_col = contact_point

    # Extract left and right edges below contact point
    left_edge, right_edge = extract_edges(binary_img, contact_row)

    # Convert to mm coordinates (origin at apex)
    apex_row = maximum(left_edge[:, 1])

    left_x = (left_edge[:, 2] .- contact_col) ./ px_per_mm
    left_y = (apex_row .- left_edge[:, 1]) ./ px_per_mm

    right_x = (right_edge[:, 2] .- contact_col) ./ px_per_mm
    right_y = (apex_row .- right_edge[:, 1]) ./ px_per_mm

    println("  ✓ Extracted contour points: $(length(left_x)) (left), $(length(right_x)) (right)")

    return DropContour(left_x, left_y, right_x, right_y, 0.0)
end

function extract_edges(binary_img, start_row)
    rows, cols = size(binary_img)
    center_col = cols ÷ 2

    left_edge = Tuple{Int,Int}[]
    right_edge = Tuple{Int,Int}[]

    for r in start_row:rows
        row = binary_img[r, :]
        drop_cols = findall(row)

        if !isempty(drop_cols)
            push!(left_edge, (r, minimum(drop_cols)))
            push!(right_edge, (r, maximum(drop_cols)))
        end
    end

    return hcat([[p[1], p[2]] for p in left_edge]...)',
    hcat([[p[1], p[2]] for p in right_edge]...)'
end

# ============================================================================
# STEP 6: YOUNG-LAPLACE EQUATION SOLVING FOR SURFACE TENSION
# ============================================================================

"""
Young-Laplace equation for pendant drop:
Solves for surface tension by fitting theoretical profile to experimental contour
"""
function calculate_surface_tension(contour::DropContour, config::AnalysisConfig)
    println("\nCalculating surface tension...")

    # Use the right half of drop (typically cleaner)
    x_exp = contour.right_x[contour.right_y.>0]
    z_exp = contour.right_y[contour.right_y.>0]

    # Sort by z coordinate
    sort_idx = sortperm(z_exp)
    x_exp = x_exp[sort_idx]
    z_exp = z_exp[sort_idx]

    # Initial guess for surface tension (water-air ~ 0.072 N/m)
    γ_initial = 0.072

    # Optimize to find best-fit surface tension
    result = fit_young_laplace(x_exp, z_exp, γ_initial, config)

    γ_fit = result.u[1]
    println("  ✓ Surface tension: $(round(γ_fit * 1000, digits=2)) mN/m")

    return γ_fit, result
end

function fit_young_laplace(x_exp, z_exp, γ_init, config::AnalysisConfig)
    # Define Young-Laplace ODE system
    function young_laplace!(du, u, p, s)
        x, z, θ = u
        γ = p[1]

        # Bond number
        β = config.density_diff * config.gravity / γ

        # Young-Laplace equations in arc-length form
        du[1] = cos(θ)  # dx/ds
        du[2] = sin(θ)  # dz/ds
        du[3] = β * z - sin(θ) / max(x, 1e-6)  # dθ/ds
    end

    # Objective function: minimize difference between theoretical and experimental
    function objective(γ, _)
        u0 = [1e-6, 0.0, π / 2]  # Initial conditions at apex
        tspan = (0.0, maximum(z_exp))
        prob = ODEProblem(young_laplace!, u0, tspan, [γ])

        sol = solve(prob, Tsit5(), saveat=z_exp, reltol=1e-6)

        if sol.retcode != :Success
            return [1e10]
        end

        # Extract x values from solution
        x_theory = [sol.u[i][1] for i in 1:length(sol.u)]

        # Calculate residual
        residual = sum((x_exp .- x_theory) .^ 2)
        return [residual]
    end

    # Optimize
    optf = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, [γ_init], lb=[0.001], ub=[0.2])
    sol = solve(prob, LBFGS())

    return sol
end

# ============================================================================
# STEP 7: RESULTS VISUALIZATION AND OUTPUT
# ============================================================================

struct AnalysisResults
    surface_tension::Float64  # N/m
    surface_tension_mn::Float64  # mN/m
    needle_diameter::Float64  # mm
    scale_factor::Float64  # px/mm
    contour::DropContour
    fit_quality::Float64
end

function display_results(results::AnalysisResults, gray_img, binary_img, contour)
    println("\n" * "="^60)
    println("ANALYSIS RESULTS")
    println("="^60)
    println("Surface Tension: $(round(results.surface_tension_mn, digits=2)) mN/m")
    println("                 $(round(results.surface_tension, digits=5)) N/m")
    println("Scale Factor:    $(round(results.scale_factor, digits=2)) px/mm")
    println("Needle Diameter: $(results.needle_diameter) mm")
    println("="^60)

    # Create visualization
    plot_results(gray_img, binary_img, contour)
end

function plot_results(gray_img, binary_img, contour::DropContour)
    p1 = plot(gray_img, title="Original Image", axis=false)
    p2 = plot(binary_img, title="Binary Image", axis=false)

    p3 = plot(title="Drop Contour", xlabel="x (mm)", ylabel="y (mm)",
        legend=:topright, aspect_ratio=:equal)
    plot!(p3, contour.left_x, contour.left_y, label="Left Edge", lw=2)
    plot!(p3, contour.right_x, contour.right_y, label="Right Edge", lw=2)

    plot(p1, p2, p3, layout=(1, 3), size=(1400, 400))
end

# ============================================================================
# MAIN PIPELINE
# ============================================================================

function analyze_pendant_drop(image_path::String, config::AnalysisConfig=default_config())
    println("\n" * "="^60)
    println("PENDANT DROP SHAPE ANALYSIS")
    println("="^60)

    try
        # Step 1: Load image
        img = load_image(image_path)

        # Step 2: Preprocess
        gray_img, binary_img, cleaned_img = preprocess_image(img, config)

        # Step 3: Detect elements
        elements = detect_elements(cleaned_img)

        # Step 4: Calibrate scale
        px_per_mm = calibrate_scale(elements, config)

        # Step 5: Extract contour
        contour = extract_contour(cleaned_img, elements.contact_point, px_per_mm)

        # Step 6: Calculate surface tension
        γ, fit_result = calculate_surface_tension(contour, config)

        # Step 7: Compile and display results
        results = AnalysisResults(
            γ, γ * 1000, config.needle_diameter_mm,
            px_per_mm, contour, fit_result.objective
        )

        display_results(results, gray_img, binary_img, contour)

        return results

    catch e
        println("\n❌ Error during analysis: $e")
        rethrow(e)
    end
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

function example_usage()
    # Configure analysis parameters
    config = AnalysisConfig(
        1.65,      # Needle diameter in mm
        1000.0,    # Density difference (water-air)
        9.81,      # Gravity
        2.0,       # Blur sigma
        :otsu,     # Thresholding method
        0.5        # Manual threshold (if used)
    )

    # Run analysis
    # Descomentar y modificar con tu ruta de imagen:
    filename = "gota pendiente 1.png"
    image_path = joinpath(@__DIR__, "data", "samples", filename)
    results = analyze_pendant_drop(image_path, config)

    println("\nTo use this script:")
    println("  1. Load an image: analyze_pendant_drop(\"image.jpg\")")
    println("  2. Or customize: analyze_pendant_drop(\"image.jpg\", custom_config)")
    println("\nMake sure the image shows a clear pendant drop with visible needle.")
end

# Run example
example_usage()