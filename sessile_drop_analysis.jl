"""
Sessile Drop Detection and Contact Angle Analysis
==================================================
Julia port of sessile_detections.py using the Images.jl ecosystem.
This module provides drop detection, geometry extraction, and contact angle
calculation using multiple methods (spherical cap, tangent, ellipse, Young-Laplace).

Usage:
    julia> include("sessile_drop_analysis.jl")
    julia> using .SessileDropAnalysis
    julia> det, angles = run_analysis("./data/samples/prueba sesil 2.png")
"""
module SessileDropAnalysis

using Images
using ImageContrastAdjustment
using ImageMorphology
using ImageFiltering
using ImageBinarization
using Statistics
using LinearAlgebra
using Printf
using Plots

# ODE solving for Young-Laplace
using OrdinaryDiffEq

export sessile_drop_adaptive, compute_contact_angles_from_detection
export contact_angle_from_apex, fit_spherical_cap, fit_elliptical
export fit_young_laplace, fit_young_laplace_unknown_sigma
export compare_yl_methods, run_analysis, DropDetectionResult

# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================

"""Result structure from sessile drop detection."""
struct DropDetectionResult
    substrate_y::Float64
    cp_left::Tuple{Float64,Float64}
    cp_right::Tuple{Float64,Float64}
    apex::Tuple{Float64,Float64}
    height_px::Float64
    base_width_px::Float64
    dome_points_array::Matrix{Float64}  # Nx2 matrix
    full_contour::Matrix{Float64}       # Nx2 matrix
    roi_coords::NTuple{4,Int}          # (x1, y1, x2, y2)
    debug_images::Dict{String,Any}     # Intermediate stages (optional)
end

# ==============================================================================
# 2. IMAGE PROCESSING UTILITIES
# ==============================================================================

"""
    trace_boundary(mask::BitMatrix) -> Matrix{Float64}

Moore-Neighbor tracing to extract ordered boundary points from a binary mask.
Returns Nx2 matrix of (x, y) coordinates.
"""
function trace_boundary(mask::BitMatrix)
    rows, cols = size(mask)
    start_idx = findfirst(mask)
    isnothing(start_idx) && return Matrix{Float64}(undef, 0, 2)

    start_r, start_c = start_idx[1], start_idx[2]
    boundary = [(start_r, start_c)]

    # Clockwise Moore neighborhood offsets (row, col)
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    current = (start_r, start_c)
    backtrack = 7
    max_iter = rows * cols

    for _ in 1:max_iter
        found_next = false
        for i in 0:7
            idx = mod(backtrack + i, 8) + 1
            dr, dc = offsets[idx]
            nr, nc = current[1] + dr, current[2] + dc

            if 1 <= nr <= rows && 1 <= nc <= cols && mask[nr, nc]
                if (nr, nc) == (start_r, start_c) && length(boundary) > 2
                    found_next = false
                    break
                end
                current = (nr, nc)
                push!(boundary, current)
                backtrack = mod(idx + 4, 8)
                found_next = true
                break
            end
        end
        !found_next && break
    end

    # Convert (row, col) to (x, y): x=col, y=row
    result = Matrix{Float64}(undef, length(boundary), 2)
    for (i, (r, c)) in enumerate(boundary)
        result[i, 1] = Float64(c)
        result[i, 2] = Float64(r)
    end
    return result
end

"""Find horizontal substrate line in an image strip using gradient analysis."""
function find_horizon_median(strip_gray)
    h, w = size(strip_gray)
    min_limit = max(1, round(Int, h * 0.05))
    max_limit = min(h - 1, round(Int, h * 0.95))

    detected_ys = Float64[]
    for c in 1:w
        col_data = Float64.(strip_gray[:, c])
        grad = diff(col_data)
        valid_range = min_limit:min(max_limit, length(grad))
        isempty(valid_range) && continue
        valid_grad = grad[valid_range]
        isempty(valid_grad) && continue
        _, best_local_idx = findmin(valid_grad)
        push!(detected_ys, Float64(valid_range[best_local_idx]))
    end
    isempty(detected_ys) ? nothing : median(detected_ys)
end

"""Gift wrapping algorithm for convex hull. Input is Nx2 matrix."""
function convex_hull_gift_wrap(points::Matrix{Float64})
    n = size(points, 1)
    n < 3 && return points

    start_idx = argmin(points[:, 1])
    hull_indices = [start_idx]
    current = start_idx

    while true
        next_pt = current == 1 ? 2 : 1
        for i in 1:n
            i == current && continue
            p1 = points[current, :]
            p2 = points[next_pt, :]
            p3 = points[i, :]
            cross = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
            (next_pt == current || cross > 0) && (next_pt = i)
        end
        next_pt == start_idx && break
        push!(hull_indices, next_pt)
        current = next_pt
        length(hull_indices) > n && break
    end
    return points[hull_indices, :]
end

"""Fit an ellipse to 2D points using direct least squares (robust SVD approach)."""
function fit_ellipse(points::Matrix{Float64})
    n = size(points, 1)
    n < 6 && return nothing

    x, y = points[:, 1], points[:, 2]

    # Normalize for numerical stability
    mx, my = mean(x), mean(y)
    scale = max(std(x), std(y), 1.0)

    xn = (x .- mx) ./ scale
    yn = (y .- my) ./ scale

    # Design matrix for conic: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    # Direct least squares with ellipse constraint 4AC - B² = 1
    D1 = hcat(xn .^ 2, xn .* yn, yn .^ 2)
    D2 = hcat(xn, yn, ones(n))

    S1 = D1' * D1
    S2 = D1' * D2
    S3 = D2' * D2

    # Constraint: 4AC - B² > 0 for ellipse
    C1 = [0.0 0.0 2.0; 0.0 -1.0 0.0; 2.0 0.0 0.0]

    try
        T = -inv(S3) * S2'
        M = S1 + S2 * T
        M = inv(C1) * M

        eigres = eigen(M)
        cond = 4 .* eigres.vectors[1, :] .* eigres.vectors[3, :] .- eigres.vectors[2, :] .^ 2
        valid_idx = findfirst(c -> c > 0, cond)
        isnothing(valid_idx) && return nothing

        a1 = eigres.vectors[:, valid_idx]
        a2 = T * a1
        coeffs = vcat(a1, a2)

        A, B, C, D_c, E, F = coeffs

        # Scale back to original coordinates
        A_o = A / scale^2
        B_o = B / scale^2
        C_o = C / scale^2
        D_o = D_c / scale - 2A * mx / scale^2 - B * my / scale^2
        E_o = E / scale - 2C * my / scale^2 - B * mx / scale^2

        denom = B_o^2 - 4A_o * C_o
        abs(denom) < 1e-12 && return nothing

        center_x = (2C_o * D_o - B_o * E_o) / denom
        center_y = (2A_o * E_o - B_o * D_o) / denom

        # Compute semi-axes from matrix form [[A, B/2], [B/2, C]]
        M_axes = [A_o B_o/2; B_o/2 C_o]
        eig_vals = eigvals(M_axes)
        all(e -> e > 0, eig_vals) || return nothing

        semi_major = 1.0 / sqrt(minimum(abs.(eig_vals)))
        semi_minor = 1.0 / sqrt(maximum(abs.(eig_vals)))

        # Validate
        (semi_major > 1e6 || semi_minor > 1e6 || semi_major < 1 || semi_minor < 1) && return nothing

        return (center_x=center_x, center_y=center_y,
            semi_major=semi_major, semi_minor=semi_minor)
    catch
        return nothing
    end
end

# ==============================================================================
# 3. MAIN DETECTION PIPELINE
# ==============================================================================

"""
    sessile_drop_adaptive(image_path::String) -> Union{DropDetectionResult, Nothing}

Main detection pipeline for sessile drop analysis.
"""
function sessile_drop_adaptive(image_path::String; capture_debug=false)
    # Debug storage
    debug_imgs = Dict{String,Any}()

    # Load image
    img = try
        load(image_path)
    catch e
        println("Error: Cannot load image: $e")
        return nothing
    end

    img_gray = Gray.(img)
    capture_debug && (debug_imgs["1_grayscale"] = img_gray)
    h, w = size(img_gray)

    # CLAHE enhancement
    img_enhanced = adjust_histogram(img_gray,
        AdaptiveEqualization(nbins=256, rblocks=8, cblocks=8, clip=0.02))
    capture_debug && (debug_imgs["2_clahe"] = img_enhanced)

    # Substrate detection
    margin_px = min(50, w ÷ 10)
    y_left = find_horizon_median(img_enhanced[:, 1:margin_px])
    y_right = find_horizon_median(img_enhanced[:, (w-margin_px+1):w])

    substrate_y = if isnothing(y_left) || isnothing(y_right)
        Float64(h * 0.8)
    else
        (y_left + y_right) / 2.0
    end
    substrate_y < h * 0.2 && (substrate_y = Float64(h * 0.8))

    # Adaptive thresholding
    # Python uses C=2 (subtract 2). Julia uses percentage (multiplicative).
    # To match Python's sensitivity (T ~ mean - 2), we need a small percentage.
    # 2/255 approx 0.8%. So percentage=1 is safer to capture lighter edges.
    img_blur = imfilter(img_enhanced, Kernel.gaussian(2))
    img_binary = binarize(img_blur, AdaptiveThreshold(window_size=21, percentage=1)) .> 0.5
    capture_debug && (debug_imgs["3_threshold"] = copy(img_binary))

    # Invert if background is white
    mean(img_binary[1:min(10, h), 1:min(10, w)]) > 0.5 && (img_binary = .!img_binary)

    # Mask below substrate
    sub_y_int = round(Int, substrate_y)
    sub_y_int > 0 && sub_y_int <= h && (img_binary[max(1, sub_y_int - 2):end, :] .= false)

    # Morphological cleanup
    # Python: Open(2) then Close(2).
    # If the drop is fragmented, Opening (erosion) hurts. We should Close (dilation) first?
    # Or just stick to Python flow but rely on better threshold.
    # Let's reduce Opening kernel or put Closing first to bridge gaps.
    img_clean = opening(closing(img_binary))
    capture_debug && (debug_imgs["4_morph"] = copy(img_clean))

    # Fill holes: Label background components and keep only the largest one (true background)
    # This removes internal holes in the drop which cause jagged detection.
    bg_labels = label_components(.!img_clean)
    bg_areas = component_lengths(bg_labels)
    # bg_areas includes background (0) which is foreground in original. 
    # component_lengths returns count for label i at index i.
    # Label 0 is ignored by component_lengths usually? No, check docs or behavior.
    # label_components returns integer array. 0 is background (false in input).
    # Since we passed .!img_clean, the true background (dark) is now true (foreground for labeling).
    # So we look for the largest component in bg_labels.

    # We want to keep the largest component of the INVERTED image (the background)
    # and set everything else to foreground (drop).

    if length(bg_areas) > 1
        # Find largest component index (excluding 0 if it exists)
        # component_lengths returns vector counting pixels for label 1, 2, ...
        largest_bg_idx = argmax(bg_areas)

        # Create new filled mask: True where background is NOT the largest component
        # i.e., Drop = (Original Drop) OR (Small Holes)
        # = NOT (Largest Background Component)
        img_filled = bg_labels .!= largest_bg_idx
    else
        img_filled = img_clean
    end
    capture_debug && (debug_imgs["5_filled"] = copy(img_filled))

    # Find components on filled image
    labels = label_components(img_filled)
    n_components = maximum(labels)
    n_components == 0 && return nothing

    # Find valid drop contours
    valid_contours = []
    for i in 1:n_components
        component_mask = labels .== i
        coords = findall(component_mask)
        isempty(coords) && continue

        area = length(coords)
        rows = [c[1] for c in coords]
        cols = [c[2] for c in coords]
        min_r, max_r = extrema(rows)
        min_c, max_c = extrema(cols)

        # Skip needle (touches top)
        min_r < 5 && continue

        # Validate drop
        if area > (w * h) * 0.005 && min_c > 5 && max_c < (w - 5)
            push!(valid_contours, (idx=i, area=area, mask=component_mask))
        end
    end

    isempty(valid_contours) && return nothing

    # Select largest
    sort!(valid_contours, by=x -> x.area, rev=true)
    drop_mask = valid_contours[1].mask

    # Get contour and hull
    contour_points = trace_boundary(drop_mask)
    size(contour_points, 1) < 3 && return nothing
    hull_points = convex_hull_gift_wrap(contour_points)

    # Filter dome points (above substrate)
    dome_mask = hull_points[:, 2] .< (substrate_y - 5)
    dome_points = hull_points[dome_mask, :]
    size(dome_points, 1) < 3 && return nothing

    # Sort by x
    dome_points = dome_points[sortperm(dome_points[:, 1]), :]

    # Extract geometry
    x_left, x_right = dome_points[1, 1], dome_points[end, 1]
    cp_left, cp_right = (x_left, substrate_y), (x_right, substrate_y)

    min_y_val = minimum(dome_points[:, 2])
    apex_candidates = dome_points[abs.(dome_points[:, 2] .- min_y_val).<1.0, :]
    apex = (mean(apex_candidates[:, 1]), min_y_val)

    height_px = substrate_y - min_y_val
    base_width_px = x_right - x_left

    # Build final contour
    final_contour = vcat([cp_left[1] cp_left[2]], dome_points, [cp_right[1] cp_right[2]])

    # ROI
    pad = 20
    roi = (max(1, round(Int, x_left - pad)), max(1, round(Int, min_y_val - pad)),
        min(w, round(Int, x_right + pad)), min(h, round(Int, substrate_y + pad)))

    # Save contour
    contour_filename = "julia_contour_$(basename(image_path)).txt"
    try
        open(contour_filename, "w") do f
            for i in 1:size(final_contour, 1)
                @printf(f, "%.2f,%.2f\n", final_contour[i, 1], final_contour[i, 2])
            end
        end
        println("Saved contour to $contour_filename")
    catch
    end

    return DropDetectionResult(substrate_y, cp_left, cp_right, apex,
        height_px, base_width_px, dome_points, final_contour, roi, debug_imgs)
end

# ==============================================================================
# 4. CONTACT ANGLE METHODS
# ==============================================================================

"""Compute contact angle using apex-based spherical cap geometry."""
function contact_angle_from_apex(apex, cp_left, cp_right, substrate_y)
    height_px = substrate_y - apex[2]
    base_width_px = abs(cp_right[1] - cp_left[1])
    (height_px <= 0 || base_width_px <= 0) && return nothing

    a = base_width_px / 2.0
    R = (height_px^2 + a^2) / (2.0 * height_px)
    cos_theta = clamp((a^2 - height_px^2) / (a^2 + height_px^2), -1.0, 1.0)
    theta_deg = rad2deg(acos(cos_theta))
    volume_px3 = π * height_px^2 * (3.0 * R - height_px) / 3.0

    return Dict("theta_deg" => theta_deg, "height_px" => height_px,
        "base_width_px" => base_width_px, "radius_px" => R, "volume_px3" => volume_px3)
end

"""Calculate contact angle using polynomial fitting near the contact point."""
function calculate_contact_angle_tangent(contour_points::Matrix{Float64},
    contact_point, substrate_y; side=:left, fit_points=30)
    cp_x, cp_y = contact_point

    # Use larger search radius
    distances = sqrt.((contour_points[:, 1] .- cp_x) .^ 2 .+ (contour_points[:, 2] .- cp_y) .^ 2)
    nearby_mask = (distances .< 100) .& (contour_points[:, 2] .< substrate_y - 2)
    nearby_points = contour_points[nearby_mask, :]
    size(nearby_points, 1) < 5 && return nothing

    # Sort by y (height from contact upward)
    nearby_points = nearby_points[sortperm(nearby_points[:, 2], rev=true), :]

    # Filter by side
    if side == :left
        mask = nearby_points[:, 1] .>= cp_x - 10
        fit_pts = nearby_points[mask, :]
    else
        mask = nearby_points[:, 1] .<= cp_x + 10
        fit_pts = nearby_points[mask, :]
    end

    n_pts = min(fit_points, size(fit_pts, 1))
    n_pts < 5 && return nothing
    fit_pts = fit_pts[1:n_pts, :]

    try
        x_fit, y_fit = fit_pts[:, 1], fit_pts[:, 2]

        # Use cubic polynomial for better accuracy
        if n_pts >= 8
            A = hcat(x_fit .^ 3, x_fit .^ 2, x_fit, ones(n_pts))
            coeffs = A \ y_fit
            slope = 3 * coeffs[1] * cp_x^2 + 2 * coeffs[2] * cp_x + coeffs[3]
        else
            A = hcat(x_fit .^ 2, x_fit, ones(n_pts))
            coeffs = A \ y_fit
            slope = 2 * coeffs[1] * cp_x + coeffs[2]
        end

        # Contact angle from slope (y inverted in image coords)
        angle_rad = atan(-slope)
        angle_deg = rad2deg(angle_rad)

        # Adjust to 0-180 range
        if side == :left
            angle_deg = angle_deg < 0 ? -angle_deg : 180 - angle_deg
        else
            angle_deg = angle_deg < 0 ? -angle_deg : 180 - angle_deg
        end

        return clamp(angle_deg, 0.0, 180.0)
    catch
        return nothing
    end
end

"""Fit a spherical cap model to the drop profile."""
function fit_spherical_cap(contour_points::Matrix{Float64}, contact_left, contact_right, substrate_y)
    base_width = abs(contact_right[1] - contact_left[1])
    height = substrate_y - minimum(contour_points[:, 2])
    (height <= 0 || base_width <= 0) && return (nothing, nothing, nothing)

    R = (height^2 + (base_width / 2)^2) / (2 * height)
    theta_deg = rad2deg(asin(min((base_width / 2) / R, 1.0)))
    volume = π * height^2 * (3R - height) / 3
    return (theta_deg, R, volume)
end

"""Fit an ellipse to the drop profile and compute contact angles."""
function fit_elliptical(contour_points::Matrix{Float64}, contact_left, contact_right, substrate_y)
    # Use all contour points for better fitting
    n = size(contour_points, 1)
    n < 10 && return (nothing, nothing, nothing, nothing, nothing)

    # Filter to dome points only
    dome_mask = contour_points[:, 2] .< (substrate_y - 5)
    dome_pts = contour_points[dome_mask, :]
    size(dome_pts, 1) < 10 && return (nothing, nothing, nothing, nothing, nothing)

    ellipse = fit_ellipse(dome_pts)
    isnothing(ellipse) && return (nothing, nothing, nothing, nothing, nothing)

    a, b = ellipse.semi_major, ellipse.semi_minor
    cx, cy = ellipse.center_x, ellipse.center_y

    dx_left = contact_left[1] - cx
    dx_right = contact_right[1] - cx
    dy = substrate_y - cy

    if abs(dy) > 1e-10
        ax_h = max(a, b)  # Horizontal semi-axis
        ax_v = min(a, b)  # Vertical semi-axis

        slope_left = -(ax_v^2 * dx_left) / (ax_h^2 * dy)
        angle_left = rad2deg(atan(abs(slope_left)))
        slope_left > 0 && (angle_left = 180 - angle_left)

        slope_right = -(ax_v^2 * dx_right) / (ax_h^2 * dy)
        angle_right = rad2deg(atan(abs(slope_right)))
        slope_right < 0 && (angle_right = 180 - angle_right)
    else
        angle_left = angle_right = 90.0
    end

    volume = (4 / 3) * π * a * b * min(a, b)
    return (angle_left, angle_right, a, b, volume)
end

# ==============================================================================
# 5. YOUNG-LAPLACE FITTING
# ==============================================================================

"""Young-Laplace ODE: u = [r, z, phi], p = (b, c)"""
function young_laplace_ode!(du, u, p, s)
    r, z, phi = u
    b, c = p
    du[1] = cos(phi)
    du[2] = sin(phi)
    du[3] = r < 1e-12 ? b : 2.0 * b - c * z - sin(phi) / r
end

"""Fit Young-Laplace equation to profile points."""
function fit_young_laplace(contour_points::Matrix{Float64}, apex, contact_left, contact_right,
    substrate_y; rho=1000.0, sigma=72e-3, g=9.81, pixel_size_m=1e-6, n_fit_points=100)

    size(contour_points, 1) < 10 && return Dict("left_deg" => nothing, "right_deg" => nothing,
        "mean_deg" => nothing, "capillary_length" => nothing, "bond_number" => nothing)

    apex_x, apex_y = apex
    base_center_x = 0.5 * (contact_left[1] + contact_right[1])
    height_px = substrate_y - apex_y
    base_width_px = abs(contact_right[1] - contact_left[1])

    half_base_m = 0.5 * base_width_px * pixel_size_m
    height_m = height_px * pixel_size_m
    R0_guess = (half_base_m^2 + height_m^2) / (2.0 * max(height_m, 1e-12))
    b0 = 1.0 / R0_guess
    c_fixed = rho * g / sigma

    function fit_side(side::Symbol)
        if side == :right
            side_mask = contour_points[:, 1] .>= base_center_x
            cx, cy = contact_right
            sign_val = 1.0
        else
            side_mask = contour_points[:, 1] .<= base_center_x
            cx, cy = contact_left
            sign_val = -1.0
        end

        pts = contour_points[side_mask.&(contour_points[:, 2].<=substrate_y+1e-6), :]
        size(pts, 1) < 10 && return (nothing, nothing)
        pts = pts[sortperm(pts[:, 2]), :]

        z_data = (pts[:, 2] .- apex_y) .* pixel_size_m
        r_data = sign_val .* (pts[:, 1] .- base_center_x) .* pixel_size_m
        mask_pos = (z_data .>= 0) .& (r_data .>= 0)
        z_data, r_data = z_data[mask_pos], r_data[mask_pos]
        length(z_data) < 10 && return (nothing, nothing)

        z_contact = (cy - apex_y) * pixel_size_m
        (z_contact <= 0) && return (nothing, nothing)

        # Grid search for best b
        best_b, best_cost = b0, Inf
        for b_scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            b_try = b0 * b_scale
            z_max = max(maximum(z_data), z_contact)
            prob = ODEProblem(young_laplace_ode!, [0.0, 0.0, 0.0], (0.0, 5.0 * z_max), (b_try, c_fixed))
            sol = try
                solve(prob, Tsit5(), saveat=z_max / 100)
            catch
                continue
            end
            sol.retcode != :Success && continue

            z_sol = [u[2] for u in sol.u]
            r_sol = [u[1] for u in sol.u]
            maximum(z_sol) < z_contact && continue

            cost = 0.0
            for i in eachindex(z_data)
                idx = findfirst(z -> z >= z_data[i], z_sol)
                if !isnothing(idx) && idx > 1
                    t = (z_data[i] - z_sol[idx-1]) / (z_sol[idx] - z_sol[idx-1])
                    r_fit = r_sol[idx-1] + t * (r_sol[idx] - r_sol[idx-1])
                    cost += (r_fit - r_data[i])^2
                else
                    cost += 1e6
                end
            end
            cost < best_cost && (best_cost = cost; best_b = b_try)
        end

        # Final angle
        z_max = max(maximum(z_data), z_contact)
        prob = ODEProblem(young_laplace_ode!, [0.0, 0.0, 0.0], (0.0, 6.0 * z_max), (best_b, c_fixed))
        sol = solve(prob, Tsit5(), saveat=z_max / 200)
        z_sol = [u[2] for u in sol.u]
        phi_sol = [u[3] for u in sol.u]

        phi_contact = 0.0
        for i in eachindex(z_sol)
            if z_sol[i] >= z_contact && i > 1
                t = (z_contact - z_sol[i-1]) / (z_sol[i] - z_sol[i-1])
                phi_contact = phi_sol[i-1] + t * (phi_sol[i] - phi_sol[i-1])
                break
            end
        end
        return (rad2deg(phi_contact), best_b)
    end

    theta_right, b_right = fit_side(:right)
    theta_left, b_left = fit_side(:left)

    valid_bs = filter(!isnothing, [b_left, b_right])
    isempty(valid_bs) && return Dict("left_deg" => theta_left, "right_deg" => theta_right,
        "mean_deg" => nothing, "capillary_length" => nothing, "bond_number" => nothing)

    b_mean = mean(valid_bs)
    R0 = 1.0 / b_mean
    mean_deg = if !isnothing(theta_left) && !isnothing(theta_right)
        0.5 * (theta_left + theta_right)
    else
        something(theta_left, theta_right)
    end

    return Dict("left_deg" => theta_left, "right_deg" => theta_right, "mean_deg" => mean_deg,
        "capillary_length" => sqrt(sigma / (rho * g)), "bond_number" => rho * g * R0^2 / sigma)
end

"""Young-Laplace fit that also estimates surface tension."""
function fit_young_laplace_unknown_sigma(contour_points::Matrix{Float64}, apex, contact_left, contact_right,
    substrate_y; rho=1000.0, g=9.81, pixel_size_m=1e-6)

    size(contour_points, 1) < 10 && return (nothing, nothing, nothing, nothing, nothing)

    apex_x, apex_y = apex
    base_center_x = 0.5 * (contact_left[1] + contact_right[1])
    height_px = substrate_y - apex_y
    base_width_px = abs(contact_right[1] - contact_left[1])

    half_base_m = 0.5 * base_width_px * pixel_size_m
    height_m = height_px * pixel_size_m
    R0_guess = (half_base_m^2 + height_m^2) / (2.0 * max(height_m, 1e-12))
    b0 = 1.0 / R0_guess
    sigma_guess = 72e-3

    function fit_side_sigma(side::Symbol)
        if side == :right
            side_mask = contour_points[:, 1] .>= base_center_x
            cx, cy = contact_right
            sign_val = 1.0
        else
            side_mask = contour_points[:, 1] .<= base_center_x
            cx, cy = contact_left
            sign_val = -1.0
        end

        pts = contour_points[side_mask.&(contour_points[:, 2].<=substrate_y+1e-6), :]
        size(pts, 1) < 10 && return (nothing, nothing, nothing)
        pts = pts[sortperm(pts[:, 2]), :]

        z_data = (pts[:, 2] .- apex_y) .* pixel_size_m
        r_data = sign_val .* (pts[:, 1] .- base_center_x) .* pixel_size_m
        mask_pos = (z_data .>= 0) .& (r_data .>= 0)
        z_data, r_data = z_data[mask_pos], r_data[mask_pos]
        length(z_data) < 10 && return (nothing, nothing, nothing)

        z_contact = (cy - apex_y) * pixel_size_m
        (z_contact <= 0) && return (nothing, nothing, nothing)

        # Grid search over (b, sigma)
        # Using a coarse grid first, then refining.
        best_params, best_cost = (b0, sigma_guess), Inf

        # Coarse grid
        for b_scale in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
            for sigma_scale in [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
                b_try = b0 * b_scale
                sigma_try = sigma_guess * sigma_scale
                sigma_try < 1e-3 && continue # Physics constraint

                c_try = rho * g / sigma_try
                z_max = max(maximum(z_data), z_contact)
                prob = ODEProblem(young_laplace_ode!, [0.0, 0.0, 0.0], (0.0, 5.0 * z_max), (b_try, c_try))
                sol = try
                    solve(prob, Tsit5(), saveat=z_max / 50) # Coarse resolution for speed
                catch
                    continue
                end
                sol.retcode != :Success && continue
                z_sol = [u[2] for u in sol.u]
                r_sol = [u[1] for u in sol.u]
                maximum(z_sol) < z_contact && continue

                cost = 0.0
                for i in eachindex(z_data)
                    idx = findfirst(z -> z >= z_data[i], z_sol)
                    if !isnothing(idx) && idx > 1
                        t = (z_data[i] - z_sol[idx-1]) / (z_sol[idx] - z_sol[idx-1])
                        r_fit = r_sol[idx-1] + t * (r_sol[idx] - r_sol[idx-1])
                        cost += (r_fit - r_data[i])^2
                    else
                        cost += 1e6
                    end
                end

                if cost < best_cost
                    best_cost = cost
                    best_params = (b_try, sigma_try)
                end
            end
        end

        # Refinement (Coordinate Descent / Simple Hill Climbing)
        # To get non-discrete values.
        b_curr, sigma_curr = best_params
        step_b = b_curr * 0.1
        step_sigma = sigma_curr * 0.1
        min_step_frac = 1e-3

        while step_b > b_curr * min_step_frac || step_sigma > sigma_curr * min_step_frac
            improved = false

            # Try variations
            candidates = [
                (b_curr + step_b, sigma_curr), (b_curr - step_b, sigma_curr),
                (b_curr, sigma_curr + step_sigma), (b_curr, sigma_curr - step_sigma)
            ]

            for (b_try, sigma_try) in candidates
                (sigma_try < 1e-3 || b_try <= 0) && continue

                c_try = rho * g / sigma_try
                z_max = max(maximum(z_data), z_contact)
                prob = ODEProblem(young_laplace_ode!, [0.0, 0.0, 0.0], (0.0, 5.0 * z_max), (b_try, c_try))
                sol = try
                    solve(prob, Tsit5(), saveat=z_max / 100) # Finer resolution
                catch
                    continue
                end

                z_sol = [u[2] for u in sol.u]
                r_sol = [u[1] for u in sol.u]
                maximum(z_sol) < z_contact && continue

                cost = 0.0
                for i in eachindex(z_data)
                    idx = findfirst(z -> z >= z_data[i], z_sol)
                    if !isnothing(idx) && idx > 1
                        t = (z_data[i] - z_sol[idx-1]) / (z_sol[idx] - z_sol[idx-1])
                        r_fit = r_sol[idx-1] + t * (r_sol[idx] - r_sol[idx-1])
                        cost += (r_fit - r_data[i])^2
                    else
                        cost += 1e6
                    end
                end

                if cost < best_cost
                    best_cost = cost
                    best_params = (b_try, sigma_try)
                    improved = true
                end
            end

            if improved
                b_curr, sigma_curr = best_params
            else
                step_b *= 0.5
                step_sigma *= 0.5
            end
        end

        b_opt, sigma_opt = best_params
        c_opt = rho * g / sigma_opt

        z_max = max(maximum(z_data), z_contact)
        prob = ODEProblem(young_laplace_ode!, [0.0, 0.0, 0.0], (0.0, 6.0 * z_max), (b_opt, c_opt))
        sol = solve(prob, Tsit5(), saveat=z_max / 200)
        z_sol = [u[2] for u in sol.u]
        phi_sol = [u[3] for u in sol.u]

        phi_contact = 0.0
        for i in eachindex(z_sol)
            if z_sol[i] >= z_contact && i > 1
                t = (z_contact - z_sol[i-1]) / (z_sol[i] - z_sol[i-1])
                phi_contact = phi_sol[i-1] + t * (phi_sol[i] - phi_sol[i-1])
                break
            end
        end
        return (rad2deg(phi_contact), b_opt, sigma_opt)
    end

    result_right = fit_side_sigma(:right)
    result_left = fit_side_sigma(:left)

    theta_right, theta_left = result_right[1], result_left[1]

    valid_sigmas = Float64[]
    valid_bs = Float64[]
    !isnothing(result_right[2]) && (push!(valid_bs, result_right[2]); push!(valid_sigmas, result_right[3]))
    !isnothing(result_left[2]) && (push!(valid_bs, result_left[2]); push!(valid_sigmas, result_left[3]))

    isempty(valid_sigmas) && return (theta_left, theta_right, nothing, nothing, nothing)

    sigma_mean = mean(valid_sigmas)
    R0 = 1.0 / mean(valid_bs)
    return (theta_left, theta_right, sigma_mean, sqrt(sigma_mean / (rho * g)), rho * g * R0^2 / sigma_mean)
end

# ==============================================================================
# 6. AGGREGATION & OUTPUT
# ==============================================================================

"""Compute contact angles using all available methods."""
function compute_contact_angles_from_detection(det::DropDetectionResult;
    rho=1000.0, sigma=72e-3, g=9.81, pixel_size_m=1e-6)
    out = Dict{String,Any}()

    out["apex_spherical"] = contact_angle_from_apex(det.apex, det.cp_left, det.cp_right, det.substrate_y)

    out["tangent"] = Dict(
        "left_deg" => calculate_contact_angle_tangent(det.dome_points_array, det.cp_left, det.substrate_y; side=:left),
        "right_deg" => calculate_contact_angle_tangent(det.dome_points_array, det.cp_right, det.substrate_y; side=:right))

    theta_s, R_s, vol_s = fit_spherical_cap(det.dome_points_array, det.cp_left, det.cp_right, det.substrate_y)
    out["spherical_fit"] = Dict("theta_deg" => theta_s, "radius_px" => R_s, "volume_px3" => vol_s)

    al, ar, a, b, vol_e = fit_elliptical(det.dome_points_array, det.cp_left, det.cp_right, det.substrate_y)
    mean_e = (!isnothing(al) && !isnothing(ar)) ? 0.5 * (al + ar) : nothing
    out["ellipse_fit"] = Dict("left_deg" => al, "right_deg" => ar, "mean_deg" => mean_e, "a_px" => a, "b_px" => b, "volume_px3" => vol_e)

    out["young_laplace"] = fit_young_laplace(det.dome_points_array, det.apex, det.cp_left, det.cp_right, det.substrate_y;
        rho=rho, sigma=sigma, g=g, pixel_size_m=pixel_size_m)
    return out
end

"""Compare Young-Laplace fit with known vs unknown surface tension."""
function compare_yl_methods(det::DropDetectionResult; pixel_size_m=1e-6, rho=1000.0)
    println("="^60)
    println("Method 1: Known surface tension (σ = 72 mN/m)")
    println("="^60)

    yl1 = fit_young_laplace(det.dome_points_array, det.apex, det.cp_left, det.cp_right, det.substrate_y;
        rho=rho, sigma=72e-3, pixel_size_m=pixel_size_m)
    @printf("Left angle:  %.2f°\n", something(yl1["left_deg"], NaN))
    @printf("Right angle: %.2f°\n", something(yl1["right_deg"], NaN))
    @printf("Bond number: %.3f\n", something(yl1["bond_number"], NaN))

    println("\n" * "="^60)
    println("Method 2: Unknown surface tension (fitted)")
    println("="^60)

    theta_l2, theta_r2, sigma_fit, _, Bo2 = fit_young_laplace_unknown_sigma(
        det.dome_points_array, det.apex, det.cp_left, det.cp_right, det.substrate_y;
        rho=rho, pixel_size_m=pixel_size_m)
    @printf("Left angle:  %.2f°\n", something(theta_l2, NaN))
    @printf("Right angle: %.2f°\n", something(theta_r2, NaN))
    @printf("Fitted σ:    %.2f mN/m\n", something(sigma_fit, NaN) * 1000)
    @printf("Bond number: %.3f\n", something(Bo2, NaN))

    println("\n" * "="^60)
    println("Comparison")
    println("="^60)
    !isnothing(yl1["left_deg"]) && !isnothing(theta_l2) && @printf("Angle diff (left):  %.2f°\n", abs(yl1["left_deg"] - theta_l2))
    !isnothing(yl1["right_deg"]) && !isnothing(theta_r2) && @printf("Angle diff (right): %.2f°\n", abs(yl1["right_deg"] - theta_r2))
    !isnothing(sigma_fit) && @printf("Surface tension error: %.2f mN/m (%.1f%%)\n", abs(sigma_fit * 1000 - 72), abs(sigma_fit * 1000 - 72) / 72 * 100)
end

"""Run complete analysis on an image and print results."""
function run_analysis(image_path::String; pixel_size_m=2.88e-5, rho=1000.0, debug=false)
    println("Processing: $image_path")
    println("-"^40)

    det = sessile_drop_adaptive(image_path; capture_debug=debug)
    if isnothing(det)
        println("Error: Could not detect drop.")
        return nothing, nothing
    end

    println("\n--- DETECTION OUTPUT ---")
    @printf("substrate_y: %.2f\n", det.substrate_y)
    @printf("cp_left: (%.2f, %.2f)\n", det.cp_left...)
    @printf("cp_right: (%.2f, %.2f)\n", det.cp_right...)
    @printf("apex: (%.2f, %.2f)\n", det.apex...)
    @printf("height_px: %.2f\n", det.height_px)
    @printf("base_width_px: %.2f\n", det.base_width_px)
    println("-"^40)

    angles = compute_contact_angles_from_detection(det; rho=rho, sigma=72e-3, pixel_size_m=pixel_size_m)

    println("\nApex-based spherical cap:")
    println(angles["apex_spherical"])
    println("\nTangent method:")
    println(angles["tangent"])
    println("\nSpherical-cap fit:")
    println(angles["spherical_fit"])
    println("\nElliptical fit:")
    println(angles["ellipse_fit"])
    println("\nYoung-Laplace fit:")
    println(angles["young_laplace"])
    println()
    compare_yl_methods(det; pixel_size_m=pixel_size_m, rho=rho)

    return det, angles
end

# ==============================================================================
# 7. VISUALIZATION FUNCTIONS
# ==============================================================================

"""
    plot_detection(det::DropDetectionResult; savepath=nothing, background_img=nothing)

Plot the detection results showing contour, contact points, apex, and substrate.
If `background_img` is provided, plots results as an overlay on the image.
"""
function plot_detection(det::DropDetectionResult; savepath=nothing, background_img=nothing, title="Sessile Drop Detection")
    if !isnothing(background_img)
        # Plot over the image
        # Images usually need yflip=false if plotting directly, but Plots.jl with images 
        # often handles coordinates top-left (0,0).
        # We'll use yflip=false because Images.jl coordinate system is usually (y, x) top-left.
        # But our coordinates are (x, y) with y=0 at top. 
        # Let's try standard plot with image.
        p = plot(background_img, title=title, aspect_ratio=:equal)
    else
        p = plot(aspect_ratio=:equal, yflip=true, legend=:topright,
            xlabel="x [px]", ylabel="y [px]", title=title)
    end

    # Plot full contour
    plot!(p, det.full_contour[:, 1], det.full_contour[:, 2],
        linewidth=2, label="Contour", color=:cyan)

    # Plot dome points (hull)
    plot!(p, vcat(det.dome_points_array[:, 1], det.dome_points_array[1, 1]),
        vcat(det.dome_points_array[:, 2], det.dome_points_array[1, 2]),
        linewidth=1, linestyle=:dot, color=:green, label="Dome (hull)")

    # Substrate line
    x_min = minimum(det.full_contour[:, 1]) - 20
    x_max = maximum(det.full_contour[:, 1]) + 20
    plot!(p, [x_min, x_max], [det.substrate_y, det.substrate_y],
        linewidth=2, linestyle=:dash, color=:magenta, label="Substrate")

    # Only show these if we have no background, to avoid clutter
    if isnothing(background_img)
        # Contact points
        scatter!(p, [det.cp_left[1], det.cp_right[1]], [det.cp_left[2], det.cp_right[2]],
            markersize=8, color=:red, markershape=:circle, label="Contacts")

        # Apex
        scatter!(p, [det.apex[1]], [det.apex[2]],
            markersize=10, color=:blue, markershape=:star5, label="Apex")
    else
        # Smaller markers for overlay
        scatter!(p, [det.cp_left[1], det.cp_right[1]], [det.cp_left[2], det.cp_right[2]],
            markersize=5, color=:red, markershape=:circle, label="")
        scatter!(p, [det.apex[1]], [det.apex[2]],
            markersize=6, color=:blue, markershape=:star5, label="")
    end

    # Annotations
    annotate!(p, det.apex[1], det.apex[2] - 15,
        text("Apex", 10, :blue, :bottom))
    annotate!(p, det.cp_left[1], det.cp_left[2] + 15,
        text("L", 10, :red, :top))
    annotate!(p, det.cp_right[1], det.cp_right[2] + 15,
        text("R", 10, :red, :top))

    !isnothing(savepath) && savefig(p, savepath)
    return p
end

"""
    plot_analysis(det::DropDetectionResult, angles::Dict; savepath=nothing, background_img=nothing)

Plot complete analysis with all fitted curves and angle results.
"""
function plot_analysis(det::DropDetectionResult, angles::Dict; savepath=nothing, background_img=nothing)
    # Create 2x2 subplot
    p1 = plot_detection(det, title="Detection / Overlay", background_img=background_img)

    # Angles summary plot
    p2 = plot(title="Contact Angle Comparison", legend=:outerright,
        xlabel="Method", ylabel="Angle [°]")

    methods = String[]
    left_angles = Float64[]
    right_angles = Float64[]

    # Apex spherical
    apex = angles["apex_spherical"]
    if !isnothing(apex)
        push!(methods, "Apex")
        push!(left_angles, apex["theta_deg"])
        push!(right_angles, apex["theta_deg"])
    end

    # Tangent
    tang = angles["tangent"]
    if !isnothing(tang["left_deg"]) || !isnothing(tang["right_deg"])
        push!(methods, "Tangent")
        push!(left_angles, something(tang["left_deg"], NaN))
        push!(right_angles, something(tang["right_deg"], NaN))
    end

    # Spherical fit
    sph = angles["spherical_fit"]
    if !isnothing(sph["theta_deg"])
        push!(methods, "Spherical")
        push!(left_angles, sph["theta_deg"])
        push!(right_angles, sph["theta_deg"])
    end

    # Ellipse fit
    ell = angles["ellipse_fit"]
    if !isnothing(ell["left_deg"]) || !isnothing(ell["right_deg"])
        push!(methods, "Ellipse")
        push!(left_angles, something(ell["left_deg"], NaN))
        push!(right_angles, something(ell["right_deg"], NaN))
    end

    # Young-Laplace
    yl = angles["young_laplace"]
    if !isnothing(yl["left_deg"]) || !isnothing(yl["right_deg"])
        push!(methods, "Y-L")
        push!(left_angles, something(yl["left_deg"], NaN))
        push!(right_angles, something(yl["right_deg"], NaN))
    end

    if !isempty(methods)
        x_pos = 1:length(methods)
        bar!(p2, x_pos .- 0.15, left_angles, bar_width=0.3, label="Left", color=:green)
        bar!(p2, x_pos .+ 0.15, right_angles, bar_width=0.3, label="Right", color=:orange)
        xticks!(p2, x_pos, methods)
    end

    # Geometry info
    p3 = plot(title="Drop Geometry", axis=false, grid=false)
    info = """
    Substrate Y: $(round(det.substrate_y, digits=1)) px
    Height: $(round(det.height_px, digits=1)) px
    Base Width: $(round(det.base_width_px, digits=1)) px
    Apex: ($(round(det.apex[1], digits=1)), $(round(det.apex[2], digits=1)))
    Left Contact: ($(round(det.cp_left[1], digits=1)), $(round(det.cp_left[2], digits=1)))
    Right Contact: ($(round(det.cp_right[1], digits=1)), $(round(det.cp_right[2], digits=1)))
    """
    annotate!(p3, 0.5, 0.5, text(info, 10, :left))

    # Results summary
    p4 = plot(title="Results Summary", axis=false, grid=false)
    results = "CONTACT ANGLES:\n"
    if !isnothing(apex)
        results *= @sprintf("  Apex Spherical: %.1f°\n", apex["theta_deg"])
    end
    if !isnothing(tang["left_deg"]) && !isnothing(tang["right_deg"])
        results *= @sprintf("  Tangent: L=%.1f° R=%.1f°\n", tang["left_deg"], tang["right_deg"])
    end
    if !isnothing(yl["mean_deg"])
        results *= @sprintf("  Young-Laplace: %.1f°\n", yl["mean_deg"])
    end
    if !isnothing(ell["mean_deg"])
        results *= @sprintf("  Ellipse: %.1f°\n", ell["mean_deg"])
    end
    annotate!(p4, 0.5, 0.5, text(results, 10, :left))

    # Combine plots
    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))

    !isnothing(savepath) && savefig(p, savepath)
    return p
end

"""
    plot_debug_stages(det::DropDetectionResult; savepath=nothing)

Plot the intermediate processing stages stored in `det.debug_images`.
"""
function plot_debug_stages(det::DropDetectionResult; savepath=nothing)
    if isempty(det.debug_images)
        println("No debug images found in result.")
        return nothing
    end

    # Extract images (default to nothing if missing)
    img_gray = get(det.debug_images, "1_grayscale", nothing)
    img_clahe = get(det.debug_images, "2_clahe", nothing)
    img_thresh = get(det.debug_images, "3_threshold", nothing)
    img_morph = get(det.debug_images, "4_morph", nothing)
    img_filled = get(det.debug_images, "5_filled", nothing)

    # Create subplots
    plots = []

    if !isnothing(img_gray)
        push!(plots, plot(img_gray, title="1. Grayscale", axis=false, ticks=false))
    end
    if !isnothing(img_clahe)
        push!(plots, plot(img_clahe, title="2. CLAHE", axis=false, ticks=false))
    end
    if !isnothing(img_thresh)
        # Convert BitMatrix to Gray for plotting if needed, but Plots handles boolean masks fine (black/white)
        push!(plots, plot(Gray.(img_thresh), title="3. Adaptive Thresh", axis=false, ticks=false))
    end
    if !isnothing(img_morph)
        push!(plots, plot(Gray.(img_morph), title="4. Morph Clean", axis=false, ticks=false))
    end
    if !isnothing(img_filled)
        push!(plots, plot(Gray.(img_filled), title="5. Hole Filled", axis=false, ticks=false))
    end

    # Add final result logic for comparison (overlay logic)
    # We can reuse plot_detection for the final frame but we need the original image for overlay
    # Since we don't have the original image purely inside 'det' (unless we store it),
    # let's just plot the contour
    p_final = plot(aspect_ratio=:equal, yflip=true, title="6. Final Contour")
    plot!(p_final, det.full_contour[:, 1], det.full_contour[:, 2], linewidth=2, label="Contour")
    push!(plots, p_final)

    # Layout logic
    n = length(plots)
    cols = 3
    rows = cld(n, cols)

    p = plot(plots..., layout=(rows, cols), size=(1200, 400 * rows))

    !isnothing(savepath) && savefig(p, savepath)
    println("Debug plot generated with $n stages.")
    return p
end

"""
    run_analysis_with_plot(image_path::String; pixel_size_m=2.88e-5, rho=1000.0, show_plot=true, save_plot=true, debug=false)

Run complete analysis and generate visualization. If debug=true, also generates a stages plot.
"""
function run_analysis_with_plot(image_path::String; pixel_size_m=2.88e-5, rho=1000.0, show_plot=true, save_plot=true, debug=false)
    # We need to load image here for plotting
    img = try
        load(image_path)
    catch
        nothing
    end



    det, angles = run_analysis(image_path; pixel_size_m=pixel_size_m, rho=rho, debug=debug)

    if isnothing(det)
        return nothing, nothing, nothing
    end

    # Generate plot
    base_name = replace(basename(image_path), r"\.[^.]+$" => "")
    plot_path = save_plot ? "$(base_name)_analysis.png" : nothing

    p = plot_analysis(det, angles; savepath=plot_path, background_img=img)

    if save_plot
        println("\nPlot saved to: $plot_path")
    end

    if debug
        debug_plot_path = save_plot ? "$(base_name)_debug_stages.png" : nothing
        p_debug = plot_debug_stages(det; savepath=debug_plot_path)
        if show_plot && !isnothing(p_debug)
            display(p_debug)
        end
        save_plot && !isnothing(debug_plot_path) && println("Debug plot saved to: $debug_plot_path")
    end

    if show_plot
        display(p)
    end

    return det, angles, p
end

export plot_detection, plot_analysis, run_analysis_with_plot, plot_debug_stages

end # module

# ==============================================================================
# STANDALONE EXECUTION
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    using .SessileDropAnalysis

    println("\n" * "="^60)
    println("SESSILE DROP ANALYSIS - Julia Port")
    println("="^60 * "\n")

    det1, angles1 = SessileDropAnalysis.run_analysis("./data/samples/prueba sesil 2.png")

    println("\n" * "="^60 * "\n")

    det2, angles2 = SessileDropAnalysis.run_analysis("./data/samples/gota depositada 1.png")
end
