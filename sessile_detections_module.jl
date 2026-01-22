module SessileDrop

using Images
using ImageContrastAdjustment
using ImageMorphology
using ImageFiltering
using ImageBinarization
using DifferentialEquations
using LsqFit
using Plots
using Statistics
using LinearAlgebra
using LinearAlgebra
using Printf
# ==============================================================================
# 1. CORE DATA STRUCTURES
# ==============================================================================

struct DropDetectionResult
    substrate_y::Float64
    cp_left::Tuple{Float64,Float64}
    cp_right::Tuple{Float64,Float64}
    apex::Tuple{Float64,Float64}
    dome_points::Vector{Tuple{Float64,Float64}}
    height_px::Float64
    base_width_px::Float64
    full_contour::Vector{Tuple{Float64,Float64}}
    roi_coords::Tuple{UnitRange{Int64},UnitRange{Int64}}
    roi_img::Matrix{RGB{N0f8}}
end

# ==============================================================================
# 2. IMAGE PROCESSING PIPELINE
# ==============================================================================

"""
    trace_boundary(mask::BitMatrix)

Implements Moore-Neighbor tracing to get an ordered list of boundary coordinates from a binary mask.
Equivalent to cv2.findContours for a single blob.
"""
function trace_boundary(mask::BitMatrix)
    # Find start point
    rows, cols = size(mask)
    start_idx = findfirst(mask)
    isnothing(start_idx) && return Tuple{Float64,Float64}[]

    start_yx = (start_idx[1], start_idx[2])
    boundary = [start_yx]

    # Clockwise Moore neighborhood offsets (y, x)
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    current = start_yx
    backtrack = 7 # Start looking from "previous" direction

    while true
        found_next = false
        for i in 0:7
            idx = (backtrack + i) % 8 + 1
            dy, dx = offsets[idx]
            ny, nx = current[1] + dy, current[2] + dx

            if 1 <= ny <= rows && 1 <= nx <= cols && mask[ny, nx]
                current = (ny, nx)
                push!(boundary, current)
                # The next backtrack direction is 2 steps back from current entry
                backtrack = (idx + 4) % 8
                found_next = true
                break
            end
        end

        if !found_next || (current == start_yx)
            break
        end

        # Safety break for degenerate cases
        length(boundary) > (rows * cols) && break
    end

    # Convert (row, col) to (x, y) for plotting/math consistency
    return [(Float64(p[2]), Float64(p[1])) for p in boundary]
end

"""
    sessile_drop_adaptive(image_path::String; use_ml_segmentation=false)

Main detection pipeline. 
If `use_ml_segmentation` is true, this is where a Flux.jl U-Net would be invoked.
"""
function sessile_drop_adaptive(image_path::String)
    # 1. Load Image
    img = try
        load(image_path)
    catch e
        println("Error loading image: $e")
        return nothing
    end

    img_gray = Gray.(img)
    h, w = size(img_gray)

    # --- STEP 1: ENHANCEMENT (CLAHE) ---
    img_enhanced = adjust_histogram(img_gray, AdaptiveEqualization(nbins=256, rblocks=8, cblocks=8, clip=0.02))

    # --- STEP 2: SUBSTRATE DETECTION ---
    margin = min(50, w ÷ 10)

    function find_horizon(strip_img)
        # Python Logic:
        # 1. Take column. 2. Calculate diff (grad). 3. Find argmin (Light -> Dark transition).
        # We process each column and take the median y.

        rows, cols = size(strip_img)
        min_limit, max_limit = round(Int, rows * 0.05), round(Int, rows * 0.95)

        detected_ys = Int[]
        for c in 1:cols
            col_data = Float64.(strip_img[:, c]) # Ensure float for diff
            grad = diff(col_data)

            # Valid range for this column's gradient
            valid_grad = grad[min_limit:max_limit]

            if isempty(valid_grad)
                continue
            end

            # Find strongest negative gradient (Light -> Dark)
            # argmin gives index in valid_grad. Add min_limit - 1 to get original index.
            # In Python: best_y = np.argmin(valid_grad) + min_limit
            _, best_local_idx = findmin(valid_grad)
            best_y = best_local_idx + min_limit - 1

            push!(detected_ys, best_y)
        end
        isempty(detected_ys) ? nothing : median(detected_ys)
    end

    y_left = find_horizon(img_enhanced[:, 1:margin])
    y_right = find_horizon(img_enhanced[:, end-margin:end])

    substrate_y = if isnothing(y_left) || isnothing(y_right)
        h * 0.8
    else
        (y_left + y_right) / 2.0
    end

    println("Initial substrate_y: $substrate_y")

    if substrate_y < h * 0.2
        println("WARNING: Substrate detected too high (< 20% height). Defaulting to 0.8*h.")
        substrate_y = h * 0.8
    end

    # --- STEP 3: SEGMENTATION ---
    # Fix: Ensure boolean output immediately
    img_binary = binarize(img_enhanced, AdaptiveThreshold(window_size=21, percentage=10)) .> 0.5

    # Invert binary if background is white
    if mean(img_binary[1:10, 1:10]) > 0.5
        img_binary = .!img_binary
    end

    # Cleanup (Morphology)
    sub_y_int = round(Int, substrate_y)

    if sub_y_int < h
        img_binary[sub_y_int:end, :] .= false
    end

    img_clean = opening(img_binary)
    img_clean = closing(img_clean)

    # --- STEP 4: CONTOUR & FILTERING ---
    labels = label_components(img_clean)
    indices = component_indices(labels)

    isempty(indices) && return nothing

    valid_mask = falses(size(img_clean))
    max_area = 0.0
    best_idx = 0

    # Cache CartesianIndices for conversion
    cartesian_map = CartesianIndices(img_clean)

    valid_drop_contours = []
    needle_cnt = nothing
    needle_y = 0.0
    needle_x = 0.0

    for (i, inds) in enumerate(indices)
        area = length(inds)
        coords = cartesian_map[inds]
        rs = [c[1] for c in coords]
        cs = [c[2] for c in coords] # Need columns for width check

        min_r, max_r = extrema(rs)
        min_c, max_c = extrema(cs)

        # Bounding box centers
        cnt_center_y = (min_r + max_r) / 2
        cnt_center_x = (min_c + max_c) / 2

        # Logic from Python:
        # 1. Needle check: touches top (min_r < 5)
        if min_r < 5
            if isnothing(needle_cnt)
                # Store needle info
                # Reconstruct cnt for store? Or just store ID. 
                # Python stores the contour to draw it.
                # Here we just flag it. 
                # If we want to draw it later we might need to separate mask.
                # For now let's just NOT count it as a drop.
                needle_y = cnt_center_y # Rough center
                # Python does: needle_x = x + w (right edge). 
                # Let's approximate right edge as max_c
                needle_x = max_c
            end
            continue # Skip needle for drop candidacy
        end

        # 2. Drop check
        if area > (w * h) * 0.005
            if min_c > 5 && max_c < (w - 5)
                println("Candidate $i: Area=$area, Center=($cnt_center_x, $cnt_center_y)")
                push!(valid_drop_contours, (i, area, inds))
            end
        else
            println("Rejected $i: Area=$area (Thresh=$((w*h)*0.005))")
        end
    end


    println("DEBUG: Loop done. valid_drop_contours count: $(length(valid_drop_contours))")

    if isempty(valid_drop_contours)
        # Fallback: Find largest contour that is NOT the needle
        println("Warning: No valid drop contours found. Attempting fallback.")

        potential_backups = []
        for (i, inds) in enumerate(indices)
            area = length(inds)
            coords = cartesian_map[inds]
            rs = [c[1] for c in coords]
            min_r = minimum(rs)
            # Avoid needle (top touching) if possible
            if min_r >= 5
                push!(potential_backups, (i, area, inds))
            end
        end

        if isempty(potential_backups)
            # Desperate fallback: just take the largest contour found
            println("Warning: No non-needle contours found. Using absolute largest component (parity with Python 'max area' fallback).")
            all_candidates = []
            for (i, inds) in enumerate(indices)
                push!(all_candidates, (i, length(inds), inds))
            end
            if isempty(all_candidates)
                println("Error: No components found at all.")
                return nothing
            end
            sort!(all_candidates, by=x -> x[2], rev=true)
            best_idx, best_area, best_inds = all_candidates[1]
            valid_drop_contours = [all_candidates[1]] # Hack to satisfy next block
            # We need to ensure we don't crash next block which expects 'valid_drop_contours' to be populated or handled
            # The existing code selects 'best_idx' from 'potential_backups[1]'. 
            # We'll just define best_idx here and skip the "else" block logic?
            # Actually, the logic below handles 'potential_backups'. 
            # Let's populate potential_backups with the desperate choice.
            push!(potential_backups, all_candidates[1])
        end

        sort!(potential_backups, by=x -> x[2], rev=true)
        best_idx, best_area, best_inds = potential_backups[1]
        println("Fallback selected component with Area=$best_area")
    else
        # Select best drop (largest area)
        sort!(valid_drop_contours, by=x -> x[2], rev=true)
        best_idx, best_area, best_inds = valid_drop_contours[1]
    end

    println("DEBUG: Selected Index $best_idx with Area $best_area")



    # Create mask of just the drop
    # Use explicit indices to avoid label mismatch issues
    drop_mask = falses(size(labels))
    drop_mask[best_inds] .= true

    # Get Contour Points (Unordered is fine for Convex Hull)
    # We manually collect boundary pixels to avoid trace_boundary issues
    contour_points = []
    rows, cols = size(drop_mask)
    for c in 1:cols
        for r in 1:rows
            if drop_mask[r, c]
                # Check 4-neighbors for background
                is_boundary = false
                if r == 1 || r == rows || c == 1 || c == cols
                    is_boundary = true
                else
                    if !drop_mask[r-1, c] || !drop_mask[r+1, c] || !drop_mask[r, c-1] || !drop_mask[r, c+1]
                        is_boundary = true
                    end
                end

                if is_boundary
                    push!(contour_points, (r, c))
                end
            end
        end
    end

    println("DEBUG: Extracted $(length(contour_points)) boundary points from mask area $(sum(drop_mask)).")

    # Convex Hull
    pts_vec = [[p[1], p[2]] for p in contour_points]
    hull_pts = convex_hull(pts_vec)
    hull_tuples = [(p[1], p[2]) for p in hull_pts]

    # --- STEP 5: GEOMETRY EXTRACTION ---
    dome_points = filter(p -> p[2] < (substrate_y - 5), hull_tuples)

    if isempty(dome_points)
        return nothing
    end

    sort!(dome_points, by=x -> x[1])

    x_left = dome_points[1][1]
    x_right = dome_points[end][1]

    cp_left = (x_left, substrate_y)
    cp_right = (x_right, substrate_y)

    min_y_val = minimum(p -> p[2], dome_points)
    apex_candidates = filter(p -> abs(p[2] - min_y_val) < 1.0, dome_points)
    apex_x = mean(p[1] for p in apex_candidates)
    apex = (apex_x, min_y_val)

    height_px = substrate_y - min_y_val
    base_width_px = x_right - x_left

    pad = 20
    roi_indices = (
        max(1, floor(Int, min_y_val - pad)):min(h, floor(Int, substrate_y + pad)),
        max(1, floor(Int, x_left - pad)):min(w, floor(Int, x_right + pad))
    )
    roi_img = img[roi_indices[1], roi_indices[2]]

    return DropDetectionResult(
        Float64(substrate_y),
        cp_left,
        cp_right,
        apex,
        dome_points,
        height_px,
        base_width_px,
        contour_points,
        # Pass Tuple{UnitRange, UnitRange} directly
        roi_indices,
        Matrix(roi_img)
    )
end

# ==============================================================================
# 3. PHYSICS MODELS & FITTING
# ==============================================================================

# --- Apex Spherical Cap ---
function fit_apex_spherical(det::DropDetectionResult)
    a = det.base_width_px / 2.0
    h = det.height_px

    if h <= 0 || a <= 0
        return nothing
    end

    R = (h^2 + a^2) / (2 * h)
    sin_theta = a / R
    theta_rad = asin(clamp(sin_theta, -1.0, 1.0))
    # If height > R, it's hydrophobic > 90. 
    # Simple check: if h > a, theta > 90 roughly. 
    # Exact: tan(theta/2) = h/a.
    theta_deg = rad2deg(2 * atan(h / a))

    vol = (pi * h^2 / 3) * (3 * R - h)

    return Dict(
        "theta_deg" => theta_deg,
        "radius_px" => R,
        "volume_px3" => vol
    )
end

# --- Young-Laplace ODE System ---
"""
    young_laplace_ode!(du, u, p, s)

ODE definition for the Young-Laplace equation in arc-length parametrization.
u = [r, z, phi]
p = [b, c] where b = curvature at apex, c = Δρ * g / σ (capillary constant)
"""
function young_laplace_ode!(du, u, p, s)
    r, z, phi = u
    b, c = p

    du[1] = cos(phi)      # dr/ds
    du[2] = sin(phi)      # dz/ds

    # dphi/ds = 2b - c*z - sin(phi)/r
    # Singularity handling at r -> 0
    if r < 1e-6
        du[3] = b # Limit as r->0, sin(phi)/r -> phi/r -> curvature b
    else
        du[3] = 2 * b - c * z - sin(phi) / r
    end
end

"""
    fit_young_laplace(det::DropDetectionResult; 
                      rho=1000.0, g=9.81, sigma=72e-3, 
                      pixel_size_m=1e-6, fit_sigma=false)

Fits the Young-Laplace equation to the experimental profile.
Uses DifferentialEquations.jl for high-speed integration.
"""
function fit_young_laplace(det::DropDetectionResult;
    rho=1000.0, g=9.81, sigma=72e-3,
    pixel_size_m=1e-6, fit_sigma=false)

    # Prepare data: Shift to Apex at (0,0) and convert to meters
    apex_x, apex_y = det.apex

    # Robust data collection: use all points or filter noise
    r_data = Float64[]
    z_data = Float64[]

    for p in det.dome_points
        r = abs(p[1] - apex_x) * pixel_size_m
        z = (p[2] - apex_y) * pixel_size_m
        if z >= 0
            push!(r_data, r)
            push!(z_data, z)
        end
    end

    if length(z_data) < 10
        return Dict("theta_deg" => NaN, "bond_number" => NaN, "fitted_sigma" => NaN)
    end

    # Sort data by Z height
    perm = sortperm(z_data)
    r_data = r_data[perm]
    z_data = z_data[perm]

    # Calculate Capillary constant c = Δρ * g / σ
    # If fitting sigma, c is variable. If not, c is fixed.
    c_fixed = (rho * g) / sigma

    # Initial Guess for b (1/R_apex)
    h_m = det.height_px * pixel_size_m
    w_m = det.base_width_px * pixel_size_m
    R_guess = (h_m^2 + (w_m / 2)^2) / (2 * max(h_m, 1e-12))
    b_guess = 1.0 / R_guess

    # Contact point for constraints
    z_contact_m = h_m
    r_contact_m = w_m / 2.0

    # ODE Definitions
    # s_span: integrate past contact
    s_max = max(6.0 * z_contact_m, 1e-6)
    s_span = (0.0, s_max)
    u0 = [0.0, 0.0, 0.0]

    # --- MODEL FUNCTION ---
    # Returns the full residual vector: [shape_residuals..., contact_r_res, contact_z_res]
    # We trick LsqFit by passing a dummy X of size (N_points + N_Constraints)
    # and Y = Zeros.

    # Pre-calculated weights for shape
    z_norm = z_data ./ max(z_contact_m, 1e-12)
    alpha = 3.0
    weights_shape = exp.(alpha .* z_norm)
    # Increase weight near contact (last 10%)
    weights_shape[z_norm.>0.9] .*= 3.0

    function full_residual_vector(p)
        b_val = p[1]
        c_val = fit_sigma ? p[2] : c_fixed

        # Check physical bounds (manual penalty)
        if b_val <= 0 || (fit_sigma && (c_val <= 0 || c_val > 1e10)) # sanity check
            return fill(1e6, length(z_data) + 2)
        end

        # Solve ODE
        prob = ODEProblem(young_laplace_ode!, u0, s_span, [b_val, c_val])
        # Use lower tolerance for speed then refine? kept high for robust match
        sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, save_everystep=false, dense=true)

        if sol.retcode != :Success && sol.retcode != :Terminated
            return fill(1e6, length(z_data) + 2)
        end

        # Interpolate R at data Z
        # We need r(z). The solution is parametric (r(s), z(s)).
        # We can use the interpolation provided by DiffEq solution object `sol(s, idxs=2)` gives z(s).
        # But we need inverse: find s for each z_data.
        # Monotonicity check on z? Usually sessile drops are monotonic in z until > 90 deg.
        # If > 90, z goes up then down? No, z is height from apex (0) down. Usually monotonic increasing.

        # Quick interpolation from dense solution arrays
        # Extract points for interpolation
        sol_times = range(0, sol.t[end], length=500)
        sol_vals = sol(sol_times)

        sol_r = [u[1] for u in sol_vals.u]
        sol_z = [u[2] for u in sol_vals.u]
        sol_phi = [u[3] for u in sol_vals.u]

        # Check integrity
        if maximum(sol_z) < maximum(z_data)
            # Didn't integrate deep enough
            return fill(1e6, length(z_data) + 2)
        end

        # Interpolation
        r_fit = zeros(length(z_data))

        # Simple linear interpo of r vs z
        # Assuming z is sorted and monotonic
        last_k = 1
        for i in 1:length(z_data)
            zt = z_data[i]
            # Find bracket
            found_k = false
            for k in last_k:(length(sol_z)-1)
                if sol_z[k] <= zt <= sol_z[k+1]
                    ratio = (zt - sol_z[k]) / (sol_z[k+1] - sol_z[k])
                    r_fit[i] = sol_r[k] + ratio * (sol_r[k+1] - sol_r[k])
                    last_k = k
                    found_k = true
                    break
                end
            end
            if !found_k
                r_fit[i] = 1e6 # penalty
            end
        end

        # Shape Residuals
        res_shape = weights_shape .* (r_fit .- r_data)

        # Contact Constraints
        # Find R, Phi at Z_contact
        # Interpolate
        r_c_fit = 0.0
        #phi_c_fit = 0.0 # Unused constraint for now unless we estimate angle from data
        found_c = false
        for k in 1:(length(sol_z)-1)
            if sol_z[k] <= z_contact_m <= sol_z[k+1]
                ratio = (z_contact_m - sol_z[k]) / (sol_z[k+1] - sol_z[k])
                r_c_fit = sol_r[k] + ratio * (sol_r[k+1] - sol_r[k])
                found_c = true
                break
            end
        end

        res_contact_r = found_c ? (r_c_fit - r_contact_m) : 1e6

        # End constraint (ensure integration goes far enough)
        res_end = (maximum(sol_z) < z_contact_m) ? 1e6 : 0.0

        # Weights
        w_contact = 150.0
        w_end = 10.0

        return vcat(res_shape, w_contact * res_contact_r, w_end * res_end)
    end

    # Wrap for curve_fit
    # curve_fit tries to minimize sum((f(x, p) - y)^2)
    # We define y = 0. f(x, p) = full_residual_vector(p).
    # x is dummy.
    y_target = zeros(length(z_data) + 2)
    x_dummy = zeros(length(y_target)) # Not used

    function wrapper_model(x, p)
        return full_residual_vector(p)
    end

    p0 = fit_sigma ? [b_guess, c_fixed] : [b_guess]

    # Try Fit
    try
        fit = curve_fit(wrapper_model, x_dummy, y_target, p0)
        p_opt = fit.param

        b_final = p_opt[1]
        c_final = fit_sigma ? p_opt[2] : c_fixed
        sigma_final = (rho * g) / c_final

        # Calculate Final Angle
        # Re-solve one last time to get angle
        prob_final = ODEProblem(young_laplace_ode!, u0, (0.0, s_span[2] * 2.0), [b_final, c_final])
        sol_final = solve(prob_final, Tsit5(), saveat=0.005) # dense

        sol_z = [u[2] for u in sol_final.u]
        sol_phi = [u[3] for u in sol_final.u]

        # Interpolate phi at z_contact
        idx = findfirst(z -> z >= z_contact_m, sol_z)

        theta_deg = if isnothing(idx)
            if !isempty(sol_z) && sol_z[end] > 0.95 * z_contact_m
                rad2deg(sol_phi[end])
            else
                NaN
            end
        else
            rad2deg(sol_phi[min(idx, length(sol_phi))])
        end
        Bo = c_final * (1 / b_final)^2

        return Dict(
            "theta_deg" => theta_deg,
            "bond_number" => Bo,
            "fitted_sigma" => sigma_final,
            "apex_curvature" => b_final,
            # Debug info for plotting
            "trace" => (r=[u[1] for u in sol_final.u] ./ pixel_size_m, z=[u[2] for u in sol_final.u] ./ pixel_size_m)
        )

    catch e
        println("Fit failed: $e")
        return Dict("theta_deg" => NaN, "bond_number" => NaN, "fitted_sigma" => NaN)
    end
end

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================

function plot_results(det::DropDetectionResult, results::Dict)
    # Convert image for plotting
    img_plot = plot(det.roi_img, title="Sessile Drop Analysis", axis=nothing)

    # Add geometry info
    h_str = "h: $(round(det.height_px, digits=1)) px"
    w_str = "w: $(round(det.base_width_px, digits=1)) px"

    # Overlay Apex
    scatter!([det.apex[1] - det.roi_coords[2][1]], [det.apex[2] - det.roi_coords[1][1]],
        color=:red, label="Apex", markersize=5)

    # Overlay Contact Points
    scatter!([det.cp_left[1] - det.roi_coords[2][1], det.cp_right[1] - det.roi_coords[2][1]],
        [det.cp_left[2] - det.roi_coords[1][1], det.cp_right[2] - det.roi_coords[1][1]],
        color=:blue, label="Contacts", markersize=5)

    # Create text annotation
    yl_res = get(results, "young_laplace", get(results, "Young-Laplace", nothing))
    yl_text = if isnothing(yl_res)
        "YL: N/A"
    else
        theta = get(yl_res, "theta_deg", get(yl_res, "mean_deg", NaN))
        sigma_val = get(yl_res, "fitted_sigma", nothing)

        # Plot the fit if available
        if haskey(yl_res, "trace")
            trace = yl_res["trace"]
            # Shift trace to image coordinates
            # Apex in image: det.apex
            # Trace r, z are from apex.
            # Mirror trace for left/right
            r_tr = trace[:r]
            z_tr = trace[:z]

            # Right side
            plot!(det.apex[1] .+ r_tr .- det.roi_coords[2][1],
                det.apex[2] .+ z_tr .- det.roi_coords[1][1],
                color=:orange, linewidth=2, label="YL Fit")
            # Left side
            plot!(det.apex[1] .- r_tr .- det.roi_coords[2][1],
                det.apex[2] .+ z_tr .- det.roi_coords[1][1],
                color=:orange, linewidth=2, label="")
        end

        txt = "YL Theta: $(round(theta, digits=2))"
        if !isnothing(sigma_val)
            txt *= "\nSigma: $(round(sigma_val*1000, digits=1)) mN/m"
        end
        txt
    end

    annotate!(10, 10, text("$h_str\n$w_str\n$yl_text", :left, :white, 10))

    display(img_plot)
end

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

function analyze_image(path::String; pixel_size_m=1e-6, fit_sigma=true)
    println("Processing $path...")
    det = sessile_drop_adaptive(path)

    if isnothing(det)
        println("Detection failed.")
        return
    end

    println("Geometry Detected:")
    println("  Height: $(det.height_px) px")
    println("  Base:   $(det.base_width_px) px")

    results = Dict()

    # 1. Apex Spherical
    results["Apex Spherical"] = fit_apex_spherical(det)

    # 2. Young-Laplace
    println("Running Young-Laplace fit (DifferentialEquations.jl)...")
    results["Young-Laplace"] = fit_young_laplace(det, pixel_size_m=pixel_size_m, fit_sigma=fit_sigma)

    # Print Summary
    println("\n--- RESULTS ---")
    if !isnothing(results["Apex Spherical"])
        println("Spherical Cap Angle: $(round(results["Apex Spherical"]["theta_deg"], digits=2))°")
    end

    yl = results["Young-Laplace"]
    println("Young-Laplace Angle: $(round(yl["theta_deg"], digits=2))°")
    if fit_sigma
        println("Fitted Surface Tension: $(round(yl["fitted_sigma"]*1000, digits=2)) mN/m")
    end

    # Plot
    plot_results(det, results)

    return results
end


# ==============================================================================
# 6. ADDITIONAL FITTING METHODS (Tangent, Polynomial)
# ==============================================================================

"""
    fit_tangent(det::DropDetectionResult; side=:left, n_points=20)

Estimates contact angle by fitting a 2nd-degree polynomial to the points 
nearest the contact line.
"""
function fit_tangent(det::DropDetectionResult; side=:left, n_points=20)
    cp = (side == :left) ? det.cp_left : det.cp_right
    substrate_y = det.substrate_y

    # Filter points near contact
    # Sort by distance to contact point
    pts = det.dome_points
    dists = [sqrt((p[1] - cp[1])^2 + (p[2] - cp[2])^2) for p in pts]

    # Get indices of closest points
    perm = sortperm(dists)
    closest_indices = perm[1:min(n_points, length(perm))]
    fit_pts = pts[closest_indices]

    if length(fit_pts) < 5
        return nothing
    end

    # Prepare for Linear Squares: y = ax^2 + bx + c
    # A = [x^2 x 1]
    xs = [p[1] for p in fit_pts]
    ys = [p[2] for p in fit_pts]

    A = hcat(xs .^ 2, xs, ones(length(xs)))

    # Solve coeffs = A \ ys
    coeffs = A \ ys
    a, b, c = coeffs

    # Slope at contact x: y' = 2ax + b
    slope = 2 * a * cp[1] + b

    # Angle
    angle_rad = atan(abs(slope))
    angle_deg = rad2deg(angle_rad)

    # Adjust for quadrant based on slope and side
    # Image coords: y down.
    # Left side: Slope should be negative (going up-right visually is smaller Y). 
    # Actually in image coords: 
    #   Drop goes UP (smaller Y). Left side x increases, y decreases -> slope negative.
    #   Right side x increases, y increases -> slope positive.

    if side == :left && slope > 0
        angle_deg = 180 - angle_deg
    elseif side == :right && slope < 0
        angle_deg = 180 - angle_deg
    end

    return angle_deg
end

"""
    compute_contact_angles_from_detection(det)

Aggregates all calculation methods into a single dictionary, matching the Python structure.
"""
function compute_contact_angles_from_detection(det::DropDetectionResult;
    rho=1000.0,
    sigma=72e-3,
    g=9.81,
    pixel_size_m=1e-6)
    out = Dict()

    # 1. Apex Spherical
    out["apex_spherical"] = fit_apex_spherical(det)

    # 2. Tangent
    t_left = fit_tangent(det, side=:left)
    t_right = fit_tangent(det, side=:right)
    out["tangent"] = Dict("left_deg" => t_left, "right_deg" => t_right)

    # 3. Spherical Fit (Profile based - Reusing apex logic as approximation for this port)
    # The Python code had a separate `fit_spherical_cap` that essentially did the same math
    # but strictly on width/height. We'll reuse the apex one for simplicity.
    out["spherical_fit"] = fit_apex_spherical(det)

    # 4. Young-Laplace (Fixed Sigma)
    yl_res = fit_young_laplace(det, rho=rho, g=g, sigma=sigma,
        pixel_size_m=pixel_size_m, fit_sigma=false)

    out["young_laplace"] = Dict(
        "mean_deg" => yl_res["theta_deg"],
        "bond_number" => yl_res["bond_number"]
    )

    return out
end

# ==============================================================================
# 7. COMPARISON & USAGE EXAMPLES
# ==============================================================================

function compare_yl_methods(det::DropDetectionResult; pixel_size_m=1e-6, rho=1000.0)
    println("="^60)
    println("Method 1: Known surface tension (σ = 72 mN/m)")
    println("="^60)

    # We call the fit function directly
    res1 = fit_young_laplace(det, rho=rho, sigma=72e-3,
        pixel_size_m=pixel_size_m, fit_sigma=false)

    # Note: Our fit_young_laplace returns a single mean angle in the simplified port,
    # whereas the Python code fitted left/right separately. 
    theta1 = res1["theta_deg"]
    Bo1 = res1["bond_number"]

    @printf("Contact angle: %.2f°\n", theta1)
    @printf("Bond number:   %.3f\n", Bo1)

    println("\n" * "="^60)
    println("Method 2: Unknown surface tension (fitted)")
    println("="^60)

    res2 = fit_young_laplace(det, rho=rho,
        pixel_size_m=pixel_size_m, fit_sigma=true)

    theta2 = res2["theta_deg"]
    sigma_fit = res2["fitted_sigma"]
    Bo2 = res2["bond_number"]

    @printf("Contact angle: %.2f°\n", theta2)
    @printf("Fitted σ:      %.2f mN/m\n", sigma_fit * 1000)
    @printf("Bond number:   %.3f\n", Bo2)

    println("\n" * "="^60)
    println("Comparison")
    println("="^60)
    @printf("Angle difference:      %.2f°\n", abs(theta1 - theta2))
    @printf("Surface tension error: %.2f mN/m\n", abs(sigma_fit * 1000 - 72))
end

# ==============================================================================
# 8. HELPER FUNCTIONS
# ==============================================================================

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


end # module

