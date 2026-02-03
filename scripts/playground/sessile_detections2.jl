module SessileDetections

using Images, ImageIO, ImageContrastAdjustment, ImageFiltering, ImageMorphology
using ImageBinarization, ImageSegmentation
using Statistics, LinearAlgebra, Interpolations
using Polynomials
using DifferentialEquations
using Optim
using ImageMorphology: opening, closing
using ImageContrastAdjustment: adjust_histogram, AdaptiveEqualization

const π2 = 2π

# ------------ Geometry helpers ------------ #

poly_area(xs, ys) = 0.5 * sum((xs .* circshift(ys, -1)) .- (ys .* circshift(xs, -1)))
poly_perimeter(xs, ys) = sum(hypot.(xs .- circshift(xs, -1), ys .- circshift(ys, -1)))

# CLAHE helper (adaptive histogram equalization)
function clahe(img; cliplimit=0.01, tilesize=(8, 8))
    # Map OpenCV-like parameters to ImageContrastAdjustment:
    # tilesize -> (rblocks, cblocks), cliplimit -> clip (fraction)
    rb, cb = tilesize
    adjust_histogram(img, AdaptiveEqualization(rblocks=rb, cblocks=cb, clip=cliplimit))
end

# Simple Bradley adaptive threshold (fallback if ImageBinarization export is unavailable)
function bradley_threshold(img; window=21, t=0.15)
    w = max(3, window)
    kernel = fill(1.0 / (w * w), w, w)
    local_mean = imfilter(img, kernel, Pad(:replicate))
    img .< local_mean .* (1 - t)
end

# Simple monotone-chain convex hull; returns indices into the original points matrix
function convex_hull_indices(points::AbstractMatrix)
    n = size(points, 1)
    n <= 1 && return collect(1:n)
    pts = [(float(points[i, 1]), float(points[i, 2]), i) for i in 1:n]
    sort!(pts, by = p -> (p[1], p[2]))
    cross(o, a, b) = (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])

    lower = Tuple{Float64, Float64, Int}[]
    for p in pts
        while length(lower) >= 2 && cross(lower[end-1], lower[end], p) <= 0
            pop!(lower)
        end
        push!(lower, p)
    end

    upper = Tuple{Float64, Float64, Int}[]
    for p in reverse(pts)
        while length(upper) >= 2 && cross(upper[end-1], upper[end], p) <= 0
            pop!(upper)
        end
        push!(upper, p)
    end

    hull = vcat(lower[1:end-1], upper[1:end-1])
    return [p[3] for p in hull]
end

# ------------ Core contact-angle models ------------ #

function contact_angle_from_apex(apex::Tuple, cp_left::Tuple, cp_right::Tuple, substrate_y::Real)
    apex_x, apex_y = apex
    xL, _ = cp_left
    xR, _ = cp_right

    height_px = substrate_y - apex_y
    base_width_px = abs(xR - xL)
    (height_px <= 0 || base_width_px <= 0) && return nothing

    a = base_width_px / 2
    R = (height_px^2 + a^2) / (2 * height_px)
    num = a^2 - height_px^2
    den = a^2 + height_px^2
    den == 0 && return nothing

    cosθ = clamp(num / den, -1, 1)
    θ_rad = acos(cosθ)
    θ_deg = rad2deg(θ_rad)
    volume_px3 = π * height_px^2 * (3R - height_px) / 3

    return Dict(
        "theta_deg" => θ_deg,
        "height_px" => float(height_px),
        "base_width_px" => float(base_width_px),
        "radius_px" => float(R),
        "volume_px3" => float(volume_px3),
    )
end

function calculate_contact_angle_tangent(contour_points, contact_point, substrate_y; side="left", fit_points=20)
    cp_x, cp_y = contact_point
    dx = contour_points[:, 1] .- cp_x
    dy = contour_points[:, 2] .- cp_y
    distances = hypot.(dx, dy)

    nearby = distances .< 50 .&& contour_points[:, 2] .< substrate_y - 3
    pts = contour_points[nearby, :]
    size(pts, 1) < 5 && return nothing

    pts = pts[sortperm(pts[:, 1]), :]
    if side == "left"
        subset = pts[pts[:, 1] .>= cp_x, :]
        fit_pts = subset[1:min(fit_points, size(subset, 1)), :]
    else
        subset = pts[pts[:, 1] .<= cp_x, :]
        n = size(subset, 1)
        fit_pts = subset[max(n - fit_points + 1, 1):n, :]
    end
    size(fit_pts, 1) < 5 && return nothing

    try
        p = fit(fit_pts[:, 1], fit_pts[:, 2], 2)
        slope = 2 * coeffs(p)[3] * cp_x + coeffs(p)[2]  # coeffs order: c0 + c1 x + c2 x^2
        angle_rad = atan(abs(slope))
        angle_deg = rad2deg(angle_rad)
        if (side == "left" && slope > 0) || (side == "right" && slope < 0)
            angle_deg = 180 - angle_deg
        end
        return angle_deg
    catch
        return nothing
    end
end

function calculate_contact_angle_tangent_from_apex(contour_points, cp, apex, substrate_y; n_points=30)
    cp_x, cp_y = cp
    apex_x, apex_y = apex
    pts = float.(contour_points)
    v = [apex_x - cp_x, apex_y - cp_y]
    norm(v) == 0 && return nothing
    v ./= norm(v)

    rel = pts .- Ref([cp_x, cp_y])
    proj = rel * v
    mask = proj .>= 0
    proj_valid = proj[mask]
    pts_valid = pts[mask, :]
    length(pts_valid) < 5 && return nothing

    idx = sortperm(proj_valid)[1:min(n_points, end)]
    fit_pts = pts_valid[idx, :]
    size(fit_pts, 1) < 2 && return nothing

    x_fit = fit_pts[:, 1]; y_fit = fit_pts[:, 2]
    A = hcat(x_fit, ones(length(x_fit)))
    m, _ = A \ y_fit
    angle_rad = atan(-m)
    return abs(rad2deg(angle_rad))
end

function fit_spherical_cap(contour_points, cp_left, cp_right, substrate_y)
    points = copy(contour_points)
    base_width = abs(cp_right[1] - cp_left[1])
    min_y = minimum(points[:, 2])
    height = substrate_y - min_y
    (height <= 0 || base_width <= 0) && return nothing, nothing, nothing

    R = (height^2 + (base_width / 2)^2) / (2 * height)
    sinθ = clamp((base_width / 2) / R, 0, 1)
    θ_deg = rad2deg(asin(sinθ))
    volume = π * height^2 * (3R - height) / 3
    return θ_deg, R, volume
end

# Direct least-squares ellipse fit (Fitzgibbon et al.)
struct EllipseParams
    cx::Float64
    cy::Float64
    a::Float64
    b::Float64
    phi::Float64
end

function fit_ellipse(points::AbstractMatrix)
    x = points[:, 1]; y = points[:, 2]
    D = hcat(x.^2, x .* y, y.^2, x, y, ones(length(x)))
    S = D' * D
    C = zeros(6, 6)
    C[1, 3] = 2; C[2, 2] = -1; C[3, 1] = 2
    eigvals, eigvecs = eigen(S, C)
    v = eigvecs[:, findfirst(>(0), diag(eigvecs' * C * eigvecs))]
    A, B, Cc, Dd, Ee, Ff = v
    den = B^2 - 4A*Cc
    den == 0 && return nothing
    cx = (2Cc*Dd - B*Ee) / den
    cy = (2A*Ee - B*Dd) / den
    up = 2 * (A*cx^2 + Cc*cy^2 + B*cx*cy - Ff)
    term = sqrt((A - Cc)^2 + B^2)
    a = sqrt(up / ((A + Cc) - term))
    b = sqrt(up / ((A + Cc) + term))
    phi = 0.5 * atan(B, A - Cc)
    return EllipseParams(cx, cy, a, b, phi)
end

function fit_elliptical(contour_points, cp_left, cp_right, substrate_y)
    points = copy(contour_points)
    size(points, 1) < 5 && return nothing, nothing, nothing, nothing, nothing
    try
        e = fit_ellipse(points)
        e === nothing && return nothing, nothing, nothing, nothing, nothing
        dx_left = cp_left[1] - e.cx
        dx_right = cp_right[1] - e.cx
        dy = substrate_y - e.cy

        slope_left = dy != 0 ? -(e.b^2 * dx_left) / (e.a^2 * dy) : Inf
        angle_left = dy != 0 ? rad2deg(atan(abs(slope_left))) : 90.0
        if slope_left > 0; angle_left = 180 - angle_left; end

        slope_right = dy != 0 ? -(e.b^2 * dx_right) / (e.a^2 * dy) : Inf
        angle_right = dy != 0 ? rad2deg(atan(abs(slope_right))) : 90.0
        if slope_right < 0; angle_right = 180 - angle_right; end

        volume = 4 / 3 * π * e.a^2 * e.b
        return angle_left, angle_right, e.a, e.b, volume
    catch
        return nothing, nothing, nothing, nothing, nothing
    end
end

# Young–Laplace fit helpers
function yl_ode!(du, u, p, s)
    r, z, phi = u
    b, rho, g, sigma = p
    du[1] = cos(phi)
    du[2] = sin(phi)
    kterm = r <= 1e-12 ? 0.0 : sin(phi) / r
    du[3] = 2b - (rho * g / sigma) * z - kterm
end

function integrate_profile(b, sigma, rho, g, zmax; s_points=500)
    u0 = [0.0, 0.0, 0.0]
    p = (b, rho, g, sigma)
    sspan = (0.0, max(5zmax, 1e-6))
    prob = ODEProblem(yl_ode!, u0, sspan, p)
    sol = solve(prob, Tsit5(); abstol=1e-9, reltol=1e-9, saveat=range(sspan[1], sspan[2], length=s_points))
    return sol[1, :], sol[2, :], sol[3, :]
end

function fit_young_laplace(contour_points, apex, cp_left, cp_right, substrate_y;
                           rho=1000.0, sigma=72e-3, g=9.81, pixel_size_m=1e-6,
                           n_fit_points=100, debug=false)
    contour_points === nothing && return nothing, nothing, nothing, nothing
    apex === nothing && return nothing, nothing, nothing, nothing
    cp_left === nothing && return nothing, nothing, nothing, nothing
    cp_right === nothing && return nothing, nothing, nothing, nothing

    pts = float.(contour_points)
    apex_x, apex_y = apex
    clx, cly = cp_left
    crx, cry = cp_right
    substrate_y <= apex_y && return nothing, nothing, nothing, nothing

    base_width_px = abs(crx - clx)
    height_px = substrate_y - apex_y
    (base_width_px <= 0 || height_px <= 0) && return nothing, nothing, nothing, nothing
    base_center_x = 0.5 * (clx + crx)

    half_base_m = 0.5 * base_width_px * pixel_size_m
    height_m = height_px * pixel_size_m
    R0_guess = (half_base_m^2 + height_m^2) / (2 * max(height_m, 1e-12))
    b0_global = 1 / R0_guess

    function fit_side(side)
        mask = side == :right ? pts[:, 1] .>= base_center_x : pts[:, 1] .<= base_center_x
        cx, cy = side == :right ? (crx, cry) : (clx, cly)
        sign = side == :right ? 1.0 : -1.0
        pts_side = pts[mask .& (pts[:, 2] .<= substrate_y .+ 1e-6), :]
        size(pts_side, 1) < 10 && return nothing
        pts_side = pts_side[sortperm(pts_side[:, 2]), :]

        z_data = (pts_side[:, 2] .- apex_y) .* pixel_size_m
        r_data = sign .* (pts_side[:, 1] .- base_center_x) .* pixel_size_m
        posmask = (z_data .>= 0) .& (r_data .>= 0)
        z_data = z_data[posmask]; r_data = r_data[posmask]
        length(z_data) < 10 && return nothing

        z_contact = (cy - apex_y) * pixel_size_m
        r_contact = sign * (cx - base_center_x) * pixel_size_m
        (z_contact <= 0 || r_contact <= 0) && return nothing

        if length(z_data) > n_fit_points
            idx = round.(Int, range(1, length(z_data), length=n_fit_points))
            z_data = z_data[idx]; r_data = r_data[idx]
        end

        function residual(b)
            b <= 0 && return fill(1e6, length(z_data) + 3)
            zmax = max(max(z_data), z_contact)
            r_sol, z_sol, phi_sol = integrate_profile(b, sigma, rho, g, zmax)
            z_sol[end] < z_contact && return fill(1e6, length(z_data) + 3)

            r_fit = LinearInterpolation(z_sol, r_sol; extrapolation_bc=NaN)(z_data)
            any(isnan, r_fit) && return fill(1e6, length(z_data) + 3)

            z_norm = z_data ./ max(z_contact, 1e-12)
            w_shape = exp.(3 .* z_norm)
            w_shape[z_norm .> 0.9] .*= 3
            res_shape = w_shape .* (r_fit .- r_data)

            r_fit_c = LinearInterpolation(z_sol, r_sol; extrapolation_bc=NaN)(z_contact)
            phi_fit_c = LinearInterpolation(z_sol, phi_sol; extrapolation_bc=NaN)(z_contact)
            any(isnan, (r_fit_c, phi_fit_c)) && return fill(1e6, length(z_data) + 3)

            near = (z_data .> 0.85z_contact) .& (z_data .< z_contact)
            if sum(near) >= 3
                p_line = fit(z_data[near], r_data[near], 1)
                dr_dz = coeffs(p_line)[2]
                θ_data = atan(1 / dr_dz)
                res_phi = phi_fit_c - θ_data
                w_phi = 50.0
            else
                res_phi = 0.0; w_phi = 0.0
            end
            res_r = r_fit_c - r_contact
            res_end = z_sol[end] - z_contact
            return vcat(res_shape, 150.0 * res_r, w_phi * res_phi, 10.0 * res_end)
        end

        opt = optimize(b -> sum(abs2, residual(b)), 1e-6, 1e4, b0_global; reltol=1e-9)
        b_opt = Optim.minimizer(opt)
        zmax = max(max(z_data), z_contact)
        r_sol, z_sol, phi_sol = integrate_profile(b_opt, sigma, rho, g, zmax; s_points=600)
        phi_contact = LinearInterpolation(z_sol, phi_sol)(z_contact)
        θ_deg = rad2deg(phi_contact)
        return θ_deg, b_opt
    end

    res_r = fit_side(:right)
    res_l = fit_side(:left)
    res_r === nothing && res_l === nothing && return nothing, nothing, nothing, nothing

    θ_r = res_r === nothing ? nothing : res_r[1]
    b_r = res_r === nothing ? nothing : res_r[2]
    θ_l = res_l === nothing ? nothing : res_l[1]
    b_l = res_l === nothing ? nothing : res_l[2]

    valid_b = filter(!isnothing, [b_r, b_l])
    isempty(valid_b) && return θ_l, θ_r, nothing, nothing
    b_mean = mean(valid_b)
    R0 = 1 / b_mean
    capillary_length = sqrt(sigma / (rho * g))
    Bo = rho * g * R0^2 / sigma
    return θ_l, θ_r, capillary_length, Bo
end

# Unknown-sigma fit mirrors the above but optimizes over (b, sigma)
function fit_young_laplace_unknown_sigma(contour_points, apex, cp_left, cp_right, substrate_y;
                                         rho=1000.0, g=9.81, pixel_size_m=1e-6,
                                         n_fit_points=100, debug=false)
    contour_points === nothing && return nothing, nothing, nothing, nothing, nothing
    apex === nothing && return nothing, nothing, nothing, nothing, nothing
    cp_left === nothing && return nothing, nothing, nothing, nothing, nothing
    cp_right === nothing && return nothing, nothing, nothing, nothing, nothing

    pts = float.(contour_points)
    apex_x, apex_y = apex
    clx, cly = cp_left
    crx, cry = cp_right
    substrate_y <= apex_y && return nothing, nothing, nothing, nothing, nothing
    base_width_px = abs(crx - clx)
    height_px = substrate_y - apex_y
    (base_width_px <= 0 || height_px <= 0) && return nothing, nothing, nothing, nothing, nothing
    base_center_x = 0.5 * (clx + crx)

    half_base_m = 0.5 * base_width_px * pixel_size_m
    height_m = height_px * pixel_size_m
    R0_guess = (half_base_m^2 + height_m^2) / (2 * max(height_m, 1e-12))
    b0_guess = 1 / R0_guess
    σ_guess = 72e-3

    function fit_side(side)
        mask = side == :right ? pts[:, 1] .>= base_center_x : pts[:, 1] .<= base_center_x
        cx, cy = side == :right ? (crx, cry) : (clx, cly)
        sign = side == :right ? 1.0 : -1.0
        pts_side = pts[mask .& (pts[:, 2] .<= substrate_y .+ 1e-6), :]
        size(pts_side, 1) < 10 && return nothing
        pts_side = pts_side[sortperm(pts_side[:, 2]), :]

        z_data = (pts_side[:, 2] .- apex_y) .* pixel_size_m
        r_data = sign .* (pts_side[:, 1] .- base_center_x) .* pixel_size_m
        posmask = (z_data .>= 0) .& (r_data .>= 0)
        z_data = z_data[posmask]; r_data = r_data[posmask]
        length(z_data) < 10 && return nothing

        z_contact = (cy - apex_y) * pixel_size_m
        r_contact = sign * (cx - base_center_x) * pixel_size_m
        (z_contact <= 0 || r_contact <= 0) && return nothing

        if length(z_data) > n_fit_points
            idx = round.(Int, range(1, length(z_data), length=n_fit_points))
            z_data = z_data[idx]; r_data = r_data[idx]
        end

        function residual(p)
            b = p[1]; σ = p[2]
            (b <= 0 || σ <= 1e-3 || σ >= 0.2) && return fill(1e6, length(z_data) + 3)
            zmax = max(max(z_data), z_contact)
            r_sol, z_sol, phi_sol = integrate_profile(b, σ, rho, g, zmax)
            z_sol[end] < z_contact && return fill(1e6, length(z_data) + 3)
            r_fit = LinearInterpolation(z_sol, r_sol; extrapolation_bc=NaN)(z_data)
            any(isnan, r_fit) && return fill(1e6, length(z_data) + 3)
            z_norm = z_data ./ max(z_contact, 1e-12)
            w_shape = exp.(3 .* z_norm); w_shape[z_norm .> 0.9] .*= 3
            res_shape = w_shape .* (r_fit .- r_data)
            r_fit_c = LinearInterpolation(z_sol, r_sol; extrapolation_bc=NaN)(z_contact)
            phi_fit_c = LinearInterpolation(z_sol, phi_sol; extrapolation_bc=NaN)(z_contact)
            any(isnan, (r_fit_c, phi_fit_c)) && return fill(1e6, length(z_data) + 3)
            near = (z_data .> 0.85z_contact) .& (z_data .< z_contact)
            if sum(near) >= 3
                p_line = fit(z_data[near], r_data[near], 1)
                dr_dz = coeffs(p_line)[2]
                θ_data = atan(1 / dr_dz)
                res_phi = phi_fit_c - θ_data
                w_phi = 50.0
            else
                res_phi = 0.0; w_phi = 0.0
            end
            res_r = r_fit_c - r_contact
            res_end = z_sol[end] - z_contact
            return vcat(res_shape, 150.0 * res_r, w_phi * res_phi, 10.0 * res_end)
        end

        opt = optimize(p -> sum(abs2, residual(p)), [b0_guess, σ_guess],
                       lower=[1e-6, 1e-3], upper=[1e4, 0.2], Fminbox(); reltol=1e-9)
        p_opt = Optim.minimizer(opt)
        b_opt, σ_opt = p_opt
        zmax = max(max(z_data), z_contact)
        r_sol, z_sol, phi_sol = integrate_profile(b_opt, σ_opt, rho, g, zmax; s_points=600)
        φ_c = LinearInterpolation(z_sol, phi_sol)(z_contact)
        θ_deg = rad2deg(φ_c)
        return θ_deg, b_opt, σ_opt
    end

    res_r = fit_side(:right)
    res_l = fit_side(:left)
    res_r === nothing && res_l === nothing && return nothing, nothing, nothing, nothing, nothing

    valid_σ = Float64[]
    valid_b = Float64[]
    θ_r = θ_l = nothing
    if res_r !== nothing
        θ_r, b_r, σ_r = res_r; push!(valid_b, b_r); push!(valid_σ, σ_r)
    end
    if res_l !== nothing
        θ_l, b_l, σ_l = res_l; push!(valid_b, b_l); push!(valid_σ, σ_l)
    end
    isempty(valid_σ) && return θ_l, θ_r, nothing, nothing, nothing
    σ_mean = mean(valid_σ); b_mean = mean(valid_b)
    R0 = 1 / b_mean
    capillary_length = sqrt(σ_mean / (rho * g))
    Bo = rho * g * R0^2 / σ_mean
    return θ_l, θ_r, σ_mean, capillary_length, Bo
end

# ------------- Detection pipeline (Images.jl) ------------- #

function estimate_horizon(gray)
    # Robust horizon using vertical gradient of central columns
    h, w = size(gray)
    col_start = Int(round(0.3w)); col_end = Int(round(0.7w))
    prof = mean(gray[:, col_start:col_end], dims=2)[:]
    g = abs.(diff(prof))
    search_start = Int(round(0.25h))  # ignore top quarter
    if search_start >= length(g)
        return nothing
    end
    idx_rel = findmax(g[search_start:end])[2]
    return search_start + idx_rel - 1
end

function sessile_drop_adaptive(image_path::AbstractString; return_debug=false)
    img = load(image_path)
    gray = float.(Gray.(img))
    height, width = size(gray)
    # cliplimit in AdaptiveEqualization expects a small number (0–1); 0.01 approximates OpenCV's 2.0 behavior
    enhanced = clahe(gray; cliplimit=0.01, tilesize=(8, 8))
    blur = imfilter(enhanced, Kernel.gaussian((5, 5)))
    binary = bradley_threshold(blur; window=51, t=0.25)
    # suppress top margin (needle region) and bottom safety edge
    top_mask = Int(round(0.1 * height))
    binary[1:top_mask, :] .= false
    binary[height:-1:max(height - 2, 1), :] .= false  # safety edge

    substrate_y = estimate_horizon(enhanced)
    substrate_y = substrate_y === nothing ? Int(round(height * 0.85)) : substrate_y

    binary[substrate_y:end, :] .= false
    kernel = trues(3, 3)
    # Apply opening then closing (two passes approximate iterations=2)
    binary_clean = opening(binary, kernel)
    binary_clean = opening(binary_clean, kernel)
    binary_clean = closing(binary_clean, kernel)
    binary_clean = closing(binary_clean, kernel)

    labels = label_components(binary_clean)
    nlabels = maximum(labels)
    nlabels == 0 && return nothing

    components = [(lab=lab,
                   inds=findall(labels .== lab)) for lab in 1:nlabels]

    function comp_stats(comp)
        inds = comp[:inds]
        ys = [I[1] for I in inds]; xs = [I[2] for I in inds]
        area = length(inds)
        return (xs=xs, ys=ys, area=area,
                bbox=(minimum(xs), minimum(ys), maximum(xs), maximum(ys)),
                center=(mean(xs), mean(ys)))
    end

    stats = [comp_stats(c) for c in components]
    # Filter: area >0.5% image and not touching borders
    valid = filter(s -> s.area > 0.005 * width * height &&
                        s.bbox[1] > 5 && s.bbox[3] < width - 5, stats)
    drop = isempty(valid) ? stats[argmax([s.area for s in stats])] :
                            valid[argmax([s.area for s in valid])]

    xs = drop.xs; ys = drop.ys
    points = hcat(xs, ys)
    hull_idx = convex_hull_indices(points)
    hull = points[hull_idx, :]

    dome_points = [p for p in eachrow(hull) if p[2] < substrate_y - 5]
    isempty(dome_points) && return nothing
    dome_points = sort(dome_points, by = p -> p[1])
    x_left = dome_points[1][1]; x_right = dome_points[end][1]
    cp_left = (x_left, substrate_y); cp_right = (x_right, substrate_y)
    apex_y = minimum(getindex.(dome_points, 2))
    apex_candidates = filter(p -> p[2] == apex_y, dome_points)
    apex_x = round(Int, mean(getindex.(apex_candidates, 1)))
    height_px = substrate_y - apex_y
    base_width = x_right - x_left
    roi_coords = (max(1, x_left - 20),
                  max(1, minimum(getindex.(dome_points, 2)) - 20),
                  min(width, x_right + 20),
                  min(height, substrate_y + 20))

    dome_points_array = reduce(vcat, [reshape(collect(p), 1, 2) for p in dome_points])
    final_cnt = vcat([cp_left], dome_points, [cp_right])

    out = Dict(
        "substrate_y" => substrate_y,
        "cp_left" => cp_left,
        "cp_right" => cp_right,
        "apex" => (apex_x, apex_y),
        "height_px" => float(height_px),
        "base_width_px" => float(base_width),
        "roi_coords" => roi_coords,
        "dome_points_array" => dome_points_array,
        "drop_contour" => final_cnt
    )
    if return_debug
        out["img"] = img
        out["enhanced_gray"] = enhanced
        out["binary_clean"] = binary_clean
    end
    return out
end

function compute_contact_angles_from_detection(det_result; rho=1000.0, sigma=72e-3, g=9.81,
                                               pixel_size_m=1e-6, yl_debug=false, yl_debug_path=nothing)
    det_result === nothing && return Dict()
    apex = det_result["apex"]; cp_left = det_result["cp_left"]; cp_right = det_result["cp_right"]
    substrate_y = det_result["substrate_y"]; dome = det_result["dome_points_array"]
    out = Dict{String, Any}()

    if !(apex === nothing || cp_left === nothing || cp_right === nothing || substrate_y === nothing)
        out["apex_spherical"] = contact_angle_from_apex(apex, cp_left, cp_right, substrate_y)
    end
    dome === nothing && return out

    angle_left = calculate_contact_angle_tangent(dome, cp_left, substrate_y; side="left")
    angle_right = calculate_contact_angle_tangent(dome, cp_right, substrate_y; side="right")
    out["tangent"] = Dict("left_deg" => angle_left, "right_deg" => angle_right)

    θ_sph, R_sph, vol_sph = fit_spherical_cap(dome, cp_left, cp_right, substrate_y)
    out["spherical_fit"] = Dict("theta_deg" => θ_sph, "radius_px" => R_sph, "volume_px3" => vol_sph)

    θ_el_l, θ_el_r, a_el, b_el, vol_el = fit_elliptical(dome, cp_left, cp_right, substrate_y)
    mean_el = (θ_el_l !== nothing && θ_el_r !== nothing) ? 0.5 * (θ_el_l + θ_el_r) : nothing
    out["ellipse_fit"] = Dict("left_deg" => θ_el_l, "right_deg" => θ_el_r, "mean_deg" => mean_el,
                              "a_px" => a_el, "b_px" => b_el, "volume_px3" => vol_el)

    θ_l, θ_r, cap_len, Bo = fit_young_laplace(dome, apex, cp_left, cp_right, substrate_y;
                                              rho=rho, sigma=sigma, g=g, pixel_size_m=pixel_size_m,
                                              n_fit_points=100, debug=yl_debug)
    mean_yl = (θ_l !== nothing && θ_r !== nothing) ? 0.5 * (θ_l + θ_r) : nothing
    out["young_laplace"] = Dict("left_deg" => θ_l, "right_deg" => θ_r, "mean_deg" => mean_yl,
                                "capillary_length" => cap_len, "bond_number" => Bo)
    return out
end

function calculate_drop_metrics(contour, cp_left, cp_right, substrate_y)
    xs = contour[:, 1]; ys = contour[:, 2]
    base_width = abs(cp_right[1] - cp_left[1])
    min_idx = argmin(ys); highest = (xs[min_idx], ys[min_idx])
    drop_height = substrate_y - highest[2]
    area = abs(poly_area(xs, ys))
    aspect_ratio = base_width > 0 ? drop_height / base_width : 0
    perimeter = poly_perimeter(xs, ys)
    volume = 0.0
    for y in unique(ys)
        y >= substrate_y && continue
        row = xs[ys .== y]
        length(row) >= 2 || continue
        width = maximum(row) - minimum(row)
        r = width / 2
        volume += π * r^2
    end
    return Dict(
        "base_width" => base_width,
        "height" => drop_height,
        "area" => area,
        "perimeter" => perimeter,
        "aspect_ratio" => aspect_ratio,
        "volume_estimate" => volume,
        "apex_position" => highest
    )
end

# ------------- Demo run ------------- #
if abspath(PROGRAM_FILE) == @__FILE__
    det = sessile_drop_adaptive("./data/samples/prueba sesil 2.png"; return_debug=true)
    println("\n--- JULIA DETECTION OUTPUT ---")
    if det !== nothing
        # Compact summary (avoid dumping arrays)
        println("substrate_y    = ", det["substrate_y"])
        println("cp_left        = ", det["cp_left"], "   cp_right = ", det["cp_right"])
        println("apex           = ", det["apex"])
        println("height_px      = ", det["height_px"], "   base_width_px = ", det["base_width_px"])
        println("roi_coords     = ", det["roi_coords"])
        println("dome_points    = ", size(det["dome_points_array"], 1), " points")
        println("drop_contour   = ", length(det["drop_contour"]), " vertices")

        angles = compute_contact_angles_from_detection(det; pixel_size_m=2.88e-5, rho=1000.0)
        println("\nComputed angles (compact):")
        println("  apex_spherical: ", angles["apex_spherical"])
        println("  tangent:        ", angles["tangent"])
        println("  spherical_fit:  ", angles["spherical_fit"])
        println("  ellipse_fit:    ", angles["ellipse_fit"])
        println("  young_laplace:  ", angles["young_laplace"])
        # Optional plots if Plots.jl is available
        try
            # Force GR to PNG backend to avoid needing a GUI/gksqt
            ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
            @eval using Plots
            p1 = plot(det["img"], title="Original", axis=false, ticks=false)
            p2 = heatmap(det["enhanced_gray"]; aspect_ratio=:equal, title="Enhanced", yflip=true, colorbar=false)
            p3 = heatmap(det["binary_clean"]; aspect_ratio=:equal, title="Binary", yflip=true, colorbar=false)
            cnt = det["drop_contour"]; xs = [p[1] for p in cnt]; ys = [p[2] for p in cnt]
            p4 = plot(det["img"], title="Contour overlay", axis=false, ticks=false)
            plot!(p4, xs, ys; lw=2, color=:lime)
            scatter!(p4, [det["cp_left"][1], det["cp_right"][1]], [det["cp_left"][2], det["cp_right"][2]]; color=:red, ms=5, label="contacts")
            scatter!(p4, [det["apex"][1]], [det["apex"][2]]; color=:blue, ms=6, label="apex")
            fig = plot(p1, p2, p3, p4; layout=(2,2), size=(900,700))
            savefig(fig, "sessile_demo.png")
            println("Saved plot to sessile_demo.png (PNG backend, no GUI needed)")
        catch err
            @warn "Skipping plots (Plots.jl not available or GR backend failed)" err
        end
    else
        @warn "Detection failed"
    end
end

end # module
