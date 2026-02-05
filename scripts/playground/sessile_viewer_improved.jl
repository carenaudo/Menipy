#!/usr/bin/env julia
# sessile_viewer.jl – Stand‑alone sessile drop viewer (no OpenCV)

using ArgParse
using FileIO, ImageIO
using Images, ImageFiltering, ImageFeatures, ImageContrastAdjustment
using ImageMorphology
using ImageBinarization
using Contour
using StagedFilters
using GLMakie
using Statistics
using LinearAlgebra
using BenchmarkTools
using Printf

# ---------- Helpers ----------
# Gray.(img) already yields a 2D array for grayscale inputs; avoid channelview indexing quirks.
to_gray(img) = Float32.(Gray.(img))
function apply_roi!(mask::AbstractMatrix{Bool}, left_col::Int, right_col::Int)
    if left_col > 1
        mask[:, 1:(left_col - 1)] .= false
    end
    if right_col < size(mask, 2)
        mask[:, (right_col + 1):end] .= false
    end
    return mask
end
function get_all_contours(mask::AbstractMatrix{Bool})
    """Extract all contours from the mask and return them sorted by length"""
    x = 1:size(mask, 1)
    y = 1:size(mask, 2)
    cs = contours(x, y, Float32.(mask), [0.5])

    # store (contour, length) with concrete types
    all_contours = Tuple{Matrix{Float32},Int}[]

    for lvl in Contour.levels(cs)
        for ln in Contour.lines(lvl)
            xs, ys = Contour.coordinates(ln)
            n = length(xs)
            if n > 10
                c = Array{Float32}(undef, n, 2)
                @inbounds for i in 1:n
                    # Contour: (row,col) => here (x=col,y=row)
                    c[i, 1] = ys[i]
                    c[i, 2] = xs[i]
                end
                push!(all_contours, (c, n))
            end
        end
    end

    sort!(all_contours; by=c -> c[2], rev=true)
    return [c[1] for c in all_contours]
end

function filter_components_by_size!(
    mask::AbstractMatrix{Bool},
    labels;
    min_fraction::Float64=0.01,
)
    rows, cols = size(mask)
    maxlbl = maximum(labels)
    maxlbl == 0 && return mask

    counts = zeros(Int, maxlbl)

    @inbounds for lbl in labels
        if lbl > 0
            counts[lbl] += 1
        end
    end

    min_size = round(Int, rows * cols * min_fraction)

    @inbounds for idx in eachindex(labels, mask)
        lbl = labels[idx]
        if lbl > 0 && counts[lbl] < min_size
            mask[idx] = false
        end
    end

    return mask
end
function savgol_smooth(y::AbstractVector, window::Int, poly::Int)
    window < 3 && return copy(y)
    isodd(window) || error("Savgol window must be odd")
    poly >= window && error("Savgol poly must be < window")
    m = (window - 1) ÷ 2
    out = similar(y, Float64)
    StagedFilters.smooth!(SavitzkyGolayFilter{m,poly}, y, out)
    return out
end

function savgol_coeffs(window::Int, poly::Int; deriv::Int=0)
    m = (window - 1) ÷ 2
    xs = collect(-m:m)
    A = [x^k for x in xs, k in 0:poly]
    b = zeros(Float64, poly + 1)
    if deriv <= poly
        b[deriv+1] = factorial(deriv)
    end
    return A * (A' * A \ b)
end

function savgol_derivative(y::AbstractVector, window::Int, poly::Int; deriv::Int=1)
    window < 3 && return copy(y)
    isodd(window) || error("Savgol window must be odd")
    poly >= window && error("Savgol poly must be < window")

    coeffs = savgol_coeffs(window, poly; deriv=deriv)
    m = (window - 1) ÷ 2
    n = length(y)
    out = similar(y, Float64, n)

    @inbounds for i in 1:n
        acc = 0.0
        @inbounds @simd for j in -m:m
            idx = i + j
            if idx < 1
                idx += n
            elseif idx > n
                idx -= n
            end
            acc += coeffs[j+m+1] * y[idx]
        end
        out[i] = acc
    end
    return out
end


function estimate_substrate(gray)
    # Improved baseline detection using gradient analysis
    prof = vec(mean(gray, dims=2))

    # Smooth the profile to reduce noise
    w = min(15, isodd(length(prof)) ? length(prof) : length(prof) - 1)
    if w >= 3
        prof_smooth = savgol_smooth(prof, w, 2)
    else
        prof_smooth = prof
    end

    # Find the strongest edge in the lower 2/3 of the image
    start_idx = round(Int, length(prof_smooth) * 0.33)
    dy = diff(prof_smooth[start_idx:end])

    # Look for positive gradient (dark to bright transition)
    idx = findmax(dy)[2] + start_idx - 1

    return idx
end

function detect_mask(gray; detector="auto", block_size=21, denoise=true)
    # Optional denoising step
    gray_proc = gray
    if denoise
        # Apply Gaussian blur to reduce noise
        gray_proc = imfilter(gray, Kernel.gaussian(1.5))
    end

    if detector == "canny"
        edges = canny(gray_proc, (0.20, 0.10), 1.0)
        return edges
    elseif detector == "otsu"
        t = Images.otsu_threshold(gray_proc)
        return gray_proc .< t
    elseif detector == "adaptive"
        k = ones(block_size, block_size) / (block_size^2)
        mean_img = imfilter(gray_proc, k, "replicate")
        return gray_proc .< mean_img .- 0.02
    elseif detector == "threshold"
        return gray_proc .< 0.5
    else # auto: try otsu then fall back to adaptive
        m = detect_mask(gray_proc; detector="otsu", denoise=false)
        if !any(m)
            m = detect_mask(gray_proc; detector="adaptive", block_size=block_size, denoise=false)
        end
        return m
    end
end

function is_droplet_shaped(contour, substrate_y, image_width)
    """Heuristic check if contour has droplet-like properties"""
    isempty(contour) && return false

    # 1. Check if contour has points above substrate
    min_y = minimum(contour[:, 2])
    if min_y >= substrate_y - 10
        return false  # Too close to or below substrate
    end

    # 2. Check width - droplet should be reasonably wide
    min_x = minimum(contour[:, 1])
    max_x = maximum(contour[:, 1])
    width = max_x - min_x
    if width < image_width * 0.05  # Less than 5% of image width
        return false
    end

    # 3. Check aspect ratio - height vs width
    height = substrate_y - min_y
    aspect_ratio = height / width
    if aspect_ratio < 0.1 || aspect_ratio > 3.0  # Extreme ratios
        return false
    end

    # 4. Check if contour is mostly in upper part of image
    avg_y = mean(contour[:, 2])
    if avg_y > substrate_y + 10
        return false  # Contour is below substrate
    end

    return true
end

function largest_contour(mask)
    """Get the largest contour from mask"""
    x = 1:size(mask, 1)
    y = 1:size(mask, 2)
    cs = contours(x, y, Float32.(mask), [0.5])
    best = nothing
    best_len = 0
    for lvl in Contour.levels(cs)
        for ln in Contour.lines(lvl)
            xs, ys = Contour.coordinates(ln)
            n = length(xs)
            if n > best_len
                best_len = n
                # Contour.jl uses (row, col) axes; swap to (x=col, y=row) for image coords.
                best = hcat(ys, xs)
            end
        end
    end
    return best === nothing ? Array{Float64}(undef, 0, 2) : Array{Float64}(best)
end

function find_contact_points(x::AbstractVector, y::AbstractVector, substrate_y)
    n = length(x)
    @assert length(y) == n

    function locate(left_to_right::Bool)
        if left_to_right
            first_i = 1
            last_i = n
            step = 1
        else
            first_i = n
            last_i = 1
            step = -1
        end

        i_prev = first_i
        d_prev = y[i_prev] - substrate_y
        x_prev = x[i_prev]

        @inbounds for i in (first_i+step):step:last_i
            d_curr = y[i] - substrate_y
            if d_prev * d_curr <= 0  # zero or sign change
                t = d_prev / (d_prev - d_curr + eps(Float32))
                xc = x_prev + t * (x[i] - x_prev)
                return (xc, substrate_y)
            end
            d_prev = d_curr
            x_prev = x[i]
        end
        return nothing
    end

    left = locate(true)
    right = locate(false)
    return left, right
end


function smooth_and_analyze(contour, substrate_y; window=21, poly=3, filter_monotonic=false, filter_substrate=false)
    pts = copy(contour)
    if filter_substrate
        pts = pts[pts[:, 2].<=substrate_y, :]
    end
    if filter_monotonic
        order = sortperm(pts[:, 1])
        pts = pts[order, :]
        xs = unique(pts[:, 1])
        keep = Vector{Float64}()
        ys = Vector{Float64}()
        for xval in xs
            ys_at_x = pts[pts[:, 1].==xval, 2]
            push!(keep, xval)
            push!(ys, minimum(ys_at_x))
        end
        pts = hcat(keep, ys)
    end
    length(pts) < 3 && return nothing
    order = sortperm(pts[:, 1])
    x = pts[order, 1]
    y = pts[order, 2]
    w = isodd(window) ? window : window + 1
    w > length(x) && (w = isodd(length(x)) ? length(x) : length(x) - 1)
    w < poly + 2 && return nothing
    y_smooth = savgol_smooth(y, w, poly)
    y_deriv = savgol_derivative(y, w, poly)
    apex_idx = argmin(y_smooth)
    apex = (x[apex_idx], y_smooth[apex_idx])
    lc, rc = find_contact_points(x, y_smooth, substrate_y)
    # fallbacks
    if lc === nothing
        mask = x .<= x[apex_idx]
        idx = argmin(abs.(y_smooth[mask] .- substrate_y))
        lc = (x[mask][idx], y_smooth[mask][idx])
    end
    if rc === nothing
        mask = x .>= x[apex_idx]
        idx = argmin(abs.(y_smooth[mask] .- substrate_y))
        rc = (x[mask][idx], y_smooth[mask][idx])
    end
    l_slope = y_deriv[argmin(abs.(x .- lc[1]))]
    r_slope = y_deriv[argmin(abs.(x .- rc[1]))]
    angle_l = rad2deg(atan(abs(l_slope)))
    angle_r = rad2deg(atan(abs(r_slope)))
    return (; x_smooth=x, y_smooth, apex, left_contact=lc, right_contact=rc,
        left_slope=l_slope, right_slope=r_slope,
        left_angle=angle_l, right_angle=angle_r)
end

# ---------- UI ----------
function build_view(image, contour; geom, svg, opts)
    GLMakie.activate!()
    fig = Figure(size=(1000, 820), fontsize=12)
    ax = GLMakie.Axis(fig[1, 1]; aspect=DataAspect(), yreversed=true, title="Sessile Test Viewer (Julia) - Improved")
    img_disp = permutedims(image, (2, 1))  # show with x=cols, y=rows
    image!(ax, img_disp; interpolate=true)

    # observables for toggles
    show_contour = GLMakie.Observable(true)
    show_savgol = GLMakie.Observable(true)
    show_contacts = GLMakie.Observable(true)
    show_tangent = GLMakie.Observable(true)
    show_text = GLMakie.Observable(true)

    if !isempty(contour)
        lines!(ax, contour[:, 1], contour[:, 2]; color=:green, linewidth=2, visible=show_contour)
    end

    if geom !== nothing && geom[:baseline_y] !== nothing
        yb = geom[:baseline_y]
        lines!(ax, [0, size(img_disp, 1)], [yb, yb]; color=:dodgerblue, linestyle=:dash, linewidth=2)
        text!(ax, 10, yb - 5; text="Baseline Y=$(round(yb,digits=1))", align=(:left, :bottom), color=:cyan)
    end

    if geom !== nothing && geom[:apex] !== nothing
        axp, ayp = geom[:apex]
        scatter!(ax, [axp], [ayp]; markersize=10, color=:red)
        text!(ax, axp + 5, ayp - 5; text="Apex ($(round(axp,digits=1)), $(round(ayp,digits=1)))", color=:red, align=(:left, :bottom))
    end

    if svg !== nothing
        lines!(ax, svg.x_smooth, svg.y_smooth; color=:royalblue, linewidth=2, visible=show_savgol)
        if svg.left_contact !== nothing
            lc = svg.left_contact
            scatter!(ax, [lc[1]], [lc[2]]; color=:magenta, markersize=8, visible=show_contacts)
            if svg.left_slope !== nothing
                m = svg.left_slope
                len = 60
                dx = len / sqrt(1 + m^2)
                dy = m * dx
                lines!(ax, [lc[1], lc[1] + dx], [lc[2], lc[2] + dy];
                    color=:cyan, linestyle=:dash, visible=show_tangent)
                text!(ax, lc[1] + dx - 40, lc[2] + dy - 10; text="$(round(svg.left_angle,digits=1))°",
                    color=:black, visible=show_tangent)
            end
        end
        if svg.right_contact !== nothing
            rc = svg.right_contact
            scatter!(ax, [rc[1]], [rc[2]]; color=:magenta, markersize=8, visible=show_contacts)
            if svg.right_slope !== nothing
                m = svg.right_slope
                len = 60
                dx = len / sqrt(1 + m^2)
                dy = m * dx
                lines!(ax, [rc[1], rc[1] - dx], [rc[2], rc[2] - dy];
                    color=:cyan, linestyle=:dash, visible=show_tangent)
                text!(ax, rc[1] - dx + 10, rc[2] - dy - 10; text="$(round(svg.right_angle,digits=1))°",
                    color=:black, visible=show_tangent)
            end
        end
    end

    # metrics box
    function metrics_text()
        lines = String[]
        push!(lines, "Contact Angle L/R: " *
                     (svg === nothing ? "N/A" :
                      "$(round(svg.left_angle,digits=1))° / $(round(svg.right_angle,digits=1))°"))
        push!(lines, "Detector: $(opts.detector)")
        push!(lines, "Denoise: $(opts.denoise ? "on" : "off")")
        push!(lines, "Savgol: $(opts.smooth_savgol ? "on" : "off")")
        push!(lines, "Filter monotonic: $(opts.filter_monotonic ? "on" : "off")")
        push!(lines, "Filter substrate: $(opts.filter_substrate ? "on" : "off")")
        push!(lines, "Savgol window/poly: $(opts.savgol_window)/$(opts.savgol_poly)")
        push!(lines, "Substrate buffer: $(opts.substrate_buffer)px")
        push!(lines, "Margin fraction: $(opts.margin_fraction)")
        push!(lines, "ROI cols: $(opts.roi_cols[1]):$(opts.roi_cols[2])")
        return join(lines, "\n")
    end
    text_pos = GLMakie.Observable(Point2f(10, 10))
    on(ax.scene.viewport) do vp
        h = vp.widths[2]
        text_pos[] = Point2f(10, h - 10)
    end
    metrics_rect = GLMakie.Observable(Rect2f(0, 0, 0, 0))
    metrics_bg = poly!(ax, metrics_rect; color=RGBAf(0, 0, 0, 0.6),
        strokewidth=0, space=:pixel, visible=show_text)
    metrics_text_plot = text!(ax, text_pos; text=metrics_text(),
        align=(:left, :top), color=:white, space=:pixel, visible=show_text)
    function update_metrics_box!()
        bb = boundingbox(metrics_text_plot, :pixel)
        pad_x, pad_y = 8.0, 6.0
        metrics_rect[] = Rect2f(bb.origin[1] - pad_x, bb.origin[2] - pad_y,
            bb.widths[1] + 2 * pad_x, bb.widths[2] + 2 * pad_y)
    end
    on(text_pos) do _
        update_metrics_box!()
    end
    update_metrics_box!()

    # controls
    g = GridLayout(fig[2, 1]; tellwidth=false)
    cb1 = Checkbox(g[1, 1])
    Label(g[1, 2], "Contour")
    on(cb1.checked) do v
        show_contour[] = v
    end
    cb1.checked[] = true
    cb2 = Checkbox(g[1, 3])
    Label(g[1, 4], "Savgol")
    on(cb2.checked) do v
        show_savgol[] = v
    end
    cb2.checked[] = true
    cb3 = Checkbox(g[1, 5])
    Label(g[1, 6], "Contacts")
    on(cb3.checked) do v
        show_contacts[] = v
    end
    cb3.checked[] = true
    cb4 = Checkbox(g[1, 7])
    Label(g[1, 8], "Tangents")
    on(cb4.checked) do v
        show_tangent[] = v
    end
    cb4.checked[] = true
    cb5 = Checkbox(g[1, 9])
    Label(g[1, 10], "Metrics")
    on(cb5.checked) do v
        show_text[] = v
    end
    cb5.checked[] = true

    fig
end

##############################
# ADVANCED PROFILER SUPPORT
##############################

# Hold entries for summary
const PROFILE_LOG = Ref(Vector{Dict{Symbol,Any}}())

function _profile_push!(entry)
    push!(PROFILE_LOG[], entry)
end

"""
profile_section(name, fun; profile=true)

Runs `fun()` once, records precise metrics (time, gctime, memory, allocs).
Returns (result, seconds).
"""
function profile_section(name::String, fun::Function; profile::Bool)
    if !profile
        return fun(), 0.0
    end

    # Run once for the actual result, then benchmark separately for metrics.
    result = fun()
    bench = @benchmarkable $fun() samples = 1 evals = 1
    trial = run(bench)

    # Trial has vectors: times, gctimes; use the first (only) sample.
    seconds = trial.times[1] / 1e9
    gc_sec = trial.gctimes[1] / 1e9
    memory = trial.memory
    allocs = trial.allocs
    # Log entry
    _profile_push!(Dict(
        :stage => name,
        :time => seconds,
        :gc => gc_sec,
        :mem => memory,
        :alloc => allocs,
    ))

    return result, seconds
end

"""
print_profile_summary(; csv_path=nothing)

Prints a detailed aligned table with percentages.
If csv_path is provided, also writes a CSV file.
"""
function print_profile_summary(; csv_path=nothing)
    entries = PROFILE_LOG[]
    isempty(entries) && return

    total_time = sum(e[:time] for e in entries)
    maxname = maximum(length(e[:stage]) for e in entries)

    println("\n================= ADVANCED PROFILE SUMMARY =================")
    @printf("%-*s   %10s   %10s   %12s   %10s   %8s\n",
        maxname, "Stage", "Time [s]", "GC [s]", "Memory [bytes]", "Allocs", "% Total")
    println(repeat("-", maxname + 66))

    for e in entries
        pct = total_time == 0 ? 0.0 : (e[:time] / total_time * 100)
        @printf("%-*s   %10.6f   %10.6f   %12d   %10d   %7.2f%%\n",
            maxname,
            e[:stage],
            e[:time],
            e[:gc],
            e[:mem],
            e[:alloc],
            pct)
    end

    println("------------------------------------------------------------")
    @printf("%-*s   %10.6f   %10s   %12s   %10s   %7.2f%%\n",
        maxname, "TOTAL", total_time, "", "", "", 100.0)
    println("============================================================\n")

    if csv_path !== nothing
        open(csv_path, "w") do io
            println(io, "stage,time,gc,memory,allocs,percent")
            for e in entries
                pct = total_time == 0 ? 0.0 : (e[:time] / total_time * 100)
                println(io, "$(e[:stage]),$(e[:time]),$(e[:gc]),$(e[:mem]),$(e[:alloc]),$(pct)")
            end
        end
        println("Saved profiling CSV → $csv_path")
    end
end

# ---------- Main ----------
function main()
    ############################################################
    # Parse arguments
    ############################################################
    s = ArgParseSettings()
    @add_arg_table s begin
        "image_path"
        help = "Path to input image"
        required = true

        "--substrate-y"
        help = "Manual substrate Y"
        arg_type = Int

        "--substrate-buffer"
        help = "Pixels above substrate to include in mask"
        arg_type = Int
        default = 10

        "--margin-fraction"
        help = "Fraction of width for margin analysis"
        arg_type = Float64
        default = 0.05

        "--block-size"
        help = "Adaptive threshold block size (odd)"
        arg_type = Int
        default = 21

        "--detector"
        help = "Detector: auto|canny|otsu|adaptive|threshold"
        arg_type = String
        default = "auto"

        "--denoise"
        help = "Apply Gaussian denoising before detection"
        action = :store_true

        "--smooth-savgol"
        help = "Enable Savgol smoothing"
        action = :store_true

        "--filter-monotonic"
        help = "Keep min Y per X"
        action = :store_true

        "--filter-substrate"
        help = "Drop points below substrate"
        action = :store_true

        "--savgol-window"
        help = "Savgol window length"
        arg_type = Int
        default = 21

        "--savgol-poly"
        help = "Savgol polynomial order"
        arg_type = Int
        default = 3

        "--morph-iterations"
        help = "Number of morphological closing iterations"
        arg_type = Int
        default = 2

        "--profile"
        help = "Print profiling info for each stage"
        action = :store_true

        "--profile-csv"
        help = "Save profiling data to CSV"
        arg_type = String
    end

    args = parse_args(s)
    prof = args["profile"]
    PROFILE_LOG[] = Dict{Symbol,Any}[]   # reset advanced profiler log

    ############################################################
    # Load image
    ############################################################
    img = load(args["image_path"])
    gray = to_gray(img)
    rows, cols = size(gray)
    margin_frac = args["margin-fraction"]
    if margin_frac < 0 || margin_frac >= 0.5
        error("margin-fraction must be in [0, 0.5). Got $(margin_frac)")
    end
    margin_cols = round(Int, cols * margin_frac)
    left_col = clamp(1 + margin_cols, 1, cols)
    right_col = clamp(cols - margin_cols, 1, cols)
    ############################################################
    # STEP 1 — Baseline detection
    ############################################################
    (sub_y, t_baseline) = profile_section(
        "Baseline estimation",
        () -> begin
            if args["substrate-y"] === nothing
                estimate_substrate(gray)
            else
                args["substrate-y"]
            end
        end;
        profile=prof
    )
    println("Estimated baseline at Y = $sub_y")

    ############################################################
    # STEP 2 — Mask detection
    ############################################################
    ############################################################
    # Pre-declare variables to avoid UndefVarError
    ############################################################
    mask = nothing
    inv_mask = nothing
    all_contours = nothing
    all_contours_inv = nothing
    contour = Array{Float64}(undef, 0, 2)
    (mask, t_mask) = profile_section(
        "Mask detection",
        () -> detect_mask(
            gray;
            detector=args["detector"],
            block_size=args["block-size"],
            denoise=args["denoise"]
        );
        profile=prof
    )
    println("Original mask: $(sum(mask)) pixels")

    ############################################################
    # STEP 3 — Morphology + cleanup
    ############################################################
    (mask, t_morph) = profile_section(
        "Morphology",
        () -> begin
            buffer = args["substrate-buffer"]
            cut_y = clamp(round(Int, sub_y) + buffer, 1, rows)

            mask[cut_y:end, :] .= 0
            println("After substrate cut (Y>=$(cut_y)): $(sum(mask))")

            if margin_cols > 0
                apply_roi!(mask, left_col, right_col)
                println("After ROI (cols $(left_col):$(right_col))): $(sum(mask))")
            end

            for i in 1:args["morph-iterations"]
                mask = closing(mask)
            end

            mask = opening(mask)
            if margin_cols > 0
                apply_roi!(mask, left_col, right_col)
            end
            println("After morphological ops: $(sum(mask))")

            mask
        end;
        profile=prof
    )

    ############################################################
    # STEP 4 — Component filtering
    ############################################################
    (mask, t_components) = profile_section(
        "Component filtering",
        () -> begin
            labels = label_components(mask)
            if maximum(labels) > 0
                filter_components_by_size!(mask, labels; min_fraction=0.01)
            end
            println("After size filtering: $(sum(mask))")
            mask
        end;
        profile=prof
    )

    ############################################################
    # STEP 5 — Contour extraction
    ############################################################
    (all_contours, t_contours) = profile_section(
        "Contour extraction",
        () -> get_all_contours(mask);
        profile=prof
    )

    contour = Array{Float64}(undef, 0, 2)

    # Try to find a droplet-like contour
    for candidate in all_contours
        if is_droplet_shaped(candidate, sub_y, cols)
            contour = candidate
            println("Selected droplet contour with $(size(contour,1)) points")
            break
        end
    end

    # Fallback: try inverted mask if no good contour found
    if isempty(contour)
        println("No droplet found in mask, trying inverted...")

        # Clean inverted mask
        (inv_mask, _) = profile_section(
            "Inverted mask cleanup",
            () -> begin
                inv = .!mask
                buffer = args["substrate-buffer"]
                cut_y = clamp(round(Int, sub_y) + buffer, 1, rows)
                inv[cut_y:end, :] .= 0
                if margin_cols > 0
                    apply_roi!(inv, left_col, right_col)
                end

                # Same morphology cleanup
                for i in 1:args["morph-iterations"]
                    inv = closing(inv)
                end
                inv = opening(inv)
                if margin_cols > 0
                    apply_roi!(inv, left_col, right_col)
                end

                # Size filtering
                labels_inv = label_components(inv)
                if maximum(labels_inv) > 0
                    filter_components_by_size!(inv, labels_inv; min_fraction=0.01)
                end

                inv
            end;
            profile=prof
        )

        # Extract contours from inverted mask
        (all_contours_inv, _) = profile_section(
            "Inverted contour extraction",
            () -> get_all_contours(inv_mask);
            profile=prof
        )

        # Select droplet-like contour
        for candidate in all_contours_inv
            if is_droplet_shaped(candidate, sub_y, cols)
                contour = candidate
                println("Selected droplet contour from inverted mask with $(size(contour, 1)) points")
                break
            end
        end
    end


    isempty(contour) && error("No valid droplet contour found.")

    ############################################################
    # STEP 6 — Geometry (apex)
    ############################################################
    geometry = Dict{Symbol,Any}(:baseline_y => sub_y, :apex => nothing)
    apex_idx = argmin(contour[:, 2])
    geometry[:apex] = (contour[apex_idx, 1], contour[apex_idx, 2])

    ############################################################
    # STEP 7 — Savitzky-Golay smoothing (optional)
    ############################################################
    (svg, t_svg) = profile_section(
        "Savitzky–Golay",
        () -> begin
            if args["smooth-savgol"]
                smooth_and_analyze(contour, sub_y;
                    window=args["savgol-window"],
                    poly=args["savgol-poly"],
                    filter_monotonic=args["filter-monotonic"],
                    filter_substrate=args["filter-substrate"])
            else
                nothing
            end
        end;
        profile=prof
    )

    ############################################################
    # STEP 8 — Show UI
    ############################################################
    fig = build_view(img, contour;
        geom=geometry,
        svg=svg,
        opts=(;
            detector=args["detector"],
            denoise=args["denoise"],
            smooth_savgol=args["smooth-savgol"],
            filter_monotonic=args["filter-monotonic"],
            filter_substrate=args["filter-substrate"],
            savgol_window=args["savgol-window"],
            savgol_poly=args["savgol-poly"],
            substrate_buffer=args["substrate-buffer"],
            margin_fraction=args["margin-fraction"],
            roi_cols=(left_col, right_col)
        )
    )

    display(fig)
    GLMakie.wait(fig.scene)

    ############################################################
    # STEP 9 — Final profiling summary
    ############################################################
    if prof
        csv_path = get(args, "profile-csv", nothing)
        print_profile_summary(; csv_path=csv_path)
    end
end


main()
