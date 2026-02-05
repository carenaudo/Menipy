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

# ---------- Helpers ----------
# Gray.(img) already yields a 2D array for grayscale inputs; avoid channelview indexing quirks.
to_gray(img) = Float32.(Gray.(img))

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
        for j in -m:m
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
    # crude baseline: row of largest downward gradient in column-mean profile
    prof = vec(mean(gray, dims=2))
    dy = diff(prof)
    idx = findmax(abs.(dy))[2]
    return idx
end

function detect_mask(gray; detector="auto", block_size=21)
    if detector == "canny"
        edges = canny(gray, (0.20, 0.10), 1.0)
        return edges
    elseif detector == "otsu"
        t = Images.otsu_threshold(gray)
        return gray .< t
    elseif detector == "adaptive"
        k = ones(block_size, block_size) / (block_size^2)
        mean_img = imfilter(gray, k, "replicate")
        return gray .< mean_img .- 0.02
    elseif detector == "threshold"
        return gray .< 0.5
    else # auto: try otsu then fall back to canny
        m = detect_mask(gray; detector="otsu")
        return any(m) ? m : detect_mask(gray; detector="canny")
    end
end

function largest_contour(mask)
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



function find_contact_points(x, y, substrate_y)
    d = y .- substrate_y
    function locate(xa, da)
        s = sign.(da)
        idx = findfirst(i -> s[i] * s[i+1] < 0, 1:length(da)-1)
        isnothing(idx) && return nothing
        i = idx
        t = -da[i] / (da[i+1] - da[i] + eps(Float32))
        xc = xa[i] + t * (xa[i+1] - xa[i])
        return (xc, substrate_y)
    end
    left = locate(x, d)
    right = locate(reverse(x), reverse(d))
    right = right === nothing ? nothing : (last(x) - (right[1] - first(x)), right[2])
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
    ax = GLMakie.Axis(fig[1, 1]; aspect=DataAspect(), yreversed=true, title="Sessile Test Viewer (Julia)")
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
        push!(lines, "Savgol: $(opts.smooth_savgol ? "on" : "off")")
        push!(lines, "Filter monotonic: $(opts.filter_monotonic ? "on" : "off")")
        push!(lines, "Filter substrate: $(opts.filter_substrate ? "on" : "off")")
        push!(lines, "Savgol window/poly: $(opts.savgol_window)/$(opts.savgol_poly)")
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

# ---------- Main ----------
function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "image_path"
        help = "Path to input image"
        required = true
        "--substrate-y"
        help = "Manual substrate Y"
        arg_type = Int
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
    end
    args = parse_args(s)
    img = load(args["image_path"])
    gray = to_gray(img)

    # --- STEP 1: Determine Baseline FIRST ---
    sub_y = args["substrate-y"]
    if sub_y === nothing
        sub_y = estimate_substrate(gray)
        println("Estimated baseline at Y = $sub_y")
    end

    # --- STEP 2: Detect Mask ---
    mask = detect_mask(gray; detector=args["detector"], block_size=args["block-size"])

    # --- STEP 3: Clean the Mask (NEW) ---
    # 1. Reflection Removal: Force everything below the baseline + buffer to black
    #    We use a small buffer (+2 pixels) to ensure we don't cut the exact contact point.
    rows, cols = size(mask)
    cut_y = clamp(round(Int, sub_y) + 2, 1, rows)
    mask[cut_y:end, :] .= 0

    # 2. Morphological Cleanup: Remove small noise specks (e.g., dust)
    mask = closing(mask)

    # --- STEP 4: Find Contour ---
    contour = largest_contour(mask)

    if isempty(contour)
        # Try inverted mask (droplet may be brighter than background)
        inv_mask = .!mask
        # Re-apply cleaning to inverted mask
        inv_mask[cut_y:end, :] .= 0
        contour = largest_contour(inv_mask)
    end

    # Fallback loop (simplified for brevity, but apply the same masking logic if used)
    if isempty(contour)
        println("Warning: No contour found even after cleaning.")
    end

    isempty(contour) && error("No contour found")

    geometry = Dict{Symbol,Any}(:baseline_y => sub_y, :apex => nothing)
    # apex from raw contour
    apex_idx = argmin(contour[:, 2])
    geometry[:apex] = (contour[apex_idx, 1], contour[apex_idx, 2])

    svg = nothing
    if args["smooth-savgol"]
        svg = smooth_and_analyze(contour, sub_y;
            window=args["savgol-window"], poly=args["savgol-poly"],
            filter_monotonic=args["filter-monotonic"],
            filter_substrate=args["filter-substrate"])
    end

    fig = build_view(img, contour; geom=geometry, svg=svg, opts=(;
        detector=args["detector"],
        smooth_savgol=args["smooth-savgol"],
        filter_monotonic=args["filter-monotonic"],
        filter_substrate=args["filter-substrate"],
        savgol_window=args["savgol-window"],
        savgol_poly=args["savgol-poly"]))
    display(fig)
    GLMakie.wait(fig.scene)
end

main()
