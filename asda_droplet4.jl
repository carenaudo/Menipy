using Images, ImageFiltering, ImageMorphology, ImageSegmentation
using Statistics, LinearAlgebra, Polynomials
using Plots
using CairoMakie
using ImageTransformations  # imresize
###############  CONTOUR DETECTION FROM JuliaImages  ################

# Neighborhood unit moves
const DIRS = [
    CartesianIndex(0, 1),   # E
    CartesianIndex(-1, 1),   # NE
    CartesianIndex(-1, 0),   # N
    CartesianIndex(-1, -1),   # NW
    CartesianIndex(0, -1),   # W
    CartesianIndex(1, -1),   # SW
    CartesianIndex(1, 0),   # S
    CartesianIndex(1, 1)    # SE
]

# Try stepping in direction k from pixel p
function step_dir(img, p, k)
    q = p + DIRS[k]
    if checkbounds(Bool, img, q) && img[q] != 0
        return q
    end
    return CartesianIndex(0, 0)
end

# Return direction index from point a to b
dir_index(a, b) = findfirst(d -> a + d == b, DIRS)

# Contour tracing
function trace_contour(img, start::CartesianIndex)
    p0 = start
    p1 = start + DIRS[1]  # east step
    while p1 == CartesianIndex(0, 0)
        p1 = p0 + DIRS[mod1(dir_index(p0, p1) + 1, 8)]
    end

    contour = CartesianIndex[]
    push!(contour, p0)

    prev = p0
    curr = p1

    while true
        push!(contour, curr)
        d0 = dir_index(prev, curr)
        k = mod1(d0 + 6, 8)  # turn right
        q = CartesianIndex(0, 0)

        while true
            q = step_dir(img, curr, k)
            if q != CartesianIndex(0, 0)
                break
            end
            k = mod1(k + 1, 8)
        end

        prev, curr = curr, q
        if curr == start && prev == p1
            break
        end
    end

    return contour
end

"""
    find_contours(img::BitMatrix)

Returns a vector of contours, each a Vector{CartesianIndex}.
"""
function find_contours(img::BitMatrix)
    visited = falses(size(img))
    contours = Vector{Vector{CartesianIndex}}()

    for I in CartesianIndices(img)
        if img[I] && !visited[I]
            c = trace_contour(img, I)
            for p in c
                visited[p] = true
            end
            push!(contours, c)
        end
    end

    return contours
end

"""
    DropletAnalysis

Estructura para almacenar los resultados del análisis de gota
"""
struct DropletAnalysis
    contact_angle_left::Float64
    contact_angle_right::Float64
    contact_angle_mean::Float64
    base_diameter::Float64
    droplet_height::Float64
    substrate_tilt::Float64
    baseline_y::Float64
    left_contact_point::Tuple{Int,Int}
    right_contact_point::Tuple{Int,Int}
    apex_point::Tuple{Int,Int}
    contour::Vector{Tuple{Int,Int}}
    substrate_line::Vector{Tuple{Int,Int}}
end

"""
    load_and_preprocess(image_path::String; blur_sigma=1.5, threshold=0.5, invert=false)

Carga y preprocesa la imagen de la gota
"""
function load_and_preprocess(image_path::String; blur_sigma=1.5, threshold=0.5, invert=false)
    # Cargar imagen
    img = load(image_path)

    # Convertir a escala de grises
    img_gray = Gray.(img)

    # Aplicar filtro Gaussiano para reducir ruido
    img_filtered = imfilter(img_gray, Kernel.gaussian(blur_sigma))

    # Binarizar (gota oscura sobre fondo claro)
    if invert
        img_binary = img_filtered .> threshold  # Gota clara sobre fondo oscuro
    else
        img_binary = img_filtered .< threshold  # Gota oscura sobre fondo claro (default)
    end

    return img, img_gray, img_filtered, img_binary
end

"""
    find_largest_component(img_binary::BitMatrix)

Encuentra el componente conectado más grande (la gota)
"""
function find_largest_component(img_binary::BitMatrix)
    labels = label_components(img_binary)
    h, w = size(img_binary)

    # Diccionario: label -> stats
    stats = Dict{Int,NamedTuple{(:area, :touch_top, :touch_bottom, :touch_left, :touch_right),
        Tuple{Int,Bool,Bool,Bool,Bool}}}()

    for y in 1:h, x in 1:w
        label = labels[y, x]
        if label == 0
            continue
        end

        s = get(stats, label,
            (area=0,
                touch_top=false,
                touch_bottom=false,
                touch_left=false,
                touch_right=false))

        stats[label] = (
            area=s.area + 1,
            touch_top=s.touch_top || y == 1,
            touch_bottom=s.touch_bottom || y == h,
            touch_left=s.touch_left || x == 1,
            touch_right=s.touch_right || x == w,
        )
    end

    if isempty(stats)
        return falses(size(img_binary))
    end

    # 1) No tocar ningún borde
    candidates = [(label, s) for (label, s) in stats
                  if !s.touch_top && !s.touch_bottom && !s.touch_left && !s.touch_right]

    # 2) Si no hay, permitir laterales pero no top ni bottom
    if isempty(candidates)
        candidates = [(label, s) for (label, s) in stats
                      if !s.touch_top && !s.touch_bottom]
    end

    # 3) Si no hay, solo exigir que no toque top
    if isempty(candidates)
        candidates = [(label, s) for (label, s) in stats
                      if !s.touch_top]
    end

    if isempty(candidates)
        error("No se encontró ningún componente tipo 'gota'. Revisa threshold/invert/ROI.")
    end

    # 🔧 Aquí el cambio importante:
    # sacar primero las áreas, luego argmax sobre esas áreas
    areas = [c[2].area for c in candidates]
    largest_idx = argmax(areas)          # esto es un Int
    largest_label = candidates[largest_idx][1]

    return labels .== largest_label
end


"""
    apply_auto_roi(img_binary; frac_bottom=0.5)

Devuelve una versión de img_binary donde solo se conserva
la fracción inferior de la imagen (ROI), el resto se pone a false.
"""
function apply_auto_roi(img_binary::BitMatrix; frac_bottom=0.5)
    h, w = size(img_binary)
    y0 = max(1, Int(round(h * (1 - frac_bottom))))  # por defecto, mitad inferior

    roi_mask = falses(h, w)
    roi_mask[y0:h, :] .= img_binary[y0:h, :]

    return roi_mask
end

"""
    extract_contour_ordered(img_binary::BitMatrix)

Extrae el contorno ordenado de la gota (solo el perímetro exterior)
"""
function extract_contour_ordered(img_binary::BitMatrix)
    # Encontrar el componente más grande
    droplet_mask = find_largest_component(img_binary)

    # Limpiar con operaciones morfológicas
    droplet_clean = closing(opening(droplet_mask, trues(3, 3)), trues(3, 3))

    # Detectar bordes (contorno)
    edges = droplet_clean .⊻ erode(droplet_clean)

    # Extraer coordenadas del contorno
    contour_points = Tuple{Int,Int}[]
    h, w = size(edges)

    for i in 1:h
        for j in 1:w
            if edges[i, j]
                push!(contour_points, (j, i))  # (x, y)
            end
        end
    end

    if isempty(contour_points)
        return Tuple{Int,Int}[], droplet_clean
    end

    # Ordenar contorno de manera continua (recorrido)
    ordered_contour = order_contour_points(contour_points)

    return ordered_contour, droplet_clean
end

"""
    debug_plots(img, img_gray, img_binary, img_binary_roi, droplet_mask, contour; save_path=nothing)

Muestra plots de depuración:
  1) Imagen original
  2) Imagen binarizada completa
  3) ROI / máscara de gota
  4) Contorno superpuesto sobre la imagen original
"""
function debug_plots(img, img_gray, img_binary, img_binary_roi, droplet_mask, contour; save_path=nothing)
    # 1. Original
    p1 = plot(img, title="Original", axis=false, ticks=false, aspect_ratio=:equal)

    # 2. Binarizada global
    p2 = plot(Gray.(img_binary), title="Binarizada (global)",
        axis=false, ticks=false, aspect_ratio=:equal)

    # 3. ROI / máscara
    #    mostrarmos primero el ROI aplicado y encima la máscara limpia
    p3 = plot(Gray.(img_binary_roi), title="ROI Inferior", axis=false, ticks=false, aspect_ratio=:equal)
    plot!(p3, Gray.(droplet_mask), alpha=0.6, label=false)

    # 4. Contorno sobre original
    p4 = plot(img, title="Contorno detectado", axis=false, ticks=false, aspect_ratio=:equal)
    if !isempty(contour)
        xs = [p[1] for p in contour]
        ys = [p[2] for p in contour]
        plot!(p4, xs, ys, color=:red, linewidth=2, label="Contorno")
    end

    p_debug = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))

    if !isnothing(save_path)
        savefig(p_debug, save_path)
        println("💾 Debug plots guardados en: $save_path")
    end

    display(p_debug)
    return p_debug
end


"""
    order_contour_points(points::Vector{Tuple{Int,Int}})

Ordena los puntos del contorno de manera continua
"""
function order_contour_points(points::Vector{Tuple{Int,Int}})
    if isempty(points)
        return points
    end

    # Separar en contorno izquierdo y derecho
    x_coords = [p[1] for p in points]
    y_coords = [p[2] for p in points]

    # Encontrar punto más alto (apex)
    apex_idx = argmin(y_coords)
    apex_x = x_coords[apex_idx]

    # Separar lado izquierdo y derecho
    left_points = filter(p -> p[1] <= apex_x, points)
    right_points = filter(p -> p[1] >= apex_x, points)

    # Ordenar lado izquierdo de arriba hacia abajo
    sort!(left_points, by=p -> p[2])

    # Ordenar lado derecho de arriba hacia abajo
    sort!(right_points, by=p -> p[2])

    # Combinar: izquierda (arriba->abajo) + derecha invertida (abajo->arriba)
    ordered = vcat(left_points, reverse(right_points))

    return ordered
end

"""
    find_substrate_line_robust(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})

Encuentra la línea de sustrato de manera robusta
"""
function find_substrate_line_robust(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})
    if isempty(contour)
        return Tuple{Int,Int}[], 0.0, (0, 0), (0, 0), 0.0, (0, 0)
    end

    x_coords = [p[1] for p in contour]
    y_coords = [p[2] for p in contour]

    max_y = maximum(y_coords)
    min_y = minimum(y_coords)
    height_range = max_y - min_y

    # 🔧 Encontrar la baseline: el punto más bajo del contorno
    # Esta es donde la gota toca el sustrato
    baseline_y = Float64(max_y)

    # 🔧 Buscar puntos de contacto en la región inferior
    # Tomar puntos en el bottom 5% de altura
    threshold_y = max_y - 0.05 * height_range
    base_candidates = filter(p -> p[2] >= threshold_y, contour)

    if length(base_candidates) < 2
        # Fallback: tomar los puntos con mayor Y
        sorted_by_y = sort(contour, by=p -> -p[2])
        base_candidates = sorted_by_y[1:min(30, length(sorted_by_y))]
    end

    if length(base_candidates) < 2
        apex_idx = argmin(y_coords)
        apex_point = contour[apex_idx]
        return Tuple{Int,Int}[], 0.0, (0, 0), (0, 0), baseline_y, apex_point
    end

    # 🔧 CLAVE: Los puntos de contacto son los extremos X de los candidatos base
    # que están MUY cerca de la baseline (dentro de 2 píxeles)
    base_candidates_at_line = filter(p -> abs(p[2] - baseline_y) <= 2, base_candidates)

    if length(base_candidates_at_line) < 2
        base_candidates_at_line = base_candidates
    end

    x_base_coords = [p[1] for p in base_candidates_at_line]

    # Contacto izquierdo: mínimo X
    left_idx = argmin(x_base_coords)
    left_contact = base_candidates_at_line[left_idx]

    # Contacto derecho: máximo X
    right_idx = argmax(x_base_coords)
    right_contact = base_candidates_at_line[right_idx]

    # Recalcular baseline como promedio de los Y de contacto
    baseline_y = (left_contact[2] + right_contact[2]) / 2.0

    # 🔧 Calcular inclinación del sustrato usando los puntos de contacto
    if abs(right_contact[1] - left_contact[1]) > 10
        slope = (right_contact[2] - left_contact[2]) / (right_contact[1] - left_contact[1])
        intercept = left_contact[2] - slope * left_contact[1]
    else
        slope = 0.0
        intercept = baseline_y
    end

    substrate_tilt = atand(slope)

    # Generar línea de sustrato
    x_min = minimum(x_coords) - 20
    x_max = maximum(x_coords) + 20
    substrate_line = [(Int(round(x)), Int(round(slope * x + intercept))) for x in range(x_min, x_max, length=150)]

    # Apex
    apex_idx = argmin(y_coords)
    apex_point = contour[apex_idx]

    println("   📏 Baseline Y: $(round(baseline_y, digits=1))")
    println("   📍 Contacto Izq: $(left_contact)")
    println("   📍 Contacto Der: $(right_contact)")
    println("   📏 Separación contactos: $(abs(right_contact[1] - left_contact[1])) px")

    return substrate_line, substrate_tilt, left_contact, right_contact, baseline_y, apex_point
end

"""
    calculate_contact_angle_polynomial(contour::Vector{Tuple{Int,Int}}, 
                                       contact_point::Tuple{Int,Int}, 
                                       baseline_y::Float64,
                                       is_left::Bool; 
                                       fit_range=30)

Calcula el ángulo de contacto usando ajuste polinomial cerca del punto de contacto
"""
function calculate_contact_angle_polynomial(contour::Vector{Tuple{Int,Int}},
    contact_point::Tuple{Int,Int},
    baseline_y::Float64,
    is_left::Bool;
    fit_range=30)
    cp_x, cp_y = contact_point

    # Filtrar puntos cerca del contacto (solo en la parte superior de la baseline)
    if is_left
        # Lado izquierdo: tomar puntos a la derecha y arriba del contacto
        nearby = filter(p -> p[1] >= cp_x && p[1] <= cp_x + fit_range && p[2] <= cp_y, contour)
    else
        # Lado derecho: tomar puntos a la izquierda y arriba del contacto
        nearby = filter(p -> p[1] <= cp_x && p[1] >= cp_x - fit_range && p[2] <= cp_y, contour)
    end

    if length(nearby) < 5
        # Fallback: tomar los N puntos más cercanos
        distances = [sqrt((p[1] - cp_x)^2 + (p[2] - cp_y)^2) for p in contour]
        sorted_idx = sortperm(distances)
        nearby = contour[sorted_idx[1:min(20, length(contour))]]
        nearby = filter(p -> p[2] <= cp_y, nearby)
    end

    if length(nearby) < 3
        return 90.0
    end

    # Ordenar por distancia al punto de contacto
    sort!(nearby, by=p -> sqrt((p[1] - cp_x)^2 + (p[2] - cp_y)^2))

    # Tomar los primeros N puntos
    n_fit = min(fit_range, length(nearby))
    fit_points = nearby[1:n_fit]

    # Extraer coordenadas
    x_fit = [Float64(p[1]) for p in fit_points]
    y_fit = [Float64(p[2]) for p in fit_points]

    # Ajustar polinomio de grado 2
    slope = NaN
    try
        if length(x_fit) >= 3
            p = fit(x_fit, y_fit, 2)
            # Derivada en el punto de contacto
            dp = derivative(p)
            slope = dp(Float64(cp_x))
        else
            # Regresión lineal
            A = hcat(x_fit, ones(length(x_fit)))
            coeff = A \ y_fit
            slope = coeff[1]
        end
    catch
        return 90.0
    end

    # Si no se pudo calcular una pendiente valida, devolver valor neutro
    if isnan(slope)
        return 90.0
    end

    # Calcular ángulo de contacto
    # El ángulo es medido desde la horizontal (baseline) hacia arriba
    angle_rad = atan(abs(slope))
    angle_deg = rad2deg(angle_rad)

    # Ajustar según el lado
    if is_left
        # Lado izquierdo: si la pendiente es negativa, el ángulo es obtuso
        if slope < 0
            contact_angle = 180.0 - angle_deg
        else
            contact_angle = angle_deg
        end
    else
        # Lado derecho: si la pendiente es positiva, el ángulo es obtuso
        if slope > 0
            contact_angle = 180.0 - angle_deg
        else
            contact_angle = angle_deg
        end
    end

    return clamp(contact_angle, 0.0, 180.0)
end

"""
    calculate_droplet_dimensions(contour::Vector{Tuple{Int,Int}}, 
                                 left_contact::Tuple{Int,Int}, 
                                 right_contact::Tuple{Int,Int},
                                 apex_point::Tuple{Int,Int})

Calcula dimensiones de la gota
"""
function calculate_droplet_dimensions(contour::Vector{Tuple{Int,Int}},
    left_contact::Tuple{Int,Int},
    right_contact::Tuple{Int,Int},
    apex_point::Tuple{Int,Int})
    if isempty(contour)
        return 0.0, 0.0
    end

    # Diámetro base
    base_diameter = abs(right_contact[1] - left_contact[1])

    # Altura de la gota
    baseline_y = (left_contact[2] + right_contact[2]) / 2
    droplet_height = baseline_y - apex_point[2]

    return base_diameter, droplet_height
end

"""
    analyze_droplet(image_path::String; 
                   blur_sigma=1.5, 
                   threshold=0.5, 
                   fit_range=25,
                   pixel_size=1.0,
                   invert=false)

Función principal de análisis ASDA
"""
function analyze_droplet(image_path::String;
    blur_sigma=1.5,
    threshold=0.5,
    fit_range=25,
    pixel_size=1.0,
    invert=false,
    debug=false)   # 👈 NUEVO

    println("=== Análisis ASDA de Gota Sésil ===\n")
    println("1. Cargando y preprocesando imagen...")
    img, img_gray, img_filtered, img_binary = load_and_preprocess(image_path,
        blur_sigma=blur_sigma,
        threshold=threshold,
        invert=invert)
    # 🔧 NUEVO: limitar análisis a la parte inferior de la imagen
    img_binary_roi = apply_auto_roi(img_binary; frac_bottom=0.5)
    println("2. Detectando contorno de la gota...")
    contour, img_cleaned = extract_contour_ordered(img_binary_roi)
    # Plots de depuración (binary → ROI → máscara → contorno)
    if debug
        println("   🔍 Generando debug plots...")
        debug_plots(img, img_gray, img_binary, img_binary_roi, img_cleaned, contour)
    end
    if isempty(contour)
        error("❌ No se pudo detectar el contorno de la gota. Ajusta el threshold o invert.")
    end

    println("   ✓ Contorno detectado: $(length(contour)) puntos")

    println("3. Identificando línea de sustrato...")
    substrate_line, substrate_tilt, left_contact, right_contact, baseline_y, apex_point =
        find_substrate_line_robust(contour, size(img_binary))

    println("4. Calculando ángulos de contacto...")
    ca_left = calculate_contact_angle_polynomial(contour, left_contact, baseline_y, true,
        fit_range=fit_range)
    ca_right = calculate_contact_angle_polynomial(contour, right_contact, baseline_y, false,
        fit_range=fit_range)
    ca_mean = (ca_left + ca_right) / 2

    println("   ✓ Ángulo izquierdo: $(round(ca_left, digits=2))°")
    println("   ✓ Ángulo derecho: $(round(ca_right, digits=2))°")

    println("5. Calculando dimensiones...")
    base_diameter, droplet_height = calculate_droplet_dimensions(contour, left_contact,
        right_contact, apex_point)

    # Convertir a unidades físicas
    base_diameter_physical = base_diameter * pixel_size
    droplet_height_physical = droplet_height * pixel_size

    println("   ✓ Diámetro base: $(round(base_diameter_physical, digits=2))")
    println("   ✓ Altura: $(round(droplet_height_physical, digits=2))")

    # Crear estructura de resultados
    results = DropletAnalysis(
        ca_left, ca_right, ca_mean,
        base_diameter_physical, droplet_height_physical,
        substrate_tilt, baseline_y,
        left_contact, right_contact, apex_point,
        contour, substrate_line
    )

    println("\n✓ Análisis completado\n")

    # Visualizar
    plot_results(img, results, img_cleaned)

    return results, img, img_cleaned
end

"""
    plot_results(img, results::DropletAnalysis, img_cleaned; save_path=nothing)

Visualiza los resultados del análisis
"""
function plot_results(img, results::DropletAnalysis, img_cleaned; save_path=nothing)
    # Crear figura con múltiples subplots
    p1 = plot(img, title="Imagen Original", axis=false, ticks=false, aspect_ratio=:equal)

    p2 = plot(Gray.(img_cleaned), title="Imagen Binarizada",
        axis=false, ticks=false, aspect_ratio=:equal)

    # Plot del análisis con fondo de imagen original
    p3 = plot(img, title="Análisis ASDA", axis=false, ticks=false, aspect_ratio=:equal)

    # Dibujar contorno
    if !isempty(results.contour)
        x_contour = [p[1] for p in results.contour]
        y_contour = [p[2] for p in results.contour]
        plot!(p3, x_contour, y_contour, color=:red, linewidth=2, label="Contorno")
    end

    # Dibujar línea de sustrato
    if !isempty(results.substrate_line)
        x_substrate = [p[1] for p in results.substrate_line]
        y_substrate = [p[2] for p in results.substrate_line]
        plot!(p3, x_substrate, y_substrate, color=:blue, linewidth=3,
            linestyle=:dash, label="Sustrato")
    end

    # Marcar puntos de contacto
    scatter!(p3, [results.left_contact_point[1]], [results.left_contact_point[2]],
        color=:lime, markersize=10, markerstrokewidth=2,
        markerstrokecolor=:black, label="Contacto Izq")
    scatter!(p3, [results.right_contact_point[1]], [results.right_contact_point[2]],
        color=:yellow, markersize=10, markerstrokewidth=2,
        markerstrokecolor=:black, label="Contacto Der")

    # Marcar apex
    scatter!(p3, [results.apex_point[1]], [results.apex_point[2]],
        color=:cyan, markersize=10, markerstrokewidth=2,
        markerstrokecolor=:black, label="Apex")

    # Panel de resultados
    p4 = plot(framestyle=:none, xlims=(0, 1), ylims=(0, 1), legend=false)
    annotate!(p4, 0.1, 0.95, text("RESULTADOS DEL ANÁLISIS", 12, :left, :bold))
    annotate!(p4, 0.1, 0.87, text("Ángulo de contacto izq: $(round(results.contact_angle_left, digits=2))°", 10, :left))
    annotate!(p4, 0.1, 0.79, text("Ángulo de contacto der: $(round(results.contact_angle_right, digits=2))°", 10, :left))
    annotate!(p4, 0.1, 0.71, text("Ángulo de contacto promedio: $(round(results.contact_angle_mean, digits=2))°", 11, :left, :bold))
    annotate!(p4, 0.1, 0.63, text("Diámetro base: $(round(results.base_diameter, digits=2)) px", 10, :left))
    annotate!(p4, 0.1, 0.55, text("Altura de gota: $(round(results.droplet_height, digits=2)) px", 10, :left))
    annotate!(p4, 0.1, 0.47, text("Inclinación sustrato: $(round(results.substrate_tilt, digits=2))°", 10, :left))
    annotate!(p4, 0.1, 0.39, text("Relación aspecto (H/D): $(round(results.droplet_height/results.base_diameter, digits=3))", 10, :left))

    # Info adicional
    asimetria = abs(results.contact_angle_left - results.contact_angle_right)
    annotate!(p4, 0.1, 0.31, text("Asimetría ángulos: $(round(asimetria, digits=2))°", 9, :left))
    annotate!(p4, 0.1, 0.23, text("Apex: ($(results.apex_point[1]), $(results.apex_point[2]))", 9, :left))

    # Combinar plots
    p_final = plot(p1, p2, p3, p4, layout=(2, 2), size=(1400, 900))

    # Guardar si se especifica
    if !isnothing(save_path)
        savefig(p_final, save_path)
        println("💾 Figura guardada en: $save_path")
    end

    display(p_final)
    return p_final
end

"""
    print_results(results::DropletAnalysis)

Imprime resultados en consola
"""
function print_results(results::DropletAnalysis)
    println("\n" * "="^70)
    println("          RESULTADOS DEL ANÁLISIS DE GOTA SÉSIL (ASDA)")
    println("="^70)
    println("\n📐 ÁNGULOS DE CONTACTO:")
    println("   • Lado Izquierdo:    $(round(results.contact_angle_left, digits=2))°")
    println("   • Lado Derecho:      $(round(results.contact_angle_right, digits=2))°")
    println("   • Promedio:          $(round(results.contact_angle_mean, digits=2))°")
    println("   • Asimetría:         $(round(abs(results.contact_angle_left - results.contact_angle_right), digits=2))°")

    println("\n📏 DIMENSIONES:")
    println("   • Diámetro Base:     $(round(results.base_diameter, digits=2)) unidades")
    println("   • Altura:            $(round(results.droplet_height, digits=2)) unidades")
    println("   • Relación H/D:      $(round(results.droplet_height/results.base_diameter, digits=3))")

    println("\n🔧 SUSTRATO:")
    println("   • Inclinación:       $(round(results.substrate_tilt, digits=2))°")

    println("\n📍 PUNTOS CLAVE:")
    println("   • Contacto Izq:      $(results.left_contact_point)")
    println("   • Contacto Der:      $(results.right_contact_point)")
    println("   • Apex:              $(results.apex_point)")

    println("\n" * "="^70 * "\n")
end

# =============================================================================
# EJEMPLO DE USO
# =============================================================================

println("""
╔═══════════════════════════════════════════════════════════════════╗
║          Script ASDA - Análisis de Gota Sésil v2.1              ║
╚═══════════════════════════════════════════════════════════════════╝

USO BÁSICO:
  gr()  # Configurar backend
  
  # Para gota OSCURA sobre fondo CLARO (más común):
  results, img, cleaned = analyze_droplet("imagen.png",
                                         threshold=0.5,
                                         invert=false)
  
  # Para gota CLARA sobre fondo OSCURO:
  results, img, cleaned = analyze_droplet("imagen.png",
                                         threshold=0.5,
                                         invert=true)

PARÁMETROS PRINCIPALES:
  • threshold:   0.0-1.0 (default=0.5)
                 - Bajo (0.3-0.4): para gotas muy oscuras
                 - Alto (0.6-0.7): para gotas más claras
                 
  • invert:      true/false (default=false)
                 - false: gota oscura, fondo claro ← MÁS COMÚN
                 - true: gota clara, fondo oscuro
                 
  • blur_sigma:  1.0-3.0 (default=1.5) - Suavizado
  • fit_range:   15-40 (default=25) - Puntos para tangente
  • pixel_size:  Factor de calibración (μm/px o mm/px)

EJEMPLO COMPLETO:
  using Plots
  gr()
  
  results, img, cleaned = analyze_droplet("droplet.png",
                                         blur_sigma=1.5,
                                         threshold=0.5,
                                         fit_range=25,
                                         pixel_size=10.5,  # μm/px
                                         invert=false)
  
  print_results(results)
  plot_results(img, results, cleaned, save_path="analisis.png")

🔍 DIAGNÓSTICO:
  - Si la imagen binarizada está INVERTIDA → cambia invert=true
  - Si no detecta la gota → ajusta threshold (prueba 0.3, 0.5, 0.7)
  - Si puntos de contacto están mal → verifica la imagen binarizada
  - Si ángulos están raros → ajusta fit_range

""")


# Descomentar y modificar con tu ruta de imagen:
filename = "prueba sesil 2.png"
image_path = joinpath(@__DIR__, "data", "samples", filename)
results, img, img_cleaned = analyze_droplet(image_path,
    blur_sigma=2,
    threshold=0.3,
    fit_range=25,
    invert=false,       # Gota oscura, fondo claro
    pixel_size=1.0,
    debug=true)  # calibración en μm/pixel


print_results(results)
plot_results(img, results, img_cleaned)
println("Script ASDA cargado. Usa analyze_droplet(\"ruta/imagen.png\") para analizar una gota.")