using Images, ImageFiltering, ImageMorphology, ImageSegmentation
using Statistics, LinearAlgebra, Polynomials
using Plots
using CairoMakie
using ImageTransformations  # imresize

# ---------------------------------------------------------------------------
# Memory-safe debug visualization helpers (Makie)
# ---------------------------------------------------------------------------

to_compact(img) = RGB{N0f8}.(img)

function shrink(img; factor=0.25)
    factor = clamp(factor, 0.05, 1.0)
    h, w = size(img, 1), size(img, 2)
    newh = max(2, Int(round(h * factor)))
    neww = max(2, Int(round(w * factor)))
    return imresize(img, (newh, neww))
end

scale_contour(contour::Vector{Tuple{Int,Int}}, factor::Float64) =
    [(Float32(x * factor), Float32(y * factor)) for (x, y) in contour]

function bottom_rows(img_binary::BitMatrix; rows=700)
    h, w = size(img_binary)
    rows = min(rows, h)
    out = falses(h, w)
    out[(h-rows+1):h, :] .= img_binary[(h-rows+1):h, :]
    return out
end

function sanity_check_roi(img_binary_roi::BitMatrix; max_on=400_000)
    on = count(img_binary_roi)
    if on > max_on
        error("ROI too large (on=$on). threshold/invert likely wrong → abort to prevent RAM blow-up.")
    end
end

"""
    debug_plots_makie(img, img_binary, img_binary_roi, droplet_mask, contour; factor=0.25, save_path=nothing)

Makie-based debug panels:
  1) Original (scaled)
  2) Binary global (scaled)
  3) ROI + mask overlay (scaled)
  4) Contour overlay (scaled)

Uses downsampling + compact types to avoid huge RAM usage.
"""
function debug_plots_makie(img, img_binary, img_binary_roi, droplet_mask, contour;
    factor=0.25, save_path=nothing)

    img_s = shrink(to_compact(img); factor=factor)
    bin_s = shrink(Gray.(img_binary); factor=factor)
    roi_s = shrink(Gray.(img_binary_roi); factor=factor)
    mask_s = shrink(Gray.(droplet_mask); factor=factor)

    contour_s = isempty(contour) ? Tuple{Float32,Float32}[] : scale_contour(contour, factor)
    xs = isempty(contour_s) ? Float32[] : Float32[first(p) for p in contour_s]
    ys = isempty(contour_s) ? Float32[] : Float32[last(p) for p in contour_s]

    fig = CairoMakie.Figure(resolution=(1200, 850))

    ax1 = CairoMakie.Axis(fig[1, 1], title="Original (scaled)")
    CairoMakie.image!(ax1, img_s)
    CairoMakie.hidedecorations!(ax1)
    CairoMakie.hidespines!(ax1)

    ax2 = CairoMakie.Axis(fig[1, 2], title="Binary global (scaled)")
    CairoMakie.image!(ax2, bin_s)
    CairoMakie.hidedecorations!(ax2)
    CairoMakie.hidespines!(ax2)

    ax3 = CairoMakie.Axis(fig[2, 1], title="ROI + mask overlay (scaled)")
    CairoMakie.image!(ax3, roi_s)
    CairoMakie.image!(ax3, mask_s, alpha=0.45)
    CairoMakie.hidedecorations!(ax3)
    CairoMakie.hidespines!(ax3)

    ax4 = CairoMakie.Axis(fig[2, 2], title="Contour overlay (scaled)")
    CairoMakie.image!(ax4, img_s)
    if !isempty(xs)
        CairoMakie.lines!(ax4, xs, ys, linewidth=2)
    end
    CairoMakie.hidedecorations!(ax4)
    CairoMakie.hidespines!(ax4)

    if !isnothing(save_path)
        CairoMakie.save(save_path, fig)
        println("💾 Makie debug figure saved to: $save_path")
    end

    display(fig)
    return fig
end

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

Extrae un contorno ORDENADO y seguro en memoria de la gota.

Estrategia (robusta y sin OOM):
  1) Seleccionar el componente conectado "gota" (find_largest_component)
  2) Limpieza morfológica (opening/closing)
  3) Borde 1-pixel: boundary = mask .& .!erode(mask)
  4) Ordenar borde caminando por vecinos 8-conectados con tope de iteraciones
"""
function extract_contour_ordered(img_binary::BitMatrix)
    # 1) máscara de gota (componente principal)
    droplet_mask = find_largest_component(img_binary)

    # 2) limpieza
    droplet_clean = closing(opening(droplet_mask, trues(3, 3)), trues(3, 3))

    # Fail-fast: si la máscara es absurda, abortar (evita loops y RAM)
    on = count(droplet_clean)
    if on == 0
        return Tuple{Int,Int}[], droplet_clean
    end
    if on > 600_000
        error("Máscara de gota demasiado grande (on=$on). Revisa threshold/invert/ROI.")
    end

    # 3) borde (contorno) 1-pixel
    er = erode(droplet_clean)
    edges = droplet_clean .& .!er

    if count(edges) == 0
        return Tuple{Int,Int}[], droplet_clean
    end

    # 4) ordenar borde por caminata local (8-neighborhood)
    ordered = order_edge_walk(edges)

    return ordered, droplet_clean
end

"""
    order_edge_walk(edges::BitMatrix) -> Vector{Tuple{Int,Int}}

Ordena píxeles del borde siguiendo vecinos 8-conectados.
Usa un tope de pasos para evitar loops infinitos.
Devuelve puntos como (x,y) = (col,row).
"""
function order_edge_walk(edges::BitMatrix)
    h, w = size(edges)

    # vecindad 8 (row, col)
    nbrs = (
        CartesianIndex(-1, 0), CartesianIndex(-1, 1), CartesianIndex(0, 1), CartesianIndex(1, 1),
        CartesianIndex(1, 0), CartesianIndex(1, -1), CartesianIndex(0, -1), CartesianIndex(-1, -1)
    )

    # punto inicial: el más alto (min row), y dentro de esos el más a la izquierda (min col)
    start = nothing
    for r in 1:h
        for c in 1:w
            if edges[r, c]
                start = CartesianIndex(r, c)
                break
            end
        end
        start !== nothing && break
    end
    start === nothing && return Tuple{Int,Int}[]

    visited = falses(h, w)
    path = Tuple{Int,Int}[]
    current = start
    prev = CartesianIndex(0, 0)

    max_steps = count(edges) + 20
    steps = 0

    while steps < max_steps
        steps += 1
        visited[current] = true
        push!(path, (current[2], current[1]))  # (x,y)

        # buscar siguiente vecino de borde no visitado (preferimos continuidad)
        nextp = CartesianIndex(0, 0)

        # 1) prioridad: vecinos no visitados
        for d in nbrs
            q = current + d
            if checkbounds(Bool, edges, q) && edges[q] && !visited[q]
                nextp = q
                break
            end
        end

        # 2) si no hay no-visitados, permitimos volver al start para cerrar
        if nextp == CartesianIndex(0, 0)
            for d in nbrs
                q = current + d
                if checkbounds(Bool, edges, q) && edges[q] && q == start
                    nextp = q
                    break
                end
            end
        end

        # 3) stop si no hay salida o cerramos ciclo
        if nextp == CartesianIndex(0, 0) || (nextp == start && steps > 10)
            break
        end

        prev = current
        current = nextp
    end

    return path
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
# ============================================================================
# PASTE THIS FUNCTION TO REPLACE find_substrate_line_robust IN YOUR CODE
# ============================================================================

"""
    find_substrate_line_robust(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})

Versión corregida: Encuentra los puntos de contacto REALES entre gota y sustrato
"""
function find_substrate_line_robust(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})
    if isempty(contour) || length(contour) < 10
        return Tuple{Int,Int}[], 0.0, (0, 0), (0, 0), 0.0, (0, 0)
    end

    x_coords = [p[1] for p in contour]
    y_coords = [p[2] for p in contour]

    # 1. Encontrar apex (punto más alto = MENOR Y en coordenadas de imagen)
    apex_idx = argmin(y_coords)
    apex_point = contour[apex_idx]
    apex_x = apex_point[1]
    apex_y = apex_point[2]

    println("\n   🔍 DIAGNÓSTICO:")
    println("   📍 Apex: $apex_point (punto más ALTO de la gota)")

    # 2. Estadísticas del contorno
    y_min = minimum(y_coords)  # Parte superior (más alto en pantalla)
    y_max = maximum(y_coords)  # Parte inferior (más bajo en pantalla)
    x_min = minimum(x_coords)
    x_max = maximum(x_coords)

    println("   📊 Rango X: $x_min → $x_max")
    println("   📊 Rango Y: $y_min → $y_max")
    println("   📊 Total puntos contorno: $(length(contour))")

    # 3. ESTRATEGIA CORRECTA:
    #    Los puntos de contacto están en la PARTE MÁS BAJA (mayor Y)
    #    Buscamos en el 10% inferior del contorno

    y_bottom_threshold = y_max - 0.1 * (y_max - y_min)

    println("   🎯 Buscando contactos con Y ≥ $(round(y_bottom_threshold, digits=1))")

    # 4. Filtrar puntos de la parte baja
    bottom_region = [p for p in contour if p[2] >= y_bottom_threshold]

    if isempty(bottom_region)
        println("   ⚠️  Región baja vacía, usando umbral más permisivo...")
        y_bottom_threshold = y_max - 0.3 * (y_max - y_min)
        bottom_region = [p for p in contour if p[2] >= y_bottom_threshold]
    end

    println("   📊 Puntos en región baja: $(length(bottom_region))")

    if isempty(bottom_region)
        println("   ❌ ERROR: No hay puntos en la región baja del contorno!")
        return Tuple{Int,Int}[], 0.0, (0, 0), (0, 0), 0.0, apex_point
    end

    # 5. CLAVE: Dividir la región baja en lado izquierdo y derecho del apex
    left_region = [p for p in bottom_region if p[1] < apex_x]
    right_region = [p for p in bottom_region if p[1] > apex_x]

    println("   📊 Puntos región baja izquierda: $(length(left_region))")
    println("   📊 Puntos región baja derecha: $(length(right_region))")

    # 6. Encontrar el punto de contacto en cada lado
    #    Contacto = punto más BAJO (mayor Y) + más EXTREMO (menor/mayor X)

    if !isempty(left_region)
        # Lado izquierdo: buscar los puntos con Y máximo
        max_y_left = maximum([p[2] for p in left_region])

        # Entre los puntos más bajos, tomar el más a la IZQUIERDA
        bottom_left = [p for p in left_region if p[2] >= max_y_left - 3]
        left_contact = bottom_left[argmin([p[1] for p in bottom_left])]
    else
        # Fallback: tomar el punto más a la izquierda de toda la región baja
        left_contact = bottom_region[argmin([p[1] for p in bottom_region])]
        println("   ⚠️  Usando fallback para contacto izquierdo")
    end

    if !isempty(right_region)
        # Lado derecho: buscar los puntos con Y máximo
        max_y_right = maximum([p[2] for p in right_region])

        # Entre los puntos más bajos, tomar el más a la DERECHA
        bottom_right = [p for p in right_region if p[2] >= max_y_right - 3]
        right_contact = bottom_right[argmax([p[1] for p in bottom_right])]
    else
        # Fallback: tomar el punto más a la derecha de toda la región baja
        right_contact = bottom_region[argmax([p[1] for p in bottom_region])]
        println("   ⚠️  Usando fallback para contacto derecho")
    end

    println("\n   ✅ PUNTOS DE CONTACTO DETECTADOS:")
    println("   📍 Contacto izquierdo: $left_contact")
    println("   📍 Contacto derecho:   $right_contact")

    # 7. VALIDACIONES
    dx = right_contact[1] - left_contact[1]
    dy = right_contact[2] - left_contact[2]

    println("\n   🔍 VALIDACIONES:")
    println("   ↔️  Separación horizontal (dx): $dx px")
    println("   ↕️  Diferencia vertical (dy):   $dy px")

    # Validar que los contactos estén bien separados
    if dx < 30
        println("   ❌ ERROR: Contactos muy cercanos (dx=$dx < 30)")
        println("   🔧 Usando estrategia de extremos...")

        # Estrategia de emergencia: usar los extremos X y Y
        left_contact = contour[argmin(x_coords)]
        right_contact = contour[argmax(x_coords)]

        # Ajustar a la parte baja
        left_y_candidates = [p for p in contour if p[1] == left_contact[1]]
        if !isempty(left_y_candidates)
            left_contact = left_y_candidates[argmax([p[2] for p in left_y_candidates])]
        end

        right_y_candidates = [p for p in contour if p[1] == right_contact[1]]
        if !isempty(right_y_candidates)
            right_contact = right_y_candidates[argmax([p[2] for p in right_y_candidates])]
        end

        println("   🔧 Nuevos contactos: $left_contact, $right_contact")
        dx = right_contact[1] - left_contact[1]
        dy = right_contact[2] - left_contact[2]
    end

    # Validar que los contactos estén por debajo del apex
    if left_contact[2] < apex_y || right_contact[2] < apex_y
        println("   ⚠️  ADVERTENCIA: Algún contacto está por encima del apex!")
    end

    # 8. Calcular baseline y línea de sustrato
    baseline_y = (left_contact[2] + right_contact[2]) / 2.0

    if abs(dx) > 10
        slope = dy / dx
        intercept = left_contact[2] - slope * left_contact[1]
    else
        slope = 0.0
        intercept = baseline_y
    end

    substrate_tilt = atand(slope)

    # Generar línea de sustrato (extender más allá de los contactos)
    x_min_line = left_contact[1] - 50
    x_max_line = right_contact[1] + 50
    substrate_line = [(Int(round(x)), Int(round(slope * x + intercept)))
                      for x in range(x_min_line, x_max_line, length=150)]

    println("\n   📏 RESULTADOS FINALES:")
    println("   📏 Baseline Y: $(round(baseline_y, digits=1)) px")
    println("   📏 Diámetro base: $dx px")
    println("   📏 Inclinación sustrato: $(round(substrate_tilt, digits=2))°")
    println("   📏 Altura gota (apex a baseline): $(round(baseline_y - apex_y, digits=1)) px")

    return substrate_line, substrate_tilt, left_contact, right_contact, baseline_y, apex_point
end


# 🔧 FUNCIÓN ALTERNATIVA SI LA ANTERIOR NO FUNCIONA:
# Esta usa un enfoque diferente basado en la curvatura del contorno

"""
    find_substrate_line_curvature(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})

Método alternativo: detecta puntos de contacto buscando cambios en la dirección vertical
"""
function find_substrate_line_curvature(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})
    if isempty(contour) || length(contour) < 10
        return Tuple{Int,Int}[], 0.0, (0, 0), (0, 0), 0.0, (0, 0)
    end

    x_coords = [p[1] for p in contour]
    y_coords = [p[2] for p in contour]

    # Apex
    apex_idx = argmin(y_coords)
    apex_point = contour[apex_idx]
    apex_x = apex_point[1]

    # Detectar puntos donde Y deja de aumentar (puntos de contacto)
    # Estos son los puntos donde la gota "toca" el sustrato

    # Calcular derivada discreta de Y
    dy = diff(y_coords)

    # Encontrar dónde la derivada cambia de positiva a plana/negativa (lado izquierdo)
    # y de negativa a plana/positiva (lado derecho)

    # Para el lado izquierdo: buscar el último máximo local de Y antes del apex
    left_candidates = [(i, contour[i]) for i in 1:apex_idx if contour[i][1] < apex_x]
    if !isempty(left_candidates)
        # Punto más bajo del lado izquierdo
        left_contact = left_candidates[argmax([p[2][2] for p in left_candidates])]
    else
        left_contact = contour[1]
    end

    # Para el lado derecho: buscar el último máximo local de Y después del apex
    right_candidates = [(i, contour[i]) for i in apex_idx:length(contour) if contour[i][1] > apex_x]
    if !isempty(right_candidates)
        # Punto más bajo del lado derecho
        right_contact = right_candidates[argmax([p[2][2] for p in right_candidates])]
    else
        right_contact = contour[end]
    end

    baseline_y = (left_contact[2] + right_contact[2]) / 2.0

    # Calcular línea de sustrato
    dx = right_contact[1] - left_contact[1]
    if abs(dx) > 10
        slope = (right_contact[2] - left_contact[2]) / dx
        intercept = left_contact[2] - slope * left_contact[1]
    else
        slope = 0.0
        intercept = baseline_y
    end

    substrate_tilt = atand(slope)

    x_min = left_contact[1] - 30
    x_max = right_contact[1] + 30
    substrate_line = [(Int(round(x)), Int(round(slope * x + intercept)))
                      for x in range(x_min, x_max, length=100)]

    println("   [ALT] Contacto Izq: $(left_contact)")
    println("   [ALT] Contacto Der: $(right_contact)")
    println("   [ALT] Baseline Y: $(round(baseline_y, digits=1))")

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

    # Debug (Makie, memory-safe): binary → ROI → máscara → contorno
    if debug
        println("   🔍 Makie debug plots (compact + downscaled)...")
        img_binary_roi_debug = bottom_rows(img_binary_roi; rows=700)
        sanity_check_roi(img_binary_roi_debug; max_on=400_000)
        debug_plots_makie(img, img_binary, img_binary_roi_debug, img_cleaned, contour; factor=0.25)
        GC.gc()
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
    p1 = Plots.plot(img, title="Imagen Original", axis=false, ticks=false, aspect_ratio=:equal)

    p2 = Plots.plot(Gray.(img_cleaned), title="Imagen Binarizada",
        axis=false, ticks=false, aspect_ratio=:equal)

    # Plot del análisis con fondo de imagen original
    p3 = Plots.plot(img, title="Análisis ASDA", axis=false, ticks=false, aspect_ratio=:equal)

    # Dibujar contorno
    if !isempty(results.contour)
        x_contour = [p[1] for p in results.contour]
        y_contour = [p[2] for p in results.contour]
        Plots.plot!(p3, x_contour, y_contour, color=:red, linewidth=2, label="Contorno")
    end

    # Dibujar línea de sustrato
    if !isempty(results.substrate_line)
        x_substrate = [p[1] for p in results.substrate_line]
        y_substrate = [p[2] for p in results.substrate_line]
        Plots.plot!(p3, x_substrate, y_substrate, color=:blue, linewidth=3,
            linestyle=:dash, label="Sustrato")
    end

    # Marcar puntos de contacto
    Plots.scatter!(p3, [results.left_contact_point[1]], [results.left_contact_point[2]],
        color=:lime, markersize=10, markerstrokewidth=2,
        markerstrokecolor=:black, label="Contacto Izq")
    Plots.scatter!(p3, [results.right_contact_point[1]], [results.right_contact_point[2]],
        color=:yellow, markersize=10, markerstrokewidth=2,
        markerstrokecolor=:black, label="Contacto Der")

    # Marcar apex
    Plots.scatter!(p3, [results.apex_point[1]], [results.apex_point[2]],
        color=:cyan, markersize=10, markerstrokewidth=2,
        markerstrokecolor=:black, label="Apex")

    # Panel de resultados
    p4 = Plots.plot(framestyle=:none, xlims=(0, 1), ylims=(0, 1), legend=false)
    Plots.annotate!(p4, 0.1, 0.95, Plots.text("RESULTADOS DEL ANÁLISIS", 12, :left, :bold))
    Plots.annotate!(p4, 0.1, 0.87, Plots.text("Ángulo de contacto izq: $(round(results.contact_angle_left, digits=2))°", 10, :left))
    Plots.annotate!(p4, 0.1, 0.79, Plots.text("Ángulo de contacto der: $(round(results.contact_angle_right, digits=2))°", 10, :left))
    Plots.annotate!(p4, 0.1, 0.71, Plots.text("Ángulo de contacto promedio: $(round(results.contact_angle_mean, digits=2))°", 11, :left, :bold))
    Plots.annotate!(p4, 0.1, 0.63, Plots.text("Diámetro base: $(round(results.base_diameter, digits=2)) px", 10, :left))
    Plots.annotate!(p4, 0.1, 0.55, Plots.text("Altura de gota: $(round(results.droplet_height, digits=2)) px", 10, :left))
    Plots.annotate!(p4, 0.1, 0.47, Plots.text("Inclinación sustrato: $(round(results.substrate_tilt, digits=2))°", 10, :left))
    Plots.annotate!(p4, 0.1, 0.39, Plots.text("Relación aspecto (H/D): $(round(results.droplet_height/results.base_diameter, digits=3))", 10, :left))

    # Info adicional
    asimetria = abs(results.contact_angle_left - results.contact_angle_right)
    Plots.annotate!(p4, 0.1, 0.31, Plots.text("Asimetría ángulos: $(round(asimetria, digits=2))°", 9, :left))
    Plots.annotate!(p4, 0.1, 0.23, Plots.text("Apex: ($(results.apex_point[1]), $(results.apex_point[2]))", 9, :left))

    # Combinar plots
    p_final = Plots.plot(p1, p2, p3, p4, layout=(2, 2), size=(1400, 900))

    # Guardar si se especifica
    if !isnothing(save_path)
        Plots.savefig(p_final, save_path)
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
println("""
╔════════════════════════════════════════════════════════════════════╗
║              FIX COMPLETO - Contact Point Detection               ║
╚════════════════════════════════════════════════════════════════════╝

INSTRUCCIONES:
  1. Copia esta función completa
  2. Reemplaza find_substrate_line_robust en tu archivo original
  3. Ejecuta el análisis nuevamente

CAMBIOS CLAVE:
  ✅ Busca contactos en el 10% INFERIOR de la gota (mayor Y)
  ✅ Divide en lado izquierdo/derecho basado en el apex
  ✅ En cada lado toma: punto MÁS BAJO + MÁS EXTREMO
  ✅ Validaciones múltiples con mensajes claros
  ✅ Estrategias fallback para casos problemáticos
  ✅ Diagnóstico completo con emojis para fácil lectura

DIAGNÓSTICO:
  Los mensajes mostrarán:
  - 📍 Ubicación del apex y contactos
  - 📊 Estadísticas del contorno
  - 🎯 Región de búsqueda
  - ↔️ Separación entre contactos
  - ⚠️ Advertencias si algo está mal

Si aún falla:
  - Verifica que la imagen binarizada sea correcta (debug=true)
  - Ajusta threshold (prueba 0.3, 0.4, 0.5, 0.6)
  - Verifica que invert esté correcto
""")

# Descomentar y modificar con tu ruta de imagen:
filename = "prueba sesil 2.png"
image_path = joinpath(@__DIR__, "data", "samples", filename)
results, img, img_cleaned = analyze_droplet(image_path,
    blur_sigma=1,
    threshold=0.1,
    fit_range=25,
    invert=false,       # Gota oscura, fondo claro
    pixel_size=1.0,
    debug=true)  # calibración en μm/pixel


print_results(results)
plot_results(img, results, img_cleaned)
println("Script ASDA cargado. Usa analyze_droplet(\"ruta/imagen.png\") para analizar una gota.")