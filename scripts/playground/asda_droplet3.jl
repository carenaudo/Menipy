using Images, ImageFiltering, ImageMorphology, ImageSegmentation
using Statistics, LinearAlgebra, Polynomials
using Plots

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
    # Etiquetar componentes conectados
    labels = label_components(img_binary)

    # Contar pixeles de cada componente (excluyendo background = 0)
    component_sizes = Dict{Int,Int}()
    components_touch_top = Set{Int}()

    h, w = size(img_binary)

    for i in 1:h
        for j in 1:w
            label = labels[i, j]
            if label > 0
                component_sizes[label] = get(component_sizes, label, 0) + 1
                # Marcar si toca el borde superior
                if i <= 5  # Primeras 5 filas
                    push!(components_touch_top, label)
                end
            end
        end
    end

    if isempty(component_sizes)
        return falses(size(img_binary))
    end

    # Filtrar componentes que tocan el borde superior
    valid_components = Dict(k => v for (k, v) in component_sizes if !(k in components_touch_top))

    if isempty(valid_components)
        # Si todos tocan el borde, usar el original
        valid_components = component_sizes
    end

    # Encontrar el componente más grande válido
    largest_label = argmax(valid_components)

    # Crear máscara con solo el componente más grande
    largest_component = labels .== largest_label

    return largest_component
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

    # Encontrar baseline (la verdadera línea de sustrato)
    # Debe estar DEBAJO de todos los puntos del contorno
    max_y = maximum(y_coords)
    min_y = minimum(y_coords)

    # 🔧 La baseline debería estar justo debajo del contorno
    # Buscamos el punto más bajo del contorno y añadimos un margen
    baseline_y = max_y + 5  # Sustrato está ~5 píxeles debajo del punto más bajo

    # Encontrar puntos de contacto: donde el contorno está más cerca de la baseline
    # Buscar en la región inferior (bottom 10%)
    height_range = max_y - min_y
    threshold_y = max_y - 0.10 * height_range
    base_candidates = filter(p -> p[2] >= threshold_y, contour)

    if length(base_candidates) < 2
        sorted_by_y = sort(contour, by=p -> -p[2])
        base_candidates = sorted_by_y[1:min(20, length(sorted_by_y))]
    end

    if length(base_candidates) < 2
        return Tuple{Int,Int}[], 0.0, (0, 0), (0, 0), baseline_y, (mean(x_coords), min_y)
    end

    # 🔧 CLAVE: Encontrar los puntos extremos izquierdo y derecho
    # que están cerca de la baseline
    x_base_coords = [p[1] for p in base_candidates]
    y_base_coords = [p[2] for p in base_candidates]

    # Punto de contacto izquierdo: mínimo X entre los candidatos de base
    left_idx = argmin(x_base_coords)
    left_contact = base_candidates[left_idx]

    # Punto de contacto derecho: máximo X entre los candidatos de base
    right_idx = argmax(x_base_coords)
    right_contact = base_candidates[right_idx]

    # 🔧 Proyectar estos puntos hacia la baseline real
    left_contact = (left_contact[1], Int(round(baseline_y)))
    right_contact = (right_contact[1], Int(round(baseline_y)))

    # Recalcular baseline como promedio de Y de los contactos proyectados
    baseline_y = (left_contact[2] + right_contact[2]) / 2

    # Ajustar línea para sustrato (horizontal en este caso)
    slope = 0.0  # 🔧 Asumir sustrato horizontal
    intercept = baseline_y

    # Si quieres detectar inclinación, usa esto:
    # x_contacts = Float64.([left_contact[1], right_contact[1]])
    # y_contacts = Float64.([left_contact[2], right_contact[2]])
    # slope = (y_contacts[2] - y_contacts[1]) / (x_contacts[2] - x_contacts[1])
    # intercept = y_contacts[1] - slope * x_contacts[1]

    substrate_tilt = atand(slope)

    # Generar línea de sustrato extendida
    x_min = minimum(x_coords)
    x_max = maximum(x_coords)
    x_line = range(x_min - 20, x_max + 20, length=150)
    substrate_line = [(Int(round(x)), Int(round(slope * x + intercept))) for x in x_line]

    # Encontrar apex (punto más alto = menor valor de y)
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
    invert=false)

    println("=== Análisis ASDA de Gota Sésil ===\n")
    println("1. Cargando y preprocesando imagen...")
    img, img_gray, img_filtered, img_binary = load_and_preprocess(image_path,
        blur_sigma=blur_sigma,
        threshold=threshold,
        invert=invert)

    println("2. Detectando contorno de la gota...")
    contour, img_cleaned = extract_contour_ordered(img_binary)

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
    invert=true,       # Gota oscura, fondo claro
    pixel_size=1.0)  # calibración en μm/pixel


print_results(results)
plot_results(img, results, img_cleaned)
println("Script ASDA cargado. Usa analyze_droplet(\"ruta/imagen.png\") para analizar una gota.")

