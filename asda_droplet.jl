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
    contour::Vector{Tuple{Int,Int}}
    substrate_line::Vector{Tuple{Int,Int}}
end

"""
    load_and_preprocess(image_path::String; blur_sigma=2.0, threshold=0.5)

Carga y preprocesa la imagen de la gota
"""
function load_and_preprocess(image_path::String; blur_sigma=1.0, threshold=0.2)
    # Cargar imagen
    img = load(image_path)
    
    # Convertir a escala de grises
    img_gray = Gray.(img)
    
    # Aplicar filtro Gaussiano para reducir ruido
    img_filtered = imfilter(img_gray, Kernel.gaussian(blur_sigma))
    
    # Binarizar con threshold adaptativo
    img_binary = img_filtered .> threshold
    
    return img, img_gray, img_filtered, img_binary
end

"""
    detect_contour(img_binary::BitMatrix)

Detecta el contorno de la gota usando operaciones morfológicas
"""
function detect_contour(img_binary::BitMatrix)
    # Operaciones morfológicas para limpiar la imagen
    img_cleaned = closing(opening(img_binary))
    
    # Detectar bordes
    edges = img_cleaned .⊻ erode(img_cleaned)
    
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
    
    # Ordenar puntos del contorno (de izquierda a derecha)
    sort!(contour_points, by = p -> p[1])
    
    return contour_points, img_cleaned
end

"""
    find_substrate_line(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})

Encuentra la línea de sustrato (baseline) analizando la parte inferior del contorno
"""
function find_substrate_line(contour::Vector{Tuple{Int,Int}}, img_size::Tuple{Int,Int})
    if isempty(contour)
        return Tuple{Int,Int}[], 0.0, (0,0), (0,0)
    end
    
    # Encontrar los puntos más bajos (mayor coordenada y)
    max_y = maximum(p[2] for p in contour)
    
    # Puntos cerca de la base (últimos 10% en altura)
    threshold_y = max_y - 0.1 * img_size[1]
    base_points = filter(p -> p[2] >= threshold_y, contour)
    
    if length(base_points) < 2
        return Tuple{Int,Int}[], 0.0, (0,0), (0,0)
    end
    
    # Encontrar puntos de contacto (extremos izquierdo y derecho en la base)
    left_contact = base_points[argmin([p[1] for p in base_points])]
    right_contact = base_points[argmax([p[1] for p in base_points])]
    
    # Ajustar línea recta a los puntos de la base
    x_base = [Float64(p[1]) for p in base_points]
    y_base = [Float64(p[2]) for p in base_points]
    
    # Regresión lineal para la línea de sustrato
    A = hcat(x_base, ones(length(x_base)))
    coeff = A \ y_base
    slope = coeff[1]
    intercept = coeff[2]
    
    # Ángulo de inclinación del sustrato (en grados)
    substrate_tilt = atand(slope)
    
    # Generar puntos de la línea de sustrato
    x_line = range(left_contact[1], right_contact[1], length=100)
    substrate_line = [(Int(round(x)), Int(round(slope * x + intercept))) for x in x_line]
    
    baseline_y = mean(y_base)
    
    return substrate_line, substrate_tilt, left_contact, right_contact, baseline_y
end

"""
    calculate_contact_angle(contour::Vector{Tuple{Int,Int}}, contact_point::Tuple{Int,Int}, 
                           is_left::Bool; fit_range=20)

Calcula el ángulo de contacto en un punto de contacto dado
"""
function calculate_contact_angle(contour::Vector{Tuple{Int,Int}}, contact_point::Tuple{Int,Int}, 
                                is_left::Bool; fit_range=20)
    # Encontrar puntos cerca del punto de contacto para ajustar una línea tangente
    cp_x, cp_y = contact_point
    
    # Filtrar puntos del contorno cerca del punto de contacto
    nearby_points = filter(p -> abs(p[1] - cp_x) <= fit_range && p[2] <= cp_y, contour)
    
    if length(nearby_points) < 3
        return 90.0  # Valor por defecto si no hay suficientes puntos
    end
    
    # Ordenar por distancia al punto de contacto
    sort!(nearby_points, by = p -> sqrt((p[1]-cp_x)^2 + (p[2]-cp_y)^2))
    
    # Tomar los primeros N puntos
    n_points = min(fit_range, length(nearby_points))
    fit_points = nearby_points[1:n_points]
    
    # Extraer coordenadas
    x_fit = [Float64(p[1]) for p in fit_points]
    y_fit = [Float64(p[2]) for p in fit_points]
    
    # Ajustar polinomio de grado 2 para obtener mejor tangente
    if length(x_fit) >= 3
        p = fit(x_fit, y_fit, 2)
        # Calcular derivada en el punto de contacto
        dp = derivative(p)
        slope = dp(Float64(cp_x))
    else
        # Regresión lineal simple
        A = hcat(x_fit, ones(length(x_fit)))
        coeff = A \ y_fit
        slope = coeff[1]
    end
    
    # Calcular ángulo
    angle_rad = atan(abs(slope))
    angle_deg = rad2deg(angle_rad)
    
    # Para gota sésil, el ángulo de contacto es medido desde la horizontal
    # Si es el lado izquierdo, el ángulo es 180 - angle
    # Si es el lado derecho, el ángulo es angle
    if is_left
        contact_angle = 180.0 - angle_deg
    else
        contact_angle = angle_deg
    end
    
    # Limitar entre 0 y 180 grados
    contact_angle = clamp(contact_angle, 0.0, 180.0)
    
    return contact_angle
end

"""
    calculate_droplet_dimensions(contour::Vector{Tuple{Int,Int}}, 
                                 left_contact::Tuple{Int,Int}, 
                                 right_contact::Tuple{Int,Int})

Calcula dimensiones de la gota (altura, diámetro base)
"""
function calculate_droplet_dimensions(contour::Vector{Tuple{Int,Int}}, 
                                     left_contact::Tuple{Int,Int}, 
                                     right_contact::Tuple{Int,Int})
    if isempty(contour)
        return 0.0, 0.0
    end
    
    # Diámetro base
    base_diameter = abs(right_contact[1] - left_contact[1])
    
    # Altura de la gota (punto más alto del contorno)
    min_y = minimum(p[2] for p in contour)
    baseline_y = (left_contact[2] + right_contact[2]) / 2
    droplet_height = baseline_y - min_y
    
    return base_diameter, droplet_height
end

"""
    analyze_droplet(image_path::String; blur_sigma=2.0, threshold=0.5, 
                   fit_range=20, pixel_size=1.0)

Función principal que realiza el análisis completo de la gota
"""
function analyze_droplet(image_path::String; 
                        blur_sigma=2.0, 
                        threshold=0.5, 
                        fit_range=20,
                        pixel_size=1.0)
    
    println("Cargando y preprocesando imagen...")
    img, img_gray, img_filtered, img_binary = load_and_preprocess(image_path, 
                                                                   blur_sigma=blur_sigma, 
                                                                   threshold=threshold)
    
    println("Detectando contorno...")
    contour, img_cleaned = detect_contour(img_binary)
    
    if isempty(contour)
        error("No se pudo detectar el contorno de la gota")
    end
    
    println("Encontrando línea de sustrato...")
    substrate_line, substrate_tilt, left_contact, right_contact, baseline_y = 
        find_substrate_line(contour, size(img_binary))
    
    println("Calculando ángulos de contacto...")
    ca_left = calculate_contact_angle(contour, left_contact, true, fit_range=fit_range)
    ca_right = calculate_contact_angle(contour, right_contact, false, fit_range=fit_range)
    ca_mean = (ca_left + ca_right) / 2
    
    println("Calculando dimensiones...")
    base_diameter, droplet_height = calculate_droplet_dimensions(contour, left_contact, right_contact)
    
    # Convertir a unidades físicas si se proporciona pixel_size
    base_diameter_physical = base_diameter * pixel_size
    droplet_height_physical = droplet_height * pixel_size
    
    # Crear estructura de resultados
    results = DropletAnalysis(
        ca_left, ca_right, ca_mean,
        base_diameter_physical, droplet_height_physical,
        substrate_tilt, baseline_y,
        left_contact, right_contact,
        contour, substrate_line
    )
    
    # Visualizar resultados
    plot_results(img, results, img_cleaned)
    
    return results, img, img_cleaned
end

"""
    plot_results(img, results::DropletAnalysis, img_cleaned)

Visualiza los resultados del análisis
"""
function plot_results(img, results::DropletAnalysis, img_cleaned)
    # Crear figura con múltiples subplots
    p1 = plot(img, title="Imagen Original", axis=false, ticks=false)
    
    p2 = plot(Gray.(img_cleaned), title="Imagen Binarizada", 
              axis=false, ticks=false)
    
    # Plot del análisis
    p3 = plot(img, title="Análisis ASDA", axis=false, ticks=false)
    
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
        plot!(p3, x_substrate, y_substrate, color=:blue, linewidth=2, label="Sustrato")
    end
    
    # Marcar puntos de contacto
    scatter!(p3, [results.left_contact_point[1]], [results.left_contact_point[2]], 
             color=:green, markersize=8, label="Contacto Izq")
    scatter!(p3, [results.right_contact_point[1]], [results.right_contact_point[2]], 
             color=:yellow, markersize=8, label="Contacto Der")
    
    # Mostrar resultados como texto
    p4 = plot(framestyle=:none, xlims=(0,1), ylims=(0,1), legend=false)
    annotate!(p4, 0.1, 0.9, text("RESULTADOS DEL ANÁLISIS", 12, :left, :bold))
    annotate!(p4, 0.1, 0.8, text("Ángulo de contacto izq: $(round(results.contact_angle_left, digits=2))°", 10, :left))
    annotate!(p4, 0.1, 0.7, text("Ángulo de contacto der: $(round(results.contact_angle_right, digits=2))°", 10, :left))
    annotate!(p4, 0.1, 0.6, text("Ángulo de contacto promedio: $(round(results.contact_angle_mean, digits=2))°", 10, :left, :bold))
    annotate!(p4, 0.1, 0.5, text("Diámetro base: $(round(results.base_diameter, digits=2)) px", 10, :left))
    annotate!(p4, 0.1, 0.4, text("Altura de gota: $(round(results.droplet_height, digits=2)) px", 10, :left))
    annotate!(p4, 0.1, 0.3, text("Inclinación sustrato: $(round(results.substrate_tilt, digits=2))°", 10, :left))
    annotate!(p4, 0.1, 0.2, text("Relación aspecto: $(round(results.droplet_height/results.base_diameter, digits=3))", 10, :left))
    
    # Combinar plots
    plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    display(current())
end

"""
    print_results(results::DropletAnalysis)

Imprime los resultados del análisis en consola
"""
function print_results(results::DropletAnalysis)
    println("\n" * "="^60)
    println("RESULTADOS DEL ANÁLISIS DE GOTA SÉSIL (ASDA)")
    println("="^60)
    println("Ángulo de Contacto:")
    println("  - Lado Izquierdo:  $(round(results.contact_angle_left, digits=2))°")
    println("  - Lado Derecho:    $(round(results.contact_angle_right, digits=2))°")
    println("  - Promedio:        $(round(results.contact_angle_mean, digits=2))°")
    println("\nDimensiones:")
    println("  - Diámetro Base:   $(round(results.base_diameter, digits=2)) unidades")
    println("  - Altura:          $(round(results.droplet_height, digits=2)) unidades")
    println("  - Relación H/D:    $(round(results.droplet_height/results.base_diameter, digits=3))")
    println("\nSustrato:")
    println("  - Inclinación:     $(round(results.substrate_tilt, digits=2))°")
    println("="^60 * "\n")
end

# =============================================================================
# EJEMPLO DE USO
# =============================================================================

# Descomentar y modificar con tu ruta de imagen:
filename = "prueba sesil 2.png"
image_path = joinpath(@__DIR__,"data","samples", filename)
results, img, img_cleaned = analyze_droplet(image_path, 
                                            blur_sigma=2.0,
                                            threshold=0.5,
                                            fit_range=30,
                                            pixel_size=1.0)  # calibración en μm/pixel
print_results(results)
plot_results(img, results, img_cleaned)
println("Script ASDA cargado. Usa analyze_droplet(\"ruta/imagen.png\") para analizar una gota.")