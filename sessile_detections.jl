# Assuming the module is loaded or included
include("sessile_detections_module.jl")
using .SessileDrop
using Plots
using DelimitedFiles

function main()
    # Define parameters (match your Python example)
    img_path = "./data/samples/prueba sesil 2.png"
    px_size = 2.88e-5  # meters per pixel
    density = 1000.0   # kg/m3

    println("Loading image: $img_path")

    # 1. Run Detection
    det = SessileDrop.sessile_drop_adaptive(img_path)

    if isnothing(det)
        println("Could not detect drop. Exiting.")
        return
    end

    # --- SAVE CONTOUR TO TXT ---
    contour_matrix = mapreduce(p -> [p[1] p[2]], vcat, det.full_contour)
    # Use basename for file naming
    cont_filename = "julia_contour_$(basename(img_path)).txt"
    writedlm(cont_filename, contour_matrix, ',')
    println("Saved contour to $cont_filename")

    # --- PRINT JULIA DETECTION OUTPUT ---
    println("\n--- JULIA DETECTION OUTPUT ---")
    println("substrate_y: $(det.substrate_y)")
    println("cp_left: $(det.cp_left)")
    println("cp_right: $(det.cp_right)")
    println("apex: $(det.apex)")
    println("height_px: $(det.height_px)")
    println("base_width_px: $(det.base_width_px)")
    # Convert ROI format from (r_range, c_range) to (x1_y1_x2_y2 tuple style like Python)
    # Python ROI: (x1, y1, x2, y2)
    # Julia ROI: (row_start:row_end, col_start:col_end) -> (y1:y2, x1:x2)
    roi_formatted = (
        det.roi_coords[2].start - 1, # x1 (python 0-idx vs julia 1-idx adjustments? Python max(0, start))
        det.roi_coords[1].start - 1, # y1
        det.roi_coords[2].stop,      # x2
        det.roi_coords[1].stop       # y2
    )
    # Note: Python `roi_coords` logic: 
    # (max(0, x_left - pad), max(0, min_y - pad), min(w, x_right+pad), min(h, sub_y+pad))
    # Julia: (max(1, ...):min(h, ...), ...)
    # Visual check needs to be approx same.
    println("roi_coords: $roi_formatted")
    println("dome_points_count: $(length(det.dome_points))")
    println("contour_points_count: $(length(det.full_contour))")
    println("-------------------------------\n")

    # 2. Compute all angles (Dictionary output)
    println("\nComputing all metrics...")
    angles = SessileDrop.compute_contact_angles_from_detection(det,
        rho=density,
        sigma=72e-3,
        pixel_size_m=px_size
    )

    # 3. Print Dictionary Results
    println("\n--- DETAILED REPORT ---")

    println("\nApex-based spherical cap:")
    display(angles["apex_spherical"])

    println("\nTangent method (left/right):")
    display(angles["tangent"])

    println("\nSpherical-cap fit:")
    display(angles["spherical_fit"])

    # Note: Elliptical fit was omitted in the main port as it requires 
    # complex non-linear geometry fitting not standard in basic Images.jl.

    println("\nYoung-Laplace fit (Fixed Sigma):")
    display(angles["young_laplace"])

    # Plot results
    println("\nDisplaying results plot...")
    SessileDrop.plot_results(det, angles)

    # 4. Run the specific comparison method
    println("\n\nRunning Detailed Y-L Comparison:")
    SessileDrop.compare_yl_methods(det, pixel_size_m=px_size, rho=density)

    # 5. Second Image Example (as in Python script)
    img_path2 = "./data/samples/gota depositada 1.png"
    println("\n\nProcessing Second Image: $img_path2")
    det2 = SessileDrop.sessile_drop_adaptive(img_path2)

    if !isnothing(det2)
        println("Computing metrics for second image...")
        angles2 = SessileDrop.compute_contact_angles_from_detection(det2,
            rho=density, sigma=72e-3, pixel_size_m=px_size
        )
        SessileDrop.compare_yl_methods(det2, pixel_size_m=px_size, rho=density)
        println("Displaying second result plot...")
        SessileDrop.plot_results(det2, angles2)
    end
end

# Run the main function
main()