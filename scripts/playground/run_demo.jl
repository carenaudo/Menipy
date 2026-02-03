# run_demo.jl
# Script to execute Sessile Drop Analysis in the Julia Interpreter

# 1. Include the analysis module
# 1. Include the analysis module
# Ensure you are in the directory containing this file (d:\programacion\Menipy)
include("sessile_drop_analysis.jl")
using .SessileDropAnalysis

# 2. Define image paths
# These are relative to the project root
img1 = "./data/samples/prueba sesil 2.png"
img2 = "./data/samples/gota depositada 1.png"

# Common parameters
params = (
    pixel_size_m=2.88e-5,
    rho=1000.0,
    show_plot=true,
    save_plot=true,
    debug=true  # Enable intermediate stage visualization
)

println("="^60)
println("SESSILE DROP ANALYSIS DEMO")
println("="^60)

# 3. Run analysis for Image 1
println("\n[1/2] Processing: $img1")
try
    det1, angles1, plot1 = SessileDropAnalysis.run_analysis_with_plot(img1; params...)
    if !isnothing(det1)
        println("   -> Success. Bond number: $(round(angles1["young_laplace"]["bond_number"], digits=3))")
    end
catch e
    println("   -> Failed: $e")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n" * "-"^60)

# 4. Run analysis for Image 2
println("\n[2/2] Processing: $img2")
try
    det2, angles2, plot2 = SessileDropAnalysis.run_analysis_with_plot(img2; params...)
    if !isnothing(det2)
        println("   -> Success. Bond number: $(round(angles2["young_laplace"]["bond_number"], digits=3))")
        # To display plot in VS Code or plot pane again:
        # display(plot2)
    end
catch e
    println("   -> Failed: $e")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n" * "="^60)
println("Demo complete.")
println("Check for generated files:")
println(" - *_contour.txt")
println(" - *_analysis.png")
