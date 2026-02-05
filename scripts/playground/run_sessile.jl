#!/usr/bin/env julia
# Minimal wrapper to launch the viewer with one image argument
if length(ARGS) < 1
    println("usage: run_sessile.jl image.png [options...]")
    exit(1)
end

include(joinpath(@__DIR__, "sessile_viewer.jl"))
main()  # uses ArgParse inside sessile_viewer.jl to read ARGS
