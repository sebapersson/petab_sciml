include(joinpath(@__DIR__, "models.jl"))

# Use data from test-case 001
cp(joinpath(@__DIR__, "..", "..", "001", "petab", "measurements.tsv"),
   joinpath(@__DIR__, "..", "petab", "measurements.tsv"); force = true)
