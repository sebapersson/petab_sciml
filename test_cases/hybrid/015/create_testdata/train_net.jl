#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values.
=#

using Optimization, OptimizationOptimisers, StableRNGs
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

# Given the "image" input the output should for the net be [0.8].
rng = StableRNG(1)
input_data = rand(rng, Float32, 10, 10, 3, 1)
output_data = 0.8
function loss(x, p)
    model_output = nn_model(input_data, x, st)[1]
    return (model_output[1] - output_data)^2
end
x0 = ComponentArray(pnn)
x0 .*= 0.1
optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
solopt = solve(prob, OptimizationOptimisers.Adam(0.001), maxiters = 1000)

# Write neural-net parameters to file
nn_ps_to_h5(nn_model, solopt.u, joinpath(@__DIR__, "..", "petab", "net1_ps.hdf5"))
# Also save input data to a petab file
input_array = _reshape_array(input_data[:, :, :], map_input)
# Python is row-major
input_array = permutedims(input_array, reverse(1:ndims(input_array)))
path_save = joinpath(@__DIR__, "..", "petab", "input_data.hdf5")
isfile(path_save) && rm(path_save)
h5open(path_save, "w") do file
    write(file, "input1", input_array)
end
