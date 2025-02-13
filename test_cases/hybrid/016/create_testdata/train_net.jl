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
input_data1 = rand(rng, Float32, 10, 10, 3, 1)
input_data2 = rand(rng, Float32, 10, 10, 3, 1)
output_data1 = 0.8
output_data2 = 1.0
function loss(x, p)
    model_output1 = nn_model(input_data1, x, st)[1]
    model_output2 = nn_model(input_data2, x, st)[1]
    loss1 = (model_output1[1] - output_data1)^2
    loss2 = (model_output2[1] - output_data2)^2
    return loss1 + loss2
end
x0 = ComponentArray(pnn)
x0 .*= 0.1
optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
solopt = solve(prob, OptimizationOptimisers.Adam(0.001), maxiters = 1000)

# Write neural-net parameters to file
nn_ps_to_h5(nn_model, solopt.u, joinpath(@__DIR__, "..", "petab", "net1_ps.hdf5"))
# Also save input data to a petab file
# Input 1
input_array1 = _reshape_array(input_data1[:, :, :], map_input)
input_array1 = permutedims(input_array1, reverse(1:ndims(input_array1)))
path_save = joinpath(@__DIR__, "..", "petab", "input_data1.hdf5")
isfile(path_save) && rm(path_save)
h5open(path_save, "w") do file
    write(file, "input1", input_array1)
end
# Input 2
input_array2 = _reshape_array(input_data2[:, :, :], map_input)
input_array2 = permutedims(input_array2, reverse(1:ndims(input_array2)))
path_save = joinpath(@__DIR__, "..", "petab", "input_data2.hdf5")
isfile(path_save) && rm(path_save)
h5open(path_save, "w") do file
    write(file, "input1", input_array2)
end
