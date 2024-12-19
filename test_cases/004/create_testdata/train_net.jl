#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values.
=#

using Optimization, OptimizationOptimisers
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

# The following should be fulfilled for input -> output:
# [10.0, 20.0] -> [0.8]
# [1.0, 1.0] -> [1.0]
input_data = [[10.0, 20.0], [1.0, 1.0]]
output_data = [0.8, 1.0]
function loss(x, p)
    model_output1 = nn_model(input_data[1], x, st)[1]
    model_output2 = nn_model(input_data[2], x, st)[1]
    return (model_output1[1] - output_data[1])^2 + (model_output2[1] - output_data[2])^2
end
x0 = ComponentArray(pnn) .* 0.1
optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
solopt = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 10000)

# Write neural-net parameters to file
nn_ps_to_h5(nn_model, solopt.u, joinpath(@__DIR__, "..", "petab", "net1_ps.hf5"))
