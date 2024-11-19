#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values.
=#

using Optimization, OptimizationOptimisers
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)
# Given input [1.0, input2] the output should for the net be [0.8]. Here, train until this
# is acheived.
input2 = 1.0
input_data = [1.0, input2]
output_data = 0.8
function loss(x, p)
    model_output = nn_model(input_data, x, st)[1]
    return (model_output[1] - output_data)^2
end
x0 = ComponentArray(pnn) .* 0.1
optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
solopt = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 10000)

# Write neural-net parameters to file
ps_df = nn_ps_to_tidy(nn_model, solopt.u, :net1)
CSV.write(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), ps_df, delim = '\t')
