#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values.
=#

using Optimization, OptimizationOptimisers
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

## Net1
# Given input [1.0, 1.0] the output should for the net be [0.8]. Here, train until this
# is acheived.
input_data = [1.0, 1.0]
output_data = 0.8
function loss1(x, p)
    model_output = nn_model1(input_data, x, st1)[1]
    return (model_output[1] - output_data)^2
end
x0 = ComponentArray(pnn1) .* 0.1
optf = OptimizationFunction(loss1, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
solopt = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 10000)
# Write neural-net parameters to file
ps_df1 = nn_ps_to_tidy(nn_model1, solopt.u, :net1)

## Net2
input_data = [2.0, 2.0]
output_data = 0.9
function loss2(x, p)
    model_output = nn_model2(input_data, x, st2)[1]
    return (model_output[1] - output_data)^2
end
x0 = ComponentArray(pnn2) .* 0.1
optf = OptimizationFunction(loss2, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
solopt = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 10000)
# Write neural-net parameters to file
ps_df2 = nn_ps_to_tidy(nn_model2, solopt.u, :net2)

ps_df = vcat(ps_df1, ps_df2)
CSV.write(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), ps_df, delim = '\t')
