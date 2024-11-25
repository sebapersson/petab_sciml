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
input_data1 = rand(rng, 10, 10, 3, 1)
input_data2 = rand(rng, 10, 10, 3, 1)
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
ps_df = nn_ps_to_tidy(nn_model, solopt.u, :net1)
CSV.write(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), ps_df, delim = '\t')
# Also save input data to a petab file
order_jl, order_py = ["W", "H", "C"], ["C", "H", "W"]
imap = zeros(Int64, length(order_jl))
for i in eachindex(order_py)
    imap[i] = findfirst(x -> x == order_py[i], order_jl)
end
map = collect(1:length(order_py)) .=> imap
input_df1 = _array_to_tidy(input_data1[:, :, :]; mapping = map)
input_df2 = _array_to_tidy(input_data2[:, :, :]; mapping = map)
CSV.write(joinpath(@__DIR__, "..", "petab", "input_data1.tsv"), input_df1, delim = '\t')
CSV.write(joinpath(@__DIR__, "..", "petab", "input_data2.tsv"), input_df2, delim = '\t')
