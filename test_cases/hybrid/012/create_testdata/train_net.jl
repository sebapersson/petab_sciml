#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values.
=#

using Optimization, OptimizationOptimisers
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

## Net1
# Reuse the net from test case 001 (as it takes ages to train)
cp(joinpath(@__DIR__, "..", "..", "001", "petab", "net1_ps.hdf5"),
            joinpath(@__DIR__, "..", "petab", "net1_ps.hdf5");
            force = true)

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
nn_ps_to_h5(nn_model2, solopt.u, joinpath(@__DIR__, "..", "petab", "net2_ps.hdf5"))
