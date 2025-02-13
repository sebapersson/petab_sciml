#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values
=#

using QuasiMonteCarlo, ForwardDiff, Optimization, OptimizationOptimisers
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

## Net1
# Reuse the net from test case 001 (as it takes ages to train)
cp(joinpath(@__DIR__, "..", "..", "001", "petab", "net1_ps.hdf5"),
            joinpath(@__DIR__, "..", "petab", "net1_ps.hdf5"))

## Net2
# Stage 1: the neural network should learn α*prey - β * prey * predator
# In this stage the network is trained directly on input data (x, y) and output data
# This can be argued cheating, but, this is just to get good tests
n = 3e3 |> Int64
α = 1.3
input_data = QuasiMonteCarlo.sample(n, [0.0, 0.0], [10.0, 10.0], LatinHypercubeSample())
output_data = [α*input_data[1, i]  for i in 1:n]
function loss2(x, p)
    loss = 0.0
    for i in eachindex(output_data)
        nnout = nn_model2(input_data[:, i], x, st2)[1]
        loss += (nnout[1] - output_data[i])^2
    end
    return loss
end
# Proved beneficial to do Adamn + LBFGS
x0 = ComponentArray(pnn2) .* 0.1
optf = OptimizationFunction(loss2, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
sol1 = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 2000)
x0 .= sol1.u
sol2 = solve(prob, Optimization.LBFGS(), maxiters = 2000)
nn_ps_to_h5(nn_model2, sol2.u, joinpath(@__DIR__, "..", "petab", "net2_ps.hdf5"))
