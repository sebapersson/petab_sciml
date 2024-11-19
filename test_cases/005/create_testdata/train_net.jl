#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values
=#

using QuasiMonteCarlo, ForwardDiff, Optimization, OptimizationOptimisers
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

# The neural network should learn x, where x and y are the inputs. The network is trained
# directly on input data (x, y) and outout data x. This can be argued cheeting, but, this
# is just to get good tests
n = 1e3 |> Int64
input_data = QuasiMonteCarlo.sample(n, [0.0, 0.0], [10.0, 10.0], LatinHypercubeSample())
output_data = [input_data[1, i] for i in 1:n]
function loss(x, p)
    loss = 0.0
    for i in eachindex(output_data)
        nnout = nn_model(input_data[:, i], x, st)[1]
        loss += (nnout[1] - output_data[i])^2
    end
    return loss
end
# Proved beneficial to do Adamn + LBFGS
x0 = ComponentArray(pnn) .* 0.1
optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
sol1 = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 5000)
x0 .= sol1.u
sol2 = solve(prob, Optimization.LBFGS(), maxiters = 2000)

# Write neural-net parameters to file
ps_df = nn_ps_to_tidy(nn_model, sol2.u, :net1)
CSV.write(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), ps_df, delim = '\t')
