#=
    Helper to train the neural-net for the testing
    For the tests it is beneficial if the neural-net has reasonable good values
=#

using QuasiMonteCarlo, ForwardDiff, Optimization, OptimizationOptimisers
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

# Stage 1: the neural network should learn γ*x*y - δ*y, where x and y are the inputs.
# In this stage the network is trained directly on input data (x, y) and outout data
# This is just to get good test parameters
n = 2e3 |> Int64
γ, δ = 0.8, 1.8
input_data = QuasiMonteCarlo.sample(n, [0.0, 0.0], [10.0, 10.0], LatinHypercubeSample())
output_data = [sum(input_data[:, i])*γ - input_data[2, i]*δ for i in 1:n]
function loss1(x, p)
    loss = 0.0
    for i in eachindex(output_data)
        nnout = nn_model(input_data[:, i], x, st)[1]
        loss += (nnout[1] - output_data[i])^2
    end
    return loss
end
# Proved beneficial to do Adamn + LBFGS
x0 = ComponentArray(pnn) .* 0.1
optf = OptimizationFunction(loss1, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
sol1 = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 10000)
x0 .= sol1.u
sol2 = solve(prob, Optimization.LBFGS(), maxiters = 2000)
xstage1 = deepcopy(sol2.u)

# Stage 2: train on ODE-solver output. This helps, as even though a low error is obtained
# in stage 1, results when solving the ODE can still be very sensitive to small numerical
# errors.
tsave = 0.1:0.1:10.0
solref = solve(oprob_simulate, Vern9(), abstol = 1e-9, reltol = 1e-9, saveat = tsave) |>
    Array
function loss2(x, p)
    _p = convert.(eltype(x), oprob_nn.p)
    _p.p_net1 .= x
    _oprob = remake(oprob_nn, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1e-9, reltol = 1e-9, saveat = tsave) |>
        Array
    return sum((sol - solref).^2)
end
# Proved beneficial to do Adamn + LBFGS
x0 = deepcopy(xstage1)
optf = OptimizationFunction(loss2, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, x0, Float64[])
sol1 = solve(prob, OptimizationOptimisers.Adam(0.001), maxiters = 10000)
x0 .= sol1.u
sol2 = solve(prob, Optimization.LBFGS(), maxiters = 8000)

nn_ps_to_h5(nn_model, sol2.u, joinpath(@__DIR__, "..", "petab", "net1_ps.hf5"))
