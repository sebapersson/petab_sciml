#=
    Hard-coded likelihood and simulated values for test case 001
=#

using FiniteDifferences, YAML
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "helper.jl"))
Random.seed!(123)

function compute_nllh(x, oprob::ODEProblem, solver, measurements::DataFrame; abstol = 1e-9,
                      reltol = 1e-9)::Real
    mprey = measurements[measurements[!, :observableId] .== "prey", :measurement]
    mpredator = measurements[measurements[!, :observableId] .== "predator", :measurement]
    tsave = unique(measurements.time)

    _oprob = remake(oprob, p = x)
    sol = solve(_oprob, solver, abstol = abstol, reltol = reltol, saveat = tsave)
    prey, predator = sol[1, :], sol[2, :]

    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - prey[i])^2 / σ^2
    end
    for i in eachindex(mpredator)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - predator[i])^2 / σ^2
    end
    return nllh
end

## Parameter estimation problem setup
measurements = CSV.read(joinpath(@__DIR__, "..", "petab", "measurements.tsv"), DataFrame)
# Objective function
x = deepcopy(p_ode)
# Read neural net parameters, and assign to x
df_ps_nn = CSV.read(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), DataFrame)
set_ps_net!(x.p_net1, df_ps_nn, :net1, nn_model)

## Compute model values
_f = (x) -> compute_nllh(x, oprob_nn, Vern9(), measurements; abstol = 1e-12, reltol = 1e-12)
llh = _f(x) .* -1
# High order finite-difference scheme
llh_grad = FiniteDifferences.grad(central_fdm(5, 1), _f, x)[1] .* -1
# Simulated values, order as in measurements
oprob_nn.p .= x
sol = solve(oprob_nn, Vern9(), abstol = 1e-9, reltol = 1e-9,
            saveat = unique(measurements.time))
simulated_values = vcat(sol[1, :], sol[2, :])

## Write values for saving to file
# YAML problem file
solutions = Dict(:llh => llh,
                 :tol_llh => 1e-3,
                 :tol_grad_llh => 1e-1,
                 :tol_simulations => 1e-3,
                 :tol_nn_output => 1e-3,
                 :grad_llh_files => ["grad_llh.tsv"],
                 :simulation_files => ["simulations.tsv"],
                 :nn_output_files => ["nn_output1", "nn_output2", "nn_output3"])
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
# Simulated values
simulations_df = deepcopy(measurements)
rename!(simulations_df, "measurement" => "simulation")
simulations_df.simulation .= simulated_values
CSV.write(joinpath(@__DIR__, "..", "simulations.tsv"), simulations_df, delim = '\t')
# Gradient values
df_net = deepcopy(df_ps_nn)
df_net.value .= llh_grad.p_net1
df_mech = DataFrame(parameterId = ["alpha", "delta", "beta"],
                    value = llh_grad[1:3])
df_grad = vcat(df_mech, df_net)
CSV.write(joinpath(@__DIR__, "..", "grad_llh.tsv"), df_grad, delim = '\t')
# Input and output files for neural network
df1 = DataFrame(netId = ["net1", "net1", "net1"],
                ioId = ["input1", "input2", "output1"],
                ioValue = [1.0, 1.0, nn_model([1.0, 1.0], x.p_net1, st)[1][1]])
df2 = DataFrame(netId = ["net1", "net1", "net1"],
                ioId = ["input1", "input2", "output1"],
                ioValue = [2.0, 1.0, nn_model([2.0, 1.0], x.p_net1, st)[1][1]])
df3 = DataFrame(netId = ["net1", "net1", "net1"],
                ioId = ["input1", "input2", "output1"],
                ioValue = [0.5, 0.7, nn_model([0.5, 0.7], x.p_net1, st)[1][1]])
CSV.write(joinpath(@__DIR__, "..", "nn_output1.tsv"), df1, delim = '\t')
CSV.write(joinpath(@__DIR__, "..", "nn_output2.tsv"), df2, delim = '\t')
CSV.write(joinpath(@__DIR__, "..", "nn_output3.tsv"), df3, delim = '\t')
