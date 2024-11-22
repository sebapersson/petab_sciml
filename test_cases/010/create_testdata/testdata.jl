#=
    Hard-coded likelihood and simulated values for test case 001
=#

using FiniteDifferences, YAML
include(joinpath(@__DIR__, "models.jl"))
Random.seed!(123)

function compute_nllh(x, oprob::ODEProblem, solver, measurements::DataFrame; abstol = 1e-9,
                      reltol = 1e-9)::Real
    mprey = measurements[measurements[!, :observableId] .== "prey", :measurement]
    mpredator = measurements[measurements[!, :observableId] .== "predator", :measurement]
    tsave = unique(measurements.time)

    _p = convert.(eltype(x), x)
    _p[1:4] .= x[1:4]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, solver, abstol = abstol, reltol = reltol, saveat = tsave)
    prey, predator = sol[1, :], sol[2, :]

    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        model_output = nn_model1([prey[i], predator[i]], x.p_net1, st1)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - model_output)^2 / σ^2
    end
    for i in eachindex(mpredator)
        model_output = nn_model2([x[1], predator[i]], x.p_net2, st2)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - model_output)^2 / σ^2
    end
    return nllh
end

## Parameter estimation problem setup
measurements = CSV.read(joinpath(@__DIR__, "..", "petab", "measurements.tsv"), DataFrame)
# Objective function
xmech = (α = 1.3, δ = 1.8, β = 0.9, γ = 0.8)
x = ComponentArray(merge(xmech, (p_net1=pnn1, p_net2 = pnn2)))
# Read neural net parameters, and assign to x
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
df_ps_nn = CSV.read(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), DataFrame)
set_ps_net!(x.p_net1, df_ps_nn, :net1, nn_model1)
set_ps_net!(x.p_net2, df_ps_nn, :net2, nn_model2)

## Compute model values
_f = (x) -> compute_nllh(x, oprob, Vern9(), measurements; abstol = 1e-12, reltol = 1e-12)
llh = _f(x) .* -1
# High order finite-difference scheme
llh_grad = FiniteDifferences.grad(central_fdm(5, 1), _f, x)[1] .* -1
# Simulated values, order as in measurements
oprob.p .= x[1:4]
sol = solve(oprob, Vern9(), abstol = 1e-9, reltol = 1e-9,
            saveat = unique(measurements.time))
prey_vals = [nn_model1(sol[:, i], x.p_net1, st1)[1][1] for i in eachindex(sol.t)]
predator_vals = [nn_model2([1.3, sol[2, i]], x.p_net2, st2)[1][1] for i in eachindex(sol.t)]
simulated_values = vcat(prey_vals, predator_vals)

## Write values for saving to file
# YAML problem file
solutions = Dict(:llh => llh,
                 :tol_llh => 1e-3,
                 :tol_grad_llh => 1e-1,
                 :tol_simulations => 1e-3,
                 :grad_llh_files => ["grad_llh.tsv"],
                 :simulation_files => ["simulations.tsv"])
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
# Simulated values
simulations_df = deepcopy(measurements)
rename!(simulations_df, "measurement" => "simulation")
simulations_df.simulation .= simulated_values
CSV.write(joinpath(@__DIR__, "..", "simulations.tsv"), simulations_df, delim = '\t')
# Gradient values
df_net = deepcopy(df_ps_nn)
df_net.value .= vcat(llh_grad.p_net1, llh_grad.p_net2)
df_mech = DataFrame(parameterId = ["alpha", "delta", "beta", "gamma"],
                    value = llh_grad[1:4])
df_grad = vcat(df_mech, df_net)
CSV.write(joinpath(@__DIR__, "..", "grad_llh.tsv"), df_grad, delim = '\t')
# Write problem yaml
problem_yaml = Dict(
    :format_version => 1,
    :parameter_file => "parameters_ude.tsv",
    :problems => [Dict(
        :condition_files => ["conditions.tsv"],
        :measurement_files => ["measurements.tsv"],
        :observable_files => ["observables.tsv"],
        :sbml_files => ["lv.xml"],
        :mapping_tables => "mapping_table.tsv")],
    :extensions => Dict(
        :petab_sciml => Dict(
            :net_files => ["net1.yaml", "net2.yaml"],
            :hybridization => Dict(
                :net1 => Dict(
                    :input => "ode",
                    :output => "observable"),
                :net2 => Dict(
                    :input => "ode",
                    :output => "observable")))))
YAML.write_file(joinpath(@__DIR__, "..", "petab", "problem_ude.yaml"), problem_yaml)
