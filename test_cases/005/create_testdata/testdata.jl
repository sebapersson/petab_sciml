#=
    Hard-coded likelihood and simulated values for test case 001
=#

using FiniteDifferences, YAML
include(joinpath(@__DIR__, "models.jl"))
Random.seed!(123)

function compute_nllh(x, oprob::ODEProblem, solver, measurements::DataFrame; abstol = 1e-9,
                      reltol = 1e-9)::Real
    mprey = measurements[measurements[!, :observableId] .== "prey_o", :measurement]
    mpredator = measurements[measurements[!, :observableId] .== "predator_o", :measurement]
    tsave = unique(measurements.time)

    _p = convert.(eltype(x), x)
    _p[1:4] .= x[1:4]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, solver, abstol = abstol, reltol = reltol, saveat = tsave)
    prey, predator = sol[1, :], sol[2, :]

    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        model_output = nn_model([prey[i], predator[i]], x.p_net1, st)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - model_output)^2 / σ^2
    end
    for i in eachindex(mpredator)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - predator[i])^2 / σ^2
    end
    return nllh
end

## Parameter estimation problem setup
measurements = CSV.read(joinpath(@__DIR__, "..", "petab", "measurements.tsv"), DataFrame)
# Objective function
xmech = (α = 1.3, δ = 1.8, β = 0.9, γ = 0.8)
x = ComponentArray(merge(xmech, (p_net1=pnn,)))
# Read neural net parameters, and assign to x
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
# Read neural net parameters, and assign to x
set_ps_net!(x.p_net1, joinpath(@__DIR__, "..", "petab", "net1_ps.hf5"), nn_model)

## Compute model values
_f = (x) -> compute_nllh(x, oprob, Vern9(), measurements; abstol = 1e-12, reltol = 1e-12)
llh = _f(x) .* -1
# High order finite-difference scheme
llh_grad = FiniteDifferences.grad(central_fdm(5, 1), _f, x)[1] .* -1
# Simulated values, order as in measurements
oprob.p .= x[1:4]
sol = solve(oprob, Vern9(), abstol = 1e-9, reltol = 1e-9,
            saveat = unique(measurements.time))
prey_vals = [nn_model(sol[:, i], x.p_net1, st)[1][1] for i in eachindex(sol.t)]
simulated_values = vcat(prey_vals, sol[2, :])

# YAML problem file
solutions = Dict(:llh => llh,
                 :tol_llh => 1e-3,
                 :tol_grad_llh => 1e-1,
                 :tol_simulations => 1e-3,
                 :grad_llh_files => Dict(
                    :mech => "grad_mech.tsv",
                    :net1 => "grad_net1.hf5"),
                 :simulation_files => ["simulations.tsv"])
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
# Simulated values
simulations_df = deepcopy(measurements)
rename!(simulations_df, "measurement" => "simulation")
simulations_df.simulation .= simulated_values
CSV.write(joinpath(@__DIR__, "..", "simulations.tsv"), simulations_df, delim = '\t')
# Gradient values
df_mech = DataFrame(parameterId = ["alpha", "delta", "beta", "gamma"],
                    value = llh_grad[1:4])
CSV.write(joinpath(@__DIR__, "..", "grad_mech.tsv"), df_mech, delim = '\t')
nn_ps_to_h5(nn_model, llh_grad.p_net1, joinpath(@__DIR__, "..", "grad_net1.hf5"))

# Write problem yaml
mapping_table = DataFrame(
    Dict("petabEntityId" => ["net1.input1", "net1.input2", "net1.output1"],
         "modelEntityId" => ["prey", "predator", "net1_output1"]))
CSV.write(joinpath(@__DIR__, "..", "petab", "mapping_table.tsv"), mapping_table; delim = '\t')
problem_yaml = Dict(
    :format_version => 2,
    :parameter_file => "parameters_ude.tsv",
    :problems => [Dict(
        :condition_files => ["conditions.tsv"],
        :measurement_files => ["measurements.tsv"],
        :observable_files => ["observables.tsv"],
        :model_files => Dict(
            :language => "sbml",
            :location => "lv.xml"),
        :mapping_files => ["mapping_table.tsv"])],
    :extensions => Dict(
        :petab_sciml => Dict(
            :net1 => Dict(
                :file => "net1.yaml",
                :parameters => "net1_ps.h5",
                :hybridization => Dict(
                    :input => "ode",
                    :output => "observable")))))
YAML.write_file(joinpath(@__DIR__, "..", "petab", "problem_ude.yaml"), problem_yaml)
