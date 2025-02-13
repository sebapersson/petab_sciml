#=
    Hard-coded likelihood and simulated values for test case 001
=#

using FiniteDifferences, YAML
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

function compute_nllh(x, oprob::ODEProblem, solver, measurements::DataFrame; abstol = 1e-9,
                      reltol = 1e-9)::Real
    mprey = measurements[measurements[!, :observableId] .== "prey_o", :measurement]
    mpredator = measurements[measurements[!, :observableId] .== "predator_o", :measurement]
    tsave = unique(measurements.time)

    nnout2 = nn_model2([2.0, 2.0], x.p_net2, st2)[1]
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p[1:2] .= x[1:2]
    _p.β = nnout2[1]
    _p.p_net1 .= x.p_net1

    _oprob = remake(oprob, p = _p)
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
xmech = (α = 1.3, δ = 1.8)
x = ComponentArray(merge(xmech, (p_net1=pnn1, p_net2 = pnn2)))
# Read neural net parameters, and assign to x
set_ps_net!(x.p_net1, joinpath(@__DIR__, "..", "petab", "net1_ps.hdf5"), nn_model1)
set_ps_net!(x.p_net2, joinpath(@__DIR__, "..", "petab", "net2_ps.hdf5"), nn_model2)

## Compute model values
_f = (x) -> compute_nllh(x, oprob_nn, Vern9(), measurements; abstol = 1e-12, reltol = 1e-12)
llh = _f(x) .* -1
# High order finite-difference scheme
llh_grad = FiniteDifferences.grad(central_fdm(5, 1), _f, x)[1] .* -1
# Simulated values, order as in measurements
_pmech = (α = 1.3, δ = 1.8, β = nn_model2([2.0, 2.0], x.p_net2, st2)[1][1])
_p = ComponentArray(merge(_pmech, (p_net1 = x.p_net1, )))
_oprob = remake(oprob_nn, p = _p)
sol = solve(_oprob, Vern9(), abstol = 1e-9, reltol = 1e-9,
            saveat = unique(measurements.time))
simulated_values = vcat(sol[1, :], sol[2, :])

## Write test values
# YAML problem file
solutions = Dict(:llh => llh,
                 :tol_llh => 1e-3,
                 :tol_grad_llh => 1e-1,
                 :tol_simulations => 1e-3,
                 :grad_llh_files => Dict(
                    :mech => "grad_mech.tsv",
                    :net1 => "grad_net1.hdf5",
                    :net2 => "grad_net2.hdf5"),
                 :simulation_files => ["simulations.tsv"])
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
# Simulated values
simulations_df = deepcopy(measurements)
rename!(simulations_df, "measurement" => "simulation")
simulations_df.simulation .= simulated_values
CSV.write(joinpath(@__DIR__, "..", "simulations.tsv"), simulations_df, delim = '\t')
# Gradient values
df_mech = DataFrame(parameterId = ["alpha", "delta"],
                    value = llh_grad[1:2])
CSV.write(joinpath(@__DIR__, "..", "grad_mech.tsv"), df_mech, delim = '\t')
nn_ps_to_h5(nn_model1, llh_grad.p_net1, joinpath(@__DIR__, "..", "grad_net1.hdf5"))
nn_ps_to_h5(nn_model2, llh_grad.p_net2, joinpath(@__DIR__, "..", "grad_net2.hdf5"))

# Write problem yaml
mapping_table = DataFrame(petabEntityId = ["prey", "predator", "gamma",
                                           "input1", "input2", "beta"],
                          modelEntityId = ["net1.input1", "net1.input2", "net1.output1",
                                           "net2.input1", "net2.input2", "net2.output1"])
CSV.write(joinpath(@__DIR__, "..", "petab", "mapping_table.tsv"), mapping_table; delim = '\t')
problem_yaml = Dict(
    :format_version => 2,
    :parameter_file => "parameters_ude.tsv",
    :problems => [Dict(
        :condition_files => ["conditions.tsv"],
        :measurement_files => ["measurements.tsv"],
        :observable_files => ["observables.tsv"],
        :model_files => Dict(
            :lv_ude => Dict(
                :language => "sbml",
                :location => "lv.xml")),
        :mapping_files => ["mapping_table.tsv"])],
    :extensions => Dict(
        :petab_sciml => Dict(
            :net1 => Dict(
                :file => "net1.yaml",
                :parameters => "net1_ps.hdf5",
                :hybridization => Dict(
                    :input => "ode",
                    :output => "ode")),
            :net2 => Dict(
                :file => "net2.yaml",
                :parameters => "net2_ps.hdf5",
                :hybridization => Dict(
                    :input => "pre_ode",
                    :output => "pre_de")))))
YAML.write_file(joinpath(@__DIR__, "..", "petab", "problem_ude.yaml"), problem_yaml)
