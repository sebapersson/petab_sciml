#=
    Hard-coded likelihood and simulated values for test case 001
=#

using FiniteDifferences, YAML
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

function compute_nllh(x, oprob::ODEProblem, solver, measurements::DataFrame; abstol = 1e-9,
                      reltol = 1e-9)::Real
    nllh, σ = 0.0, 0.05
    for (i, cond) in pairs(["cond1", "cond2"])
        mcond = measurements[measurements[!, :simulationConditionId] .== cond, :]
        mprey = mcond[mcond[!, :observableId] .== "prey_o", :measurement]
        mpredator = mcond[mcond[!, :observableId] .== "predator_o", :measurement]
        tsave = unique(mcond.time)
        if cond == "cond1"
            nnout = nn_model(input_data1, x.p_net1, st)[1]
        else
            nnout = nn_model(input_data2, x.p_net1, st)[1]
        end
        _p = convert.(eltype(x), ComponentArray(oprob.p))
        _p[1:3] .= x[1:3]
        _p[4] = nnout[1]
        _oprob = remake(oprob, p = _p)
        sol = solve(_oprob, solver, abstol = abstol, reltol = reltol, saveat = tsave)
        prey, predator = sol[1, :], sol[2, :]

        for j in eachindex(mprey)
            nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[j] - prey[j])^2 / σ^2
        end
        for j in eachindex(mpredator)
            nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[j] - predator[j])^2 / σ^2
        end
    end
    return nllh
end

## Parameter estimation problem setup
measurements = CSV.read(joinpath(@__DIR__, "..", "petab", "measurements.tsv"), DataFrame)
# Objective function
xmech = (α = 1.3, δ = 1.8, β = 0.9)
x = ComponentArray(merge(xmech, (p_net1=pnn,)))
# Read neural net parameters, and assign to x
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
# Read neural net parameters, and assign to x
set_ps_net!(x.p_net1, joinpath(@__DIR__, "..", "petab", "net1_ps.hf5"), nn_model)
# Neural-net input data
# Neural-net input data
__input_data1 = h5read(joinpath(@__DIR__, "..", "petab", "input_data1.hf5"), "input1")
_input_data1 = permutedims(__input_data1, reverse(1:ndims(__input_data1)))
_input_data1 = _reshape_array(_input_data1, map_input)
input_data1 = reshape(_input_data1, (size(_input_data1)..., 1))
__input_data2 = h5read(joinpath(@__DIR__, "..", "petab", "input_data2.hf5"), "input1")
_input_data2 = permutedims(__input_data2, reverse(1:ndims(__input_data2)))
_input_data2 = _reshape_array(_input_data2, map_input)
input_data2 = reshape(_input_data2, (size(_input_data2)..., 1))

## Compute model values
_f = (x) -> compute_nllh(x, oprob, Vern9(), measurements; abstol = 1e-12, reltol = 1e-12)
llh = _f(x) .* -1
# High order finite-difference scheme
llh_grad = FiniteDifferences.grad(central_fdm(5, 1), _f, x)[1] .* -1
# Simulated values, order as in measurements
_p = (α = 1.3, δ = 1.8, β = 0.9, γ = nn_model(input_data1, x.p_net1, st)[1][1])
_oprob = remake(oprob, p = _p)
sol = solve(_oprob, Vern9(), abstol = 1e-9, reltol = 1e-9,
            saveat = unique(measurements.time))
simulated_values1 = vcat(sol[1, :], sol[2, :])
_p = (α = 1.3, δ = 1.8, β = 0.9, γ = nn_model(input_data2, x.p_net1, st)[1][1])
_oprob = remake(oprob, p = _p)
sol = solve(_oprob, Vern9(), abstol = 1e-9, reltol = 1e-9,
            saveat = unique(measurements.time))
simulated_values2 = vcat(sol[1, :], sol[2, :])
simulated_values = vcat(simulated_values1, simulated_values2)

## Write test values
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
df_mech = DataFrame(parameterId = ["alpha", "delta", "beta"],
                    value = llh_grad[1:3])
CSV.write(joinpath(@__DIR__, "..", "grad_mech.tsv"), df_mech, delim = '\t')
nn_ps_to_h5(nn_model, llh_grad.p_net1, joinpath(@__DIR__, "..", "grad_net1.hf5"))

# Write problem yaml
mapping_table = DataFrame(
    Dict("petab.MODEL_ENTITY_ID" => ["net1.input1", "net1.output1"],
         "petab.PETAB_ENTITY_ID" => ["net1_input1", "gamma"]))
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
                    :input => "pre_ode",
                    :output => "pre_ode")))))
YAML.write_file(joinpath(@__DIR__, "..", "petab", "problem_ude.yaml"), problem_yaml)
