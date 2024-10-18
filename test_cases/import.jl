using SBMLImporter, Catalyst, ModelingToolkit

path_xml = joinpath(@__DIR__, "analytic_solution", "Test_model2.xml")
prn, cb = load_SBML(path_xml)

sys = convert(ODESystem, prn.rn) |> structural_simplify
equations(sys)
Differential(t)(prey(t)) ~ (alpha*prey(t)) / default + (-beta*prey(t)*predator(t)) / default
Differential(t)(predator(t)) ~ (delta*predator(t)) / default


function lv_true!(du, u, p, t)
    prey, predator = u
    @unpack α, β, γ, δ, = p

    du[1] = α*prey - β*prey*predator # prey
    du[2] = γ*prey*predator - δ*predator # predator
    return nothing
end

using OrdinaryDiffEq, Plots
u0 = [0.44249296, 4.6280594]
ptrue = (α = 1.3, β = 0.9, γ = 0.8, δ = 1.8)
lv_prob = ODEProblem(lv_true!, u0, (0.0, 10.0), ptrue)
sol = solve(lv_prob, Vern9(), abstol = 1e-9, reltol = 1e-9, saveat = 1:10)
prey_obs = sol[1, :] + randn(10) .* 0.1
predator_obs = sol[2, :] + randn(10) .* 0.1
plot(sol)

using DataFrames, CSV
df1 = DataFrame(observableId = "prey",
                simulationConditionId = "cond1",
                measurement = prey_obs,
                time = sol.t)
df2 = DataFrame(observableId = "predator",
                simulationConditionId = "cond1",
                measurement = predator_obs,
                time = sol.t)
df_save = vcat(df1, df2)
CSV.write(joinpath(@__DIR__, "petab/measurements.tsv"), df_save, delim = '\t')

using DataFrames, CSV
condtions = DataFrame(conditionId = ["cond1", "cond2"],
                      input1 = [10.0, "input1_cond1"],
                      input2 = [20.0, "input2_cond2"])
CSV.write(joinpath(@__DIR__, "case4/conditions.tsv"), condtions, delim = '\t')
