include(joinpath(@__DIR__, "models.jl"))

Random.seed!(123)
sol = solve(oprob_simulate, Vern9(), abstol = 1e-9, reltol = 1e-9, saveat = 1:1:10)
df1 = DataFrame(observableId = "prey",
                simulationConditionId = "cond1",
                measurement = sol[1, :] + randn(10) .* 0.05,
                time = sol.t)
df2 = DataFrame(observableId = "predator",
                simulationConditionId = "cond1",
                measurement = sol[2, :] + randn(10) .* 0.05,
                time = sol.t)
df = vcat(df1, df2)
CSV.write(joinpath(@__DIR__, "..", "petab", "measurements.tsv"), df, delim = '\t')
