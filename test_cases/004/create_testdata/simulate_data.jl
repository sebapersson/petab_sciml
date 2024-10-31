include(joinpath(@__DIR__, "models.jl"))

# γ = 0.8 for cond1 and γ = 1.0 for cond2
Random.seed!(123)
_oprob = remake(oprob, p = (α = 1.3, δ = 1.8, β = 0.9, γ = 0.8))
sol = solve(_oprob, Vern9(), abstol = 1e-9, reltol = 1e-9, saveat = 1:1:10)
df1 = DataFrame(observableId = "prey",
                simulationConditionId = "cond1",
                measurement = sol[1, :] + randn(10) .* 0.05,
                time = sol.t)
df2 = DataFrame(observableId = "predator",
                simulationConditionId = "cond1",
                measurement = sol[2, :] + randn(10) .* 0.05,
                time = sol.t)
_oprob = remake(oprob, p = (α = 1.3, δ = 1.8, β = 0.9, γ = 1.0))
sol = solve(_oprob, Vern9(), abstol = 1e-9, reltol = 1e-9, saveat = 1:1:10)
df3 = DataFrame(observableId = "prey",
                simulationConditionId = "cond2",
                measurement = sol[1, :] + randn(10) .* 0.05,
                time = sol.t)
df4 = DataFrame(observableId = "predator",
                simulationConditionId = "cond2",
                measurement = sol[2, :] + randn(10) .* 0.05,
                time = sol.t)
df = vcat(df1, df2, df3, df4)
CSV.write(joinpath(@__DIR__, "..", "petab", "measurements.tsv"), df, delim = '\t')
