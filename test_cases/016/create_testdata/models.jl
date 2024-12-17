using Lux, ComponentArrays, OrdinaryDiffEq, CSV, DataFrames
using Catalyst: @unpack
import Random
rng = Random.default_rng()

# A Lux.jl Neural-Network model
nn_model = @compact(
    layer1 = Conv((5, 5), 3 => 1; cross_correlation=true),
    layer2 = FlattenLayer(),
    layer3 = Dense(36 => 1, Lux.relu)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end |> f64
_pnn, _st = Lux.setup(rng, nn_model)
pnn = _pnn |> f64
const st = _st
nndict = Dict(:net1 => (st, nn_model))

function lv16!(du, u, p, t)
    prey, predator = u
    @unpack α, δ, β, γ = p

    du[1] = α*prey - β * prey * predator # prey
    du[2] = γ*prey*predator - δ*predator # predator
    return nothing
end

# ODEProblem for model
u0 = [0.44249296, 4.6280594]
p_mechanistic = (α = 1.3, δ = 1.8, β = 0.9, γ = 0.8) |> ComponentArray
oprob = ODEProblem(lv16!, u0, (0.0, 10.0), p_mechanistic)

# For mapping the input
order_jl, order_py = ["W", "H", "C"], ["C", "H", "W"]
imap = zeros(Int64, length(order_jl))
for i in eachindex(order_py)
    imap[i] = findfirst(x -> x == order_py[i], order_jl)
end
map_input = collect(1:length(order_py)) .=> imap
