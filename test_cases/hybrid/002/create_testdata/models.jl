using Lux, ComponentArrays, OrdinaryDiffEq, CSV, DataFrames
using Catalyst: @unpack
import Random
rng = Random.default_rng()

# A Lux.jl Neural-Network model
nn_model = @compact(
    layer1 = Dense(2, 5, Lux.tanh),
    layer2 = Dense(5, 5, Lux.tanh),
    layer3 = Dense(5, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end
pnn, _st = Lux.setup(rng, nn_model)
const st = _st
nndict = Dict(:net1 => (st, nn_model))

# ODE-model with neural network
function _lv!(du, u, p, t, nn)
    prey, predator = u
    @unpack α, β, p_net1 = p

    st1, nn1 = nn[:net1]
    du_nn = nn1([prey, predator], p_net1, st1)[1]

    du[1] = α*prey - β * prey * predator # prey
    du[2] = du_nn[1] # predator
    return nothing
end
lv! = let nn = nndict
    (du, u, p, t) -> _lv!(du, u, p, t, nn)
end

# True model
function lv_simulate!(du, u, p, t)
    prey, predator = u
    @unpack α, δ, β, γ = p

    du[1] = α*prey - β * prey * predator # prey
    du[2] = γ*prey*predator - δ*predator # predator
    return nothing
end

# ODEProblem for the different models
# Simulate
u0 = [0.44249296, 4.6280594]
p_mechanistic = (α = 1.3, δ = 1.8, β = 0.9, γ = 0.8)
oprob_simulate = ODEProblem(lv_simulate!, u0, (0.0, 10.0), p_mechanistic)
# UDE
p_mechanistic = (α = 1.3, β = 0.9)
p_ode = ComponentArray(merge(p_mechanistic, (p_net1=pnn,)))
oprob_nn = ODEProblem(lv!, u0, (0.0, 10.0), p_ode)
