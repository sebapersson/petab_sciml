nn_model = @compact(
    layer1 = Conv((5, 5), 1 => 2; cross_correlation = true),
    layer2 = Conv((2, 5), 2 => 1; cross_correlation = true),
) do x
    embed = layer1(x)
    out = layer2(embed)
    @return out
end

input_order_jl, input_order_py = ["W", "H", "C"], ["C", "H", "W"]
output_order_jl, output_order_py = ["W", "H", "C"], ["C", "H", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, Float32, 15, 10, 1, 1)
    output = nn_model(input, ps, st)[1]
    save_ps(dirsave, i, nn_model, ps)
    save_io(dirsave, i, input[:, :, :, 1], input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output[:, :, :, 1], output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py)
