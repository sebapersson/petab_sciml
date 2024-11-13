nn_model = @compact(
    layer1 = Conv((5, 5), 1 => 2; cross_correlation = true),
    layer2 = Conv((2, 5), 2 => 1; cross_correlation = true),
) do x
    embed = layer1(x)
    out = layer2(embed)
    @return out
end

input_order_jl = ["W", "H", "C"]
input_order_py = ["C", "H", "W"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 15, 10, 1, 1)
    output = nn_model(input, ps, st)[1]
    save_ps(joinpath(@__DIR__, ".."), i, nn_model, ps)
    save_input(joinpath(@__DIR__, ".."), i, input[:, :, :, 1], input_order_jl, input_order_py)
    df_output = _array_to_tidy(output[:, :, :, 1];  mapping = [1 => 3, 2 => 2, 3 => 1])
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py)
