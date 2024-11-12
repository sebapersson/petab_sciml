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

input_order_jl = ["W"]
input_order_py = ["W"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 2)
    output = nn_model(input, ps, st)[1]
    save_ps(joinpath(@__DIR__, ".."), i, nn_model, ps)
    save_input(joinpath(@__DIR__, ".."), i, input, input_order_jl, input_order_py)
    df_output = _array_to_tidy(output)
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py)
