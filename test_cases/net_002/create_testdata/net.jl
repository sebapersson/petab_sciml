nn_model = @compact(
    layer1 = Dense(2, 5),
    layer2 = Dense(2, 10),
    layer3 = Bilinear((5, 10) => 2)
) do x
    x1 = layer1(x)
    x2 = layer2(x)
    out = layer3((x1, x2))
    @return out
end

input_order_jl = ["W"]
input_order_py = ["W"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 2)
    output = vec(nn_model(input, ps, st)[1])
    save_ps(joinpath(@__DIR__, ".."), i, nn_model, ps)
    save_input(joinpath(@__DIR__, ".."), i, input, input_order_jl, input_order_py)
    df_output = _array_to_tidy(output)
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py)
