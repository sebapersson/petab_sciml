
nn_model = @compact(
    layer1 = LPPool((1, 2, 3); p = 2)
) do x
    out = layer1(x)
    @return out
end

input_order_jl = ["W", "H", "D", "C"]
input_order_py = ["C", "D", "H", "W"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 6, 5, 4, 1, 1)
    output = nn_model(input, ps, st)[1]
    save_input(joinpath(@__DIR__, ".."), i, input[:, :, :, :, 1], input_order_jl, input_order_py)
    df_output = _array_to_tidy(output[:, :, :, :, 1];  mapping = [1 => 4, 2 => 3, 3 => 2, 4 => 1])
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py; ps = false)
