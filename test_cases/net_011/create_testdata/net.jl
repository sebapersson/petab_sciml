nn_model = @compact(
    flatten1 = FlattenLayer(),
) do x
    out = flatten1(x)
    @return out
end

input_order_jl = ["W", "H", "C", "N"]
input_order_py = ["N", "C", "H", "W"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 5, 4, 3, 1)
    output = nn_model(input, ps, st)[1]
    save_input(joinpath(@__DIR__, ".."), i, input, input_order_jl, input_order_py)
    df_output = _array_to_tidy(output; mapping = [1 => 2, 2 => 1])
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py; ps = false)
