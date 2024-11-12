nn_model = @compact(
    flatten1 = FlattenRowMajor(flatten_all = true),
) do x
    out = flatten1(x)
    @return out
end

input_order_jl = ["W", "H"]
input_order_py = ["W", "H"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 4, 3)
    output = nn_model(input, ps, st)[1]
    save_input(joinpath(@__DIR__, ".."), i, input, input_order_jl, input_order_py)
    df_output = _array_to_tidy(output)
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py; ps = false)
