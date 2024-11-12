nn_model = @compact(
    layer1 = Dense(10, 2),
    drop = Dropout(0.5)
) do x
    x1 = drop(x)
    out = layer1(x1)
    @return out
end

input_order_jl = ["W"]
input_order_py = ["W"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 10)
    output = zeros(2)
    for i in 1:50000
        _output, st = nn_model(input, ps, st)
        output += _output
    end
    output ./= 50000
    save_ps(joinpath(@__DIR__, ".."), i, nn_model, ps)
    save_input(joinpath(@__DIR__, ".."), i, input, input_order_jl, input_order_py)
    df_output = _array_to_tidy(output)
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py; dropout = true)
