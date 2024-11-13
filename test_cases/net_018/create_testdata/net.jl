
nn_model = @compact(
    conv1 = Conv((5, 5), 1 => 6; cross_correlation = true),
    conv2 = Conv((5, 5), 6 => 16; cross_correlation = true),
    max_pool1 = MaxPool((2, 2)),
    fc1 = Dense(64, 120),
    fc2 = Dense(120, 84),
    fc3 = Dense(84, 10),
    flatten1 = FlattenLayer()
) do x
    c1 = conv1(x)
    s2 = max_pool1(c1)
    c3 = conv2(s2)
    s4 = max_pool1(c3)
    s4 = flatten1(s4)
    f5 = fc1(s4)
    f6 = fc2(f5)
    output = fc3(f6)
    @return output
end

input_order_jl = ["W", "H", "C", "N"]
input_order_py = ["N", "C", "H", "W"]
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 20, 20, 1, 1)
    output = nn_model(input, ps, st)[1]
    save_ps(joinpath(@__DIR__, ".."), i, nn_model, ps)
    save_input(joinpath(@__DIR__, ".."), i, input, input_order_jl, input_order_py)
    df_output = _array_to_tidy(output;  mapping = [1 => 2, 2 => 1])
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."), input_order_jl, input_order_py)
