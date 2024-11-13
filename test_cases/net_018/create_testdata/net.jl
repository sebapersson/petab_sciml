
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

input_order_jl, input_order_py = ["W", "H", "C", "N"], ["N", "C", "H", "W"]
output_order_jl, output_order_py = ["W", "N"], ["N", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 20, 20, 1, 1)
    output = nn_model(input, ps, st)[1]
    save_ps(dirsave, i, nn_model, ps)
    save_io(dirsave, i, input, input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output, output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py)
