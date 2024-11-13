nn_model = @compact(
    layer1 = Conv((2, 2), 5 => 1; cross_correlation = true),
    drop = Dropout(0.5; dims = 3)
) do x
    x1 = drop(x)
    out = layer1(x1)
    @return out
end

input_order_jl, input_order_py = ["W", "H", "C"], ["C", "H", "W"]
output_order_jl, output_order_py = ["W", "H", "C"], ["C", "H", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 4, 4, 5, 1)
    output = zeros(3, 3, 1, 1)
    for i in 1:40000
        _output, st = nn_model(input, ps, st)
        output += _output
    end
    output ./= 40000
    save_ps(dirsave, i, nn_model, ps)
    save_io(dirsave, i, input[:, :, :, 1], input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output[:, :, :, 1], output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py; dropout = true)
