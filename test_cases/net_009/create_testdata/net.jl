nn_model = @compact(
    layer1 = MaxPool((1, 2, 3))
) do x
    out = layer1(x)
    @return out
end

input_order_jl, input_order_py = ["W", "H", "D", "C"], ["C", "D", "H", "W"]
output_order_jl, output_order_py = ["W", "H", "D", "C"], ["C", "D", "H", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 6, 5, 4, 1, 1)
    output = nn_model(input, ps, st)[1]
    save_io(dirsave, i, input[:, :, :, :, 1], input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output[:, :, :, :, 1], output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py; ps = false)
