nn_model = @compact(
    layer1 = Dense(5 => 50),
    layer2 = Dense(50 => 2),
    norm1 = BatchNorm(50)
) do x
    embed = layer1(x)
    embed = norm1(embed)
    out = layer2(embed)
    @return out
end

# TODO: Issue on Float64 input required
input_order_jl, input_order_py = ["W", "N"], ["N", "W"]
output_order_jl, output_order_py = ["W", "N"], ["N", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, Float32, 5, 2)
    output = nn_model(input, ps, st)[1]
    save_ps(dirsave, i, nn_model, ps)
    save_io(dirsave, i, input, input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output, output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py)
