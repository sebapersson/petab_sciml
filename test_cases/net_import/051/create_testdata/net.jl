nn_model = @compact(
    norm1 = LayerNorm((20, ); affine = true),
    layer1 = Dense(20 => 5)
) do x
    embed = norm1(x)
    out = layer1(embed)
    @return out
end

input_order_jl, input_order_py = ["W", "N"], ["N", "W"]
output_order_jl, output_order_py = ["W", "N"], ["N", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, Float32, 20, 1)
    output = nn_model(input, ps, st)[1]
    save_ps(dirsave, i, nn_model, ps)
    save_io(dirsave, i, input, input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output, output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py)
