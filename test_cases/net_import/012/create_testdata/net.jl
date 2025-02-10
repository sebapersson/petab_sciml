nn_model = @compact(
    flatten1 = FlattenLayer(),
) do x
    out = flatten1(x)
    @return out
end

input_order_jl, input_order_py = ["W", "H", "D", "C", "N"], ["N", "C", "D", "H", "W"]
output_order_jl, output_order_py = ["W", "N"], ["N", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, Float32, 5, 4, 3, 2, 1)
    output = nn_model(input, ps, st)[1]
    save_io(dirsave, i, input, input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output, output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py; ps = false)
