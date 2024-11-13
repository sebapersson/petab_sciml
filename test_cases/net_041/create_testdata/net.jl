input_order_jl, input_order_py = ["W", "H"], ["H", "W"]
output_order_jl, output_order_py = ["W", "H"], ["H", "W"]
dirsave = joinpath(@__DIR__, "..")
for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 5, 4)
    output = Lux.logsoftmax(input)
    save_io(dirsave, i, input, input_order_jl, input_order_py, :input)
    save_io(dirsave, i, output, output_order_jl, output_order_py, :output)
end
write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py; ps = false)
