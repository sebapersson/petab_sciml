nn_model = @compact(
    layer1 = Dense(10, 2),
    drop = Dropout(0.5)
) do x
    x1 = drop(x)
    out = layer1(x1)
    @return out
end

for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 10)
    output = zeros(2)
    for i in 1:40000
        _output, st = nn_model(input, ps, st)
        output += _output
    end
    output ./= 40000
    df_ps = nn_ps_to_tidy(nn_model, ps, :net)
    df_input = _array_to_tidy(input)
    df_output = _array_to_tidy(output)
    CSV.write(joinpath(@__DIR__, "..", "net_ps_$i.tsv"), df_ps, delim = '\t')
    CSV.write(joinpath(@__DIR__, "..", "net_input_$i.tsv"), df_input, delim = '\t')
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."); dropout = true)
