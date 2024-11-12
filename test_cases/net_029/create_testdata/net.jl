nn_model = @compact(
    layer1 = Dense(2, 5, Lux.gelu),
    layer2 = Dense(5, 1),
) do x
    embed = layer1(x)
    out = layer2(embed)
    @return out
end

for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 2)
    output = nn_model(input, ps, st)[1]
    df_ps = nn_ps_to_tidy(nn_model, ps, :net)
    df_input = _array_to_tidy(input)
    df_output = _array_to_tidy(output)
    CSV.write(joinpath(@__DIR__, "..", "net_ps_$i.tsv"), df_ps, delim = '\t')
    CSV.write(joinpath(@__DIR__, "..", "net_input_$i.tsv"), df_input, delim = '\t')
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
write_yaml(joinpath(@__DIR__, ".."))
