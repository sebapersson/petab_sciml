using Lux, CSV, DataFrames, StableRNGs, YAML

include(joinpath(@__DIR__, "..", "..", "helper.jl"))

# A Lux.jl Neural-Network model
nn_model = @compact(
    layer1 = Dense(2, 5, Lux.leakyrelu),
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
solutions = Dict(:net_file => "net.yaml",
                 :net_ps => ["net_ps_1.tsv", "net_ps_2.tsv", "net_ps_3.tsv"],
                 :net_input => ["net_input_1.tsv", "net_input_2.tsv", "net_input_3.tsv"],
                 :net_output => ["net_output_1.tsv", "net_output_2.tsv", "net_output_3.tsv"])
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
