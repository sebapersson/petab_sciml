using Lux, CSV, DataFrames, StableRNGs, YAML

include(joinpath(@__DIR__, "..", "..", "helper.jl"))

# A Lux.jl Neural-Network model
nn_model = @compact(
    layer1 = LPPool((3, 2, 1); p = 2)
) do x
    out = layer1(x)
    @return out
end

for i in 1:3
    rng = StableRNG(i)
    ps, st = Lux.setup(rng, nn_model)
    input = rand(rng, 6, 5, 4, 1, 1)
    output = nn_model(input, ps, st)[1]
    df_input = _array_to_tidy(input[:, :, :, :, 1]; mapping = [1 => 4, 2 => 1, 3 => 2, 4 => 3])
    df_output = _array_to_tidy(output[:, :, :, :, 1];  mapping = [1 => 4, 2 => 1, 3 => 2, 4 => 3])
    CSV.write(joinpath(@__DIR__, "..", "net_input_$i.tsv"), df_input, delim = '\t')
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
solutions = Dict(:net_file => "net.yaml",
                 :net_ps => ["net_ps_1.tsv", "net_ps_2.tsv", "net_ps_3.tsv"],
                 :net_input => ["net_input_1.tsv", "net_input_2.tsv", "net_input_3.tsv"],
                 :net_output => ["net_output_1.tsv", "net_output_2.tsv", "net_output_3.tsv"])
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
