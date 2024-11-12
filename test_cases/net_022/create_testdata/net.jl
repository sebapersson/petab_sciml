using Lux, CSV, DataFrames, StableRNGs, YAML

include(joinpath(@__DIR__, "..", "..", "helper.jl"))

# A Lux.jl Neural-Network model
nn_model = @compact(
    layer1 = Conv((2, 2), 5 => 1; cross_correlation = true),
    drop = Dropout(0.5; dims = 3)
) do x
    x1 = drop(x)
    out = layer1(x1)
    @return out
end

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
    df_ps = nn_ps_to_tidy(nn_model, ps, :net)
    # PyTorch does not need the batch
    df_input = _array_to_tidy(input[:, :, :, 1]; mapping = [1 => 3, 2 => 1, 3 => 2])
    df_output = _array_to_tidy(output[:, :, :, 1];  mapping = [1 => 3, 2 => 1, 3 => 2])
    CSV.write(joinpath(@__DIR__, "..", "net_ps_$i.tsv"), df_ps, delim = '\t')
    CSV.write(joinpath(@__DIR__, "..", "net_input_$i.tsv"), df_input, delim = '\t')
    CSV.write(joinpath(@__DIR__, "..", "net_output_$i.tsv"), df_output, delim = '\t')
end
solutions = Dict(:net_file => "net.yaml",
                 :net_ps => ["net_ps_1.tsv", "net_ps_2.tsv", "net_ps_3.tsv"],
                 :net_input => ["net_input_1.tsv", "net_input_2.tsv", "net_input_3.tsv"],
                 :net_output => ["net_output_1.tsv", "net_output_2.tsv", "net_output_3.tsv"],
                 :dropout => 40000)
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
