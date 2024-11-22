using CSV, DataFrames

## Net1
# Reuse the net from test case 001 (as it takes ages to train)
ps1_df = CSV.read(joinpath(@__DIR__, "..", "..", "001", "petab", "parameters_nn.tsv"), DataFrame;
                  stringtype = String)

## Net2
# Reuse net2 from case 010
_ps2_df = CSV.read(joinpath(@__DIR__, "..", "..", "010", "petab", "parameters_nn.tsv"), DataFrame;
                   stringtype = String)
ps2_df = _ps2_df[52:end, :]

ps_df = vcat(ps1_df, ps2_df)
CSV.write(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), ps_df, delim = '\t')
