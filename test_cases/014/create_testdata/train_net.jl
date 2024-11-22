using CSV, DataFrames

## Net1
# Reuse the net from test case 003
ps1_df = CSV.read(joinpath(@__DIR__, "..", "..", "003", "petab", "parameters_nn.tsv"), DataFrame;
                  stringtype = String)

## Net2
# Reuse net2 from case 013
_ps2_df = CSV.read(joinpath(@__DIR__, "..", "..", "013", "petab", "parameters_nn.tsv"), DataFrame;
                   stringtype = String)
ps2_df = _ps2_df[52:end, :]

ps_df = vcat(ps1_df, ps2_df)
CSV.write(joinpath(@__DIR__, "..", "petab", "parameters_nn.tsv"), ps_df, delim = '\t')
