#=
    Helper functions for writing neural-network parameters to file
=#

using Lux, ComponentArrays, DataFrames
using Catalyst: @unpack

function nn_ps_to_tidy(nn, ps::Union{ComponentArray, NamedTuple}, netname::Symbol)::DataFrame
    df_ps = DataFrame()
    for (layername, layer) in pairs(nn.layers)
        ps_layer = ps[layername]
        df_layer = layer_ps_to_tidy(layer, ps_layer, netname, layername)
        df_ps = vcat(df_ps, df_layer)
    end
    return df_ps
end

function set_ps_net!(ps::ComponentArray, df_ps::DataFrame, netname::Symbol, nn)::Nothing
    df_net = df_ps[startswith.( df_ps[!, :parameterId], "$(netname)_"), :]
    for (id, layer) in pairs(nn.layers)
        df_layer = df_net[startswith.(df_net[!, :parameterId], "$(netname)_$(id)_"), :]
        ps_layer = ps[id]
        set_ps_layer!(ps_layer, layer, df_layer)
        ps[id] = ps_layer
    end
    return nothing
end

"""
    layer_ps_to_tidy(layer::Lux.Dense, ps, netname, layername)::DataFrame

Transforms parameters (`ps`) for a Lux layer to a tidy DataFrame `df` with columns `value`
and `parameterId`.

A `Lux.Dense` layer has two sets of parameters, weights and optionally biases. The
weight matrix `W` has dimensions `size(W) = (out_dims, in_dims)`, and if present, the bias
vector `B` has dimensions `size(B) = out_dims`. Thus, in `df` the column `parameterId`
has values:
- `netname_layername_weight_i_j`: weight for output `i` and input `j`
- `netname_layername_bias_i`: bias for output `i`
"""
function layer_ps_to_tidy(layer::Lux.Dense, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack in_dims, out_dims, use_bias = layer
    weight_names = fill("", in_dims * out_dims)
    for i in 1:out_dims
        for j in 1:in_dims
            ix = out_dims * (j - 1) + i
            weight_names[ix] = "weight_$(i)_$(j)"
        end
    end
    df_weight = DataFrame(parameterId = "$(netname)_$(layername)_" .* weight_names,
                          value = vec(ps.weight))
    if use_bias == true
        bias_names = "bias_" .* string.(1:out_dims)
        df_bias = DataFrame(parameterId = "$(netname)_$(layername)_" .* bias_names,
                            value = ps.bias)
    else
        df_bias = DataFrame()
    end
    return vcat(df_weight, df_bias)
end

function set_ps_layer!(ps::ComponentArray, layer::Lux.Dense, df_ps::DataFrame)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    @assert size(ps.weight) == (out_dims, in_dims) "layer size does not match ps.weight size"
    if use_bias == true
        @assert size(ps.bias) == (out_dims, ) "layer size does not match ps.bias size"
    end

    df_weights = df_ps[occursin.("weight_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_weights[!, :parameterId])
        j, k = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", id))[(end-1):end])
        ps.weight[j, k] = df_weights[i, :value]
    end

    use_bias == false && return nothing
    df_bias = df_ps[occursin.("bias_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_bias[!, :parameterId])
        j = parse(Int64, collect(m.match for m in eachmatch(r"\d+", id))[end])
        ps.bias[j] = df_bias[i, :value]
    end
    return nothing
end
