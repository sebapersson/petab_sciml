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

A `Lux.Dense` layer has two sets of parameters, `weight` and optionally `bias`. For
`Dense` and all other form of layers `weight` and `bias` are stored as:

- `netname_layername_weight_ix`: weight for output `i` and input `j`
- `netname_layername_bias_ix`: bias for output `i`

Where `ix` depends on the `Tensor` the parameters are stored in. For example, if
`size(weight) = (5, 2)` `ix` is on the form `ix = i_j`. Here, it is important to note that
the PEtab standard uses Julia tensor. For example, `x = ones(5, 3, 2)` can be thought of
as a Tensor with height 5, width 3 and depth 2. In PyTorch `x` would correspond to
`x = torch.ones(2, 5, 3)`.

For `Dense` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(out_features, in_features)`
- `bias` of dimension `(out_features)`
"""
function layer_ps_to_tidy(layer::Lux.Dense, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack in_dims, out_dims, use_bias = layer
    df_weight = _ps_weight_to_tidy(ps, (out_dims, in_dims), netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_dims, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `ConvTranspose` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(kernel_size, in_channels, out_channels)`
- `bias` of dimension `(out_channels)`

Note that `kernel_size` for `ConvTranspose` and `Conv` layers can be a `Tuple`, for example
for `Torch` `conv2d` it is a two-dimensional tuple.
"""
function layer_ps_to_tidy(layer::Lux.ConvTranspose, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    df_weight = _ps_weight_to_tidy(ps, (kernel_size..., out_chs, in_chs), netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `Conv` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(kernel_size, out_channels, in_channels)`
- `bias` of dimension `(out_channels)`
"""
function layer_ps_to_tidy(layer::Lux.Conv, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    df_weight = _ps_weight_to_tidy(ps, (kernel_size..., in_chs, out_chs), netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Union{Lux.MaxPool}, ...)::DataFrame

Pooling layers do not have parameters.
"""
function layer_ps_to_tidy(layer::Union{Lux.MaxPool},::Union{NamedTuple, ComponentArray}, ::Symbol, ::Symbol)::DataFrame
    return DataFrame()
end


function set_ps_layer!(ps::ComponentArray, layer::Lux.ConvTranspose, df_ps::DataFrame)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    _set_ps_weight!(ps, (kernel_size..., out_chs, in_chs), df_ps)
    _set_ps_bias!(ps, (out_chs, ), df_ps, use_bias)
    return nothing
end
function set_ps_layer!(ps::ComponentArray, layer::Lux.Conv, df_ps::DataFrame)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    _set_ps_weight!(ps, (kernel_size..., in_chs, out_chs), df_ps)
    _set_ps_bias!(ps, (out_chs, ), df_ps, use_bias)
    return nothing
end
function set_ps_layer!(ps::ComponentArray, layer::Lux.Dense, df_ps::DataFrame)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    _set_ps_weight!(ps, (out_dims, in_dims), df_ps)
    _set_ps_bias!(ps, (out_dims, ), df_ps, use_bias)
    return nothing
end
function set_ps_layer!(::ComponentArray, ::Union{Lux.MaxPool}, ::DataFrame)::Nothing
    return nothing
end

function _set_ps_weight!(ps::ComponentArray, weight_dims, df_ps::DataFrame)::Nothing
    @assert size(ps.weight) == weight_dims "layer size does not match ps.weight size"
    df_weights = df_ps[occursin.("weight_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_weights[!, :parameterId])
        ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", id))[(end-length(weight_dims)+1):end])
        ps.weight[ix...] = df_weights[i, :value]
    end
    return nothing
end

function _set_ps_bias!(ps::ComponentArray, bias_dims, df_ps::DataFrame, use_bias)::Nothing
    use_bias == false && return nothing
    @assert size(ps.bias) == bias_dims "layer size does not match ps.bias size"
    df_bias = df_ps[occursin.("bias_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_bias[!, :parameterId])
        ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", id))[(end-length(bias_dims)+1):end])
        ps.bias[ix...] = df_bias[i, :value]
    end
    return nothing
end

function _ps_weight_to_tidy(ps, weight_dims, netname::Symbol, layername::Symbol)::DataFrame
    iweight = getfield.(findall(x -> true, ones(weight_dims)), :I)
    weight_names =  ["weight" * prod("_" .* string.(ix)) for ix in iweight]
    df_weight = DataFrame(parameterId = "$(netname)_$(layername)_" .* weight_names,
                          value = vec(ps.weight))
    return df_weight
end

function _ps_bias_to_tidy(ps, bias_dims, netname::Symbol, layername::Symbol, use_bias)::DataFrame
    use_bias == false && return DataFrame()
    if length(bias_dims) > 1
        ibias = getfield.(findall(x -> true, ones(bias_dims)), :I)
    else
        ibias = 1:bias_dims[1]
    end
    bias_names =  ["bias" * prod("_" .* string.(ix)) for ix in ibias]
    df_bias = DataFrame(parameterId = "$(netname)_$(layername)_" .* bias_names,
                        value = vec(ps.bias))
    return df_bias
end

function _array_to_tidy(x::Array)::DataFrame
    dims = size(x)
    if length(dims) == 1
        ix = 1:dims[1]
    else
        ix = getfield.(findall(y -> true, ones(dims)), :I)
    end
    ix_names =  [prod(string.(i) .* ";")[1:end-1] for i in ix]
    dfx = DataFrame(value = vec(x),
                    ix = ix_names)
    return dfx
end
