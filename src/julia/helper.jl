#=
    Helper functions for writing neural-network parameters to file
=#

using Lux, ComponentArrays, DataFrames, Random
using Catalyst: @unpack

function write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py; ps::Bool=true, dropout::Bool=false)::Nothing
    solutions = Dict(
        :net_file => "net.yaml",
        :net_input => ["net_input_1.tsv", "net_input_2.tsv", "net_input_3.tsv"],
        :net_output => ["net_output_1.tsv", "net_output_2.tsv", "net_output_3.tsv"],
        :input_order_jl => input_order_jl,
        :input_order_py => input_order_py,
        :output_order_jl => output_order_jl,
        :output_order_py => output_order_py)
    if ps
        solutions[:net_ps] = ["net_ps_1.tsv", "net_ps_2.tsv", "net_ps_3.tsv"]
    end
    if dropout
        solutions[:dropout] = 40000
    end
    YAML.write_file(joinpath(dirsave, "solutions.yaml"), solutions)
    return nothing
end

function save_io(dirsave, i::Integer, input, order_jl, order_py, iotype::Symbol)::Nothing
    @assert length(order_jl) == length(order_py) "Length of input format vectors do not match"
    if order_jl == order_py
        df_input = _array_to_tidy(input)
    else
        imap = zeros(Int64, length(order_jl))
        for i in eachindex(order_py)
            imap[i] = findfirst(x -> x == order_py[i], order_jl)
        end
        map = collect(1:length(order_py)) .=> imap
        df_input = _array_to_tidy(input; mapping = map)
    end
    if iotype == :input
        CSV.write(joinpath(dirsave, "net_input_$i.tsv"), df_input, delim = "\t")
    elseif iotype == :output
        CSV.write(joinpath(dirsave, "net_output_$i.tsv"), df_input, delim = "\t")
    end
    return nothing
end

function save_ps(dirsave, i::Integer, nn_model, ps)::Nothing
    df_ps = nn_ps_to_tidy(nn_model, ps, :net)
    CSV.write(joinpath(dirsave, "net_ps_$i.tsv"), df_ps, delim = '\t')
    return nothing
end

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
    df_weight = _ps_weight_to_tidy(ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_dims, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `Conv` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, in_channels, out_channels)`. This is fixed
    by the importer.
"""
function layer_ps_to_tidy(layer::Lux.Conv, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if length(kernel_size) == 1
        _psweigth = _reshape_array(ps.weight, [1 => 3, 2 => 2, 3 => 1])
    elseif length(kernel_size) == 2
        #=
            Julia (Lux.jl) and PyTorch encode images differently, and thus the W-matrix:
            In PyTorch: (in_chs, out_chs, H, W)
            In Julia  : (W, H, in_chs, out_chs)
            Thus, except acounting for tensor encoding, kernel dimension is flipped
        =#
        _psweigth = _reshape_array(ps.weight, [1 => 4, 2 => 3, 3 => 2, 4 => 1])
    elseif length(kernel_size) == 3
        #=
            Julia (Lux.jl) and PyTorch encode 3d-images differently, and thus the W-matrix:
            In PyTorch: (in_chs, out_chs, D, H, W)
            In Julia  : (W, H, D, in_chs, out_chs)
            Thus, except acounting for tensor encoding, kernel dimension is flipped
        =#
        _psweigth = _reshape_array(ps.weight, [1 => 5, 2 => 4, 3 => 3, 4 => 2, 5 => 1])
    end
    _ps = ComponentArray(weight = _psweigth)
    df_weight = _ps_weight_to_tidy(_ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.ConvTranspose, ...)::DataFrame

For `ConvTranspose` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, out_channels, in_channels)`. This is fixed
    by the importer.
"""
function layer_ps_to_tidy(layer::Lux.ConvTranspose, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    if length(kernel_size) == 1
        _psweigth = _reshape_array(ps.weight, [1 => 3, 2 => 2, 3 => 1])
    elseif length(kernel_size) == 2
        # For the mapping, see comment above on image format in Lux.Conv
        _psweigth = _reshape_array(ps.weight, [1 => 4, 2 => 3, 3 => 2, 4 => 1])
    elseif length(kernel_size) == 3
        # See comment on Lux.Conv
        _psweigth = _reshape_array(ps.weight, [1 => 5, 2 => 4, 3 => 3, 4 => 2, 5 => 1])
    end
    _ps = ComponentArray(weight = _psweigth)
    df_weight = _ps_weight_to_tidy(_ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_chs, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Lux.Bilinear, ...)::DataFrame

For `Bilinear` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(out_features, in_features1, in_features2)`
- `bias` of dimension `(out_features)`
"""
function layer_ps_to_tidy(layer::Lux.Bilinear, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    df_weight = _ps_weight_to_tidy(ps, netname, layername)
    df_bias = _ps_bias_to_tidy(ps, (out_dims, ), netname, layername, use_bias)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ...)::DataFrame

For `BatchNorm` and `InstanceNorm` layer possible parameters that are saved to a DataFrame
are:
- `scale/weight` of dimension `(num_features)`
- `bias` of dimension `(num_features)`
!!! note
    in Lux.jl the dimension argument `num_features` is chs (number of input channels)
"""
function layer_ps_to_tidy(layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ps::Union{NamedTuple, ComponentArray}, netname::Symbol, layername::Symbol)::DataFrame
    @unpack affine, chs = layer
    affine == false && return DataFrame()
    df_weight = _ps_weight_to_tidy(ps, netname, layername; scale = true)
    df_bias = _ps_bias_to_tidy(ps, (chs, ), netname, layername, true)
    return vcat(df_weight, df_bias)
end
"""
    layer_ps_to_tidy(layer::Union{Lux.MaxPool, Lux.FlattenLayer}, ...)::DataFrame

Pooling layers do not have parameters.
"""
function layer_ps_to_tidy(layer::Union{Lux.MaxPool, Lux.MeanPool, Lux.LPPool, Lux.AdaptiveMaxPool, Lux.AdaptiveMeanPool, Lux.FlattenLayer, Lux.Dropout, Lux.AlphaDropout}, ::Union{NamedTuple, ComponentArray}, ::Symbol, ::Symbol)::DataFrame
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
function set_ps_layer!(ps::ComponentArray, layer::Lux.Bilinear, df_ps::DataFrame)::Nothing
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    _set_ps_weight!(ps, (out_dims, in1_dims, in2_dims), df_ps)
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

function _ps_weight_to_tidy(ps, netname::Symbol, layername::Symbol; scale::Bool = false)::DataFrame
    # For Batchnorm in Lux.jl the weight layer is refered to as scale.
    if scale == false
        ps_weight = ps.weight
    else
        ps_weight = ps.scale
    end
    if length(size(ps_weight)) > 1
        iweight = getfield.(findall(x -> true, ones(size(ps_weight))), :I)
    else
        iweight = 1:length(ps_weight)
    end
    iweight = [iw .- 1 for iw in iweight]
    weight_names =  ["weight" * prod("_" .* string.(ix)) for ix in iweight]
    df_weight = DataFrame(parameterId = "$(netname)_$(layername)_" .* weight_names,
                          value = vec(ps_weight))
    return df_weight
end

function _ps_bias_to_tidy(ps, bias_dims, netname::Symbol, layername::Symbol, use_bias)::DataFrame
    use_bias == false && return DataFrame()
    if length(bias_dims) > 1
        ibias = getfield.(findall(x -> true, ones(bias_dims)), :I)
    else
        ibias = (0:bias_dims[1]-1)
    end
    bias_names =  ["bias" * prod("_" .* string.(ix)) for ix in ibias]
    df_bias = DataFrame(parameterId = "$(netname)_$(layername)_" .* bias_names,
                        value = vec(ps.bias))
    return df_bias
end

function _array_to_tidy(xin::Array; mapping = nothing)::DataFrame
    if isnothing(mapping)
        x = xin
    else
        x = _reshape_array(xin, mapping)
    end
    dims = size(x)
    if length(dims) == 1
        ix = 1:dims[1]
    else
        ix = getfield.(findall(y -> true, ones(dims)), :I)
    end
    ix_names = [prod(string.(i .- 1) .* ";")[1:end-1] for i in ix]
    dfx = DataFrame(value = vec(x),
                    ix = ix_names)
    return dfx
end

function _reshape_array(x, mapping)
    dims_out = size(x)[last.(mapping)]
    xout = reshape(deepcopy(x), dims_out)
    for i in eachindex(Base.CartesianIndices(x))
        inew = zeros(Int64, length(i.I))
        for j in eachindex(i.I)
            inew[j] = i.I[mapping[j].second]
        end
        xout[inew...] = x[i]
    end
    return xout
end
