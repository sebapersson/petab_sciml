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
        xsave = input
    else
        imap = zeros(Int64, length(order_jl))
        for i in eachindex(order_py)
            imap[i] = findfirst(x -> x == order_py[i], order_jl)
        end
        map = collect(1:length(order_py)) .=> imap
        xsave = _reshape_array(input, map)
    end
    # Python for which the standard is defined is row-major
    if length(size(xsave)) > 1
        xsave = permutedims(xsave, reverse(1:ndims(xsave)))
    end

    if iotype == :input
        h5open(joinpath(dirsave, "net_input_$i.h5"), "w") do file
            write(file, "input", xsave)
        end
    elseif iotype == :output
        h5open(joinpath(dirsave, "net_output_$i.h5"), "w") do file
            write(file, "output", xsave)
        end
    end
    return nothing
end

function save_ps(dirsave, i::Integer, nn_model, ps)::Nothing
    nn_ps_to_h5(nn_model, ps, joinpath(dirsave, "net_ps_$i.h5"))
    return nothing
end

function nn_ps_to_h5(nn, ps::Union{ComponentArray, NamedTuple}, path::String)::Nothing
    if isfile(path)
        rm(path)
    end
    file = h5open(path, "w")
    for (layername, layer) in pairs(nn.layers)
        layer_ps_to_h5!(file, layer, ps[layername], layername)
    end
    close(file)
    return nothing
end

function set_ps_net!(ps::ComponentArray, df_ps::Nothing, netname::Symbol, nn)::Nothing
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
    layer_ps_to_h5!(file, layer::Lux.Dense, ps::Union{ComponentArray, NamedTuple}, layername::Symbol)::Nothing

Transforms parameters (`ps`) for a Lux layer into hdf5 format.

For `Dense` layer possible parameters that are saved are:
- `weight` of dimension `(out_features, in_features)`
- `bias` of dimension `(out_features)`
"""
function layer_ps_to_h5!(file, layer::Lux.Dense, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    layer_ps_to_h5!(file, layer::Lux.ConvTranspose, ...)::Nothing

For `Conv` layer possible parameters are
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, in_channels, out_channels)`. This is fixed
    by the importer.
"""
function layer_ps_to_h5!(file, layer::Lux.Conv, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
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
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, _ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    layer_ps_to_h5!(file, layer::Lux.ConvTranspose, ...)::Nothing

For `ConvTranspose` layer possible parameters that are saved to a DataFrame are:
- `weight` of dimension `(in_channels, out_channels, kernel_size)`
- `bias` of dimension `(out_channels)`

!!! note
    Note, in Lux.jl `weight` has `(kernel_size, out_channels, in_channels)`. This is fixed
    by the importer.
"""
function layer_ps_to_h5!(file, layer::Lux.ConvTranspose, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
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
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, _ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    layer_ps_to_h5!(file, layer::Lux.Bilinear, ...)::Nothing

For `Bilinear` layer possible parameters that are saved are:
- `weight` of dimension `(out_features, in_features1, in_features2)`
- `bias` of dimension `(out_features)`
"""
function layer_ps_to_h5!(file, layer::Lux.Bilinear, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
"""
    layer_ps_to_h5!(file, layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ...)::Nothing

For `BatchNorm` and `InstanceNorm` layer possible parameters that are saved are:
- `scale/weight` of dimension `(num_features)`
- `bias` of dimension `(num_features)`
!!! note
    in Lux.jl the dimension argument `num_features` is chs (number of input channels)
"""
function layer_ps_to_h5!(file, layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack affine, chs = layer
    affine == false && return nothing
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps; scale = true)
    _ps_bias_to_h5!(g, ps, true)
    return nothing
end
"""
    layer_ps_to_h5!(file, layer::Lux.LayerNorm, ...)::Nothing

For `LayerNorm` layer possible parameters that are saved to a DataFrame are:
- `scale/weight` of `size(input)` dimension
- `bias` of `size(input)` dimension
!!! note
    Input order differs between Lux.jl and PyTorch. Order `["C", "D", "H", "W"]` in
    PyTorch corresponds to `["W", "H", "D", "C"]` in Lux.jl. Basically, regardless of input
    dimension the Lux.jl dimension is the PyTorch dimension reversed.
!!! note
    In Lux.jl the input dimension in `size(input, 1)`.
"""
function layer_ps_to_h5!(file, layer::LayerNorm, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack shape, affine = layer
    affine == false && return DataFrame()
    if length(shape) == 4
        _psweigth = _reshape_array(ps.scale[:, :, :, :, 1], [1 => 4, 2 => 3, 3 => 2, 4 => 1])
        _psbias = _reshape_array(ps.bias[:, :, :, :, 1], [1 => 4, 2 => 3, 3 => 2, 4 => 1])
    elseif length(shape) == 3
        _psweigth = _reshape_array(ps.scale[:, :, :, 1], [1 => 3, 2 => 2, 3 => 1])
        _psbias = _reshape_array(ps.bias[:, :, :, 1], [1 => 3, 2 => 2, 3 => 1])
    elseif length(shape) == 2
        _psweigth = _reshape_array(ps.scale[:, :, 1], [1 => 2, 2 => 1])
        _psbias = _reshape_array(ps.bias[:, :, 1], [1 => 2, 2 => 1])
    elseif length(shape) == 1
        _psweigth = ps.scale[:, 1]
        _psbias = ps.bias[:, 1]
    end
    _ps = ComponentArray(weight = _psweigth, bias = _psbias)
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, _ps)
    _ps_bias_to_h5!(g, _ps, true)
    return nothing
end
"""
    layer_ps_to_h5!(file, layer::Union{Lux.MaxPool, Lux.FlattenLayer}, ...)::Nothing

Pooling layers do not have parameters.
"""
function layer_ps_to_h5!(file, layer::Union{Lux.MaxPool, Lux.MeanPool, Lux.LPPool, Lux.AdaptiveMaxPool, Lux.AdaptiveMeanPool, Lux.FlattenLayer, Lux.Dropout, Lux.AlphaDropout}, ::Union{NamedTuple, ComponentArray, Vector{<:AbstractFloat}}, ::Symbol)::Nothing
    return nothing
end

function set_ps_layer!(ps::ComponentArray, layer::Lux.Dense, df_ps::Nothing)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    _set_ps_weight!(ps, (out_dims, in_dims), df_ps)
    _set_ps_bias!(ps, (out_dims, ), df_ps, use_bias)
    return nothing
end
function set_ps_layer!(ps::ComponentArray, layer::Lux.Conv, df_ps::Nothing)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    @assert size(ps.weight) == (kernel_size..., in_chs, out_chs) "Error in dimension of weights for Conv layer"
    _ps_weight = _get_ps_layer(df_ps, length((kernel_size..., in_chs, out_chs)), :weight)
    if length(kernel_size) == 1
        ps_weight = _reshape_array(_ps_weight, [1 => 3, 2 => 2, 3 => 1])
    elseif length(kernel_size) == 2
        ps_weight = _reshape_array(_ps_weight, [1 => 4, 2 => 3, 3 => 2, 4 => 1])
    elseif length(kernel_size) == 3
        ps_weight = _reshape_array(_ps_weight, [1 => 5, 2 => 4, 3 => 3, 4 => 2, 5 => 1])
    end
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_chs, ) "Error in dimension of bias for Conv layer"
    ps_bias = _get_ps_layer(df_ps, 1, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function set_ps_layer!(::Union{ComponentArray, Vector{<:AbstractFloat}}, ::Union{Lux.MaxPool, Lux.FlattenLayer}, ::Nothing)::Nothing
    return nothing
end

function _set_ps_weight!(ps::ComponentArray, weight_dims, df_ps::Nothing)::Nothing
    @assert size(ps.weight) == weight_dims "layer size does not match ps.weight size"
    df_weights = df_ps[occursin.("weight_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_weights[!, :parameterId])
        _ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", id))[(end-length(weight_dims)+1):end])
        ix = [x + 1 for x in _ix]
        ps.weight[ix...] = df_weights[i, :value]
    end
    return nothing
end

function _set_ps_bias!(ps::ComponentArray, bias_dims, df_ps::Nothing, use_bias)::Nothing
    use_bias == false && return nothing
    @assert size(ps.bias) == bias_dims "layer size does not match ps.bias size"
    df_bias = df_ps[occursin.("bias_", df_ps[!, :parameterId]), :]
    for (i, id) in pairs(df_bias[!, :parameterId])
        _ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", id))[(end-length(bias_dims)+1):end])
        ix = [x + 1 for x in _ix]
        ps.bias[ix...] = df_bias[i, :value]
    end
    return nothing
end

function _ps_weight_to_h5!(g, ps; scale::Bool = false)::Nothing
    # For Batchnorm in Lux.jl the weight layer is referred to as scale.
    if scale == false
        ps_weight = ps.weight
    else
        ps_weight = ps.scale
    end
    # To account for Python (for which the standard is defined) is row-major
    ps_weight = permutedims(ps_weight, reverse(1:ndims(ps_weight)))
    g["weight"] = ps_weight
    return nothing
end

function _ps_bias_to_h5!(g, ps, use_bias)::Nothing
    use_bias == false && return nothing
    if length(size(ps.bias)) > 1
        ps_bias = permutedims(ps.bias, reverse(1:ndims(ps.bias)))
    else
        ps_bias = vec(ps.bias)
    end
    g["bias"] = ps_bias
    return nothing
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

function _get_ps_layer(df_layer::Nothing, lendim::Int64, which::Symbol)
    if which == :weight
        df = df_layer[occursin.("weight_", df_layer[!, :parameterId]), :]
    else
        df = df_layer[occursin.("bias_", df_layer[!, :parameterId]), :]
    end
    ix = Any[]
    for pid in df[!, :parameterId]
        _ix = parse.(Int64, collect(m.match for m in eachmatch(r"\d+", pid))[((end-lendim+1):end)])
        # Python -> Julia indexing
        _ix .+= 1
        push!(ix, Tuple(_ix))
    end
    out = zeros(maximum(ix))
    for i in eachindex(ix)
        out[ix[i]...] = df[i, :value]
    end
    return out
end
