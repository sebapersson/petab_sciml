#=
    Helper functions for writing neural-network parameters to file
=#

using Lux, ComponentArrays, DataFrames, Random, HDF5
using Catalyst: @unpack

function write_yaml(dirsave, input_order_jl, input_order_py, output_order_jl, output_order_py; ps::Bool=true, dropout::Bool=false)::Nothing
    solutions = Dict(
        :net_file => "net.yaml",
        :net_input => ["net_input_1.h5", "net_input_2.h5", "net_input_3.h5"],
        :net_output => ["net_output_1.h5", "net_output_2.h5", "net_output_3.h5"],
        :input_order_jl => input_order_jl,
        :input_order_py => input_order_py,
        :output_order_jl => output_order_jl,
        :output_order_py => output_order_py)
    if ps
        solutions[:net_ps] = ["net_ps_1.h5", "net_ps_2.h5", "net_ps_3.h5"]
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

function set_ps_net!(ps::ComponentArray, path_h5::String, nn)::Nothing
    file = h5open(path_h5, "r")
    for (layerid, layer) in pairs(nn.layers)
        ps_layer = ps[layerid]
        isempty(ps_layer) && continue
        _set_ps_layer!(ps_layer, layer, file[string(layerid)])
        ps[layerid] = ps_layer
    end
    close(file)
    return nothing
end

function layer_ps_to_h5!(file, layer::Lux.Dense, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
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
function layer_ps_to_h5!(file, layer::Lux.Bilinear, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack in1_dims, in2_dims, out_dims, use_bias = layer
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps)
    _ps_bias_to_h5!(g, ps, use_bias)
    return nothing
end
function layer_ps_to_h5!(file, layer::Union{Lux.BatchNorm, Lux.InstanceNorm}, ps::Union{NamedTuple, ComponentArray}, layername::Symbol)::Nothing
    @unpack affine, chs = layer
    affine == false && return nothing
    g = create_group(file, string(layername))
    _ps_weight_to_h5!(g, ps; scale = true)
    _ps_bias_to_h5!(g, ps, true)
    return nothing
end
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
function layer_ps_to_h5!(file, layer::Union{Lux.MaxPool, Lux.MeanPool, Lux.LPPool, Lux.AdaptiveMaxPool, Lux.AdaptiveMeanPool, Lux.FlattenLayer, Lux.Dropout, Lux.AlphaDropout}, ::Union{NamedTuple, ComponentArray, Vector{<:AbstractFloat}}, ::Symbol)::Nothing
    return nothing
end

function _set_ps_layer!(ps::ComponentArray, layer::Lux.Dense, file_group)::Nothing
    @unpack in_dims, out_dims, use_bias = layer
    @assert size(ps.weight) == (out_dims, in_dims) "Error in dimension of weights for Dense layer"
    ps_weight = _get_ps_layer(file_group, :weight)
    @views ps.weight .= ps_weight

    use_bias == false && return nothing
    @assert size(ps.bias) == (out_dims, ) "Error in dimension of bias for Dense layer"
    ps_bias = _get_ps_layer(file_group, :bias)
    @views ps.bias .= ps_bias
    return nothing
end
function _set_ps_layer!(ps::ComponentArray, layer::Lux.Conv, file_group)::Nothing
    @unpack kernel_size, use_bias, in_chs, out_chs = layer
    @assert size(ps.weight) == (kernel_size..., in_chs, out_chs) "Error in dimension of weights for Conv layer"
    _ps_weight = _get_ps_layer(file_group, :weight)
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
    ps_bias = _get_ps_layer(file_group, :bias)
    @views ps.bias .= ps_bias
    return nothing
end

function _get_ps_layer(file_group, which::Symbol)
    if which == :weight
        ps = HDF5.read_dataset(file_group, "weight")
    else
        ps = HDF5.read_dataset(file_group, "bias")
    end
    # Julia is column-major, while the standard format is row-major
    return permutedims(ps, reverse(1:ndims(ps)))
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
