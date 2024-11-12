using Lux, Random

struct FlattenRowMajor <: LuxCore.AbstractLuxLayer
    flatten_all::Bool
end
function FlattenRowMajor(;flatten_all::Bool=false)
    return FlattenRowMajor(flatten_all)
end

LuxCore.initialparameters(::AbstractRNG, ::FlattenRowMajor) = NamedTuple()

LuxCore.initialstates(::AbstractRNG, ::FlattenRowMajor) = NamedTuple()

LuxCore.parameterlength(::FlattenRowMajor) = 0

LuxCore.statelength(::FlattenRowMajor) = 0

function (f::FlattenRowMajor)(x::AbstractArray{T, N}, _, st::NamedTuple) where {T, N}
    if f.flatten_all == false
        if length(size(x)) == 3
            _x = permutedims(x, (2, 1, 3))
        elseif f.flatten_all == false && length(size(x)) == 4
            _x = permutedims(x, (2, 1, 3, 4))
        else
            throw(ArgumentError("x has dim we cannot handle"))
        end
        return reshape(_x, :, size(_x, N)), st
    end
    if length(size(x)) == 2
        _x = permutedims(x, (2, 1))
    elseif length(size(x)) == 3
        _x = permutedims(x, (2, 1, 3))
    else
        throw(ArgumentError("x has dim we cannot handle"))
    end
    return vec(_x), st
end
