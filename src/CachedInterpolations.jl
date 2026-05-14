module CachedInterpolations

using Interpolations: Interpolations, AbstractInterpolation, BSpline, InPlace, NoInterp, OnCell, Quadratic
using Interpolations: weightedindexes, value_weights, gradient_weights, InterpGetindex
using StaticArrays: StaticArrays, SVector
import Base: getindex

export CachedInterpolation, cachedinterpolators

"""
CachedInterpolations implements a performance enhancement for
quadratic interpolation of a large multidimensional array.  The first
`N` dimensions are interpolating, and the remainder are tiling
(`NoInterp`), so that one is computing interpolants like
```
    for i_2 = 1:size(P, 4), i_1 = 1:size(P, 3)
        y_1 = x_1[i_1, i_2]
        y_2 = x_2[i_1, i_2]
        B[i_1, i_2] = P[y_1, y_2, i_1, i_2]
    end
```
where `x_1`, `x_2` are floating-point indexes and `i_1`, `i_2` are
integer indexes.  A `CachedInterpolation` simulates an array-of-arrays
interface for this task, while in reality using only a single
underlying `P` array.

The performance enhancement comes from caching: when `P` is bigger
than the computer's memory, `P` may be a memory-mapped file, and direct
access to values of `P` will therefore be limited by disk I/O. If one
has a task in which all elements of `B` have to be evaluated
repeatedly for different values of `y_1`, `y_2`, but that these values
often change by only a small amount from one iteration to the next
(e.g., in a descent-based optimization task), then often the
interpolation will be computed from the same underlying entries in `P`
and just use different interpolation coefficients.  Since the
interpolation is quadratic, it is therefore sufficient to cache a
3-by-3-by-... view of the interpolating dimensions of `P`, each
centered on the current `y_1, y_2`.

Create an array-of-interpolating arrays with the function
`cachedinterpolators`.
"""
CachedInterpolations

"""
    CachedInterpolation

A single interpolating tile over a tiling array. A `CachedInterpolation`
represents a "movable" 3×3×... view of `P[:, :, i_1, i_2]` for a specific
`(i_1, i_2)` pair. Calling `itp(y_1, y_2)` returns the scalar
quadratic-interpolated value at position `(y_1, y_2)` (floating-point or
integer coordinates), caching the surrounding 3×3×... block of `P` to avoid
redundant I/O when coordinates change by only a small amount between calls.
Integer-index access is also available via `itp[y_1, y_2]`.

Create arrays of `CachedInterpolation`s with [`cachedinterpolators`](@ref).

# Examples
```jldoctest
julia> A = reshape([0.0; 1.0; 0.0], (3, 1));

julia> itps = cachedinterpolators(A, 1);

julia> itp = itps[1];

julia> itp(2.0)   # quadratic-interpolated peak of [0, 1, 0]
0.75

julia> itp(1.5)   # off-center
0.5
```
"""
mutable struct CachedInterpolation{T, N, M, O, K} <: AbstractInterpolation{T, N, BSpline{Quadratic{InPlace}}}
    # Note: M = N+K
    const coefs::Array{T, M}   # tiled array of 3x3x... buffers
    const parent::Array{T, M}  # the overall array (`P` in the documentation above)
    center::NTuple{N, Int}  # rounded (y_1, y_2) of prev. eval for this tile
    const tileindex::CartesianIndex{K}
end

const splitN = Base.IteratorsMD.split
Base.size(itp::CachedInterpolation{T, N}) where {T, N} = splitN(size(itp.parent), Val(N))[1]
Base.size(itp::CachedInterpolation{T, N}, d) where {T, N} = d <= N ? size(itp.parent, d) : 1

Base.axes(itp::CachedInterpolation{T, N, M, O}) where {T, N, M, O} =
    map((x, o) -> x .- o, splitN(axes(itp.parent), Val(N))[1], O)
Base.axes(itp::CachedInterpolation{T, N, M, O}, d) where {T, N, M, O} =
    d <= N ? axes(itp.parent, d) .- O[d] : Base.OneTo(1)

"""
    cachedinterpolators(parent::Array, N)
    cachedinterpolators(parent::Array, N, origin)

Create an `Array` of [`CachedInterpolation`](@ref) objects from the dense array
`parent`, where the first `N` dimensions of `parent` are interpolating and the
remaining dimensions are tiling. `parent` must be a dense `Array`; subtypes of
`AbstractArray` such as `SubArray` or `SharedArray` are not supported.

The expression `parent[y_1, y_2, i_1, i_2]` becomes:

```julia
itp = itps[i_1, i_2]
itp(y_1, y_2)
```

Each `itp` caches a 3×3×... block of `parent` centered on the most-recent
evaluation point, so repeated calls with nearby coordinates avoid redundant
memory or disk I/O (useful when `parent` is a memory-mapped file).

The optional `origin` argument is a tuple of `N` integers (default: all zeros)
that offsets the coordinate system. With a nonzero `origin`, dimension `d` is
addressed as `y_d + origin[d]` in `parent`, shifting the user-visible axes by
`-origin`. For example, setting `origin = (div(size(parent,d)+1, 2) for d in
1:N)` centers the axes at zero when the interpolating dimensions have odd size.

# Examples
```jldoctest
julia> A = reshape([0.0; 1.0; 0.0], (3, 1));

julia> itps = cachedinterpolators(A, 1);

julia> itps[1](2.0)   # quadratic-interpolated peak of [0, 1, 0]
0.75

julia> itps_c = cachedinterpolators(A, 1, (2,));  # origin at index 2

julia> axes(itps_c[1])   # coordinates shifted; 0 now addresses the peak
(-1:1,)

julia> itps_c[1](0.0)    # peak via centered coordinates
0.75
```
"""
function cachedinterpolators(parent::Array{T, M}, N::Integer, origin = ntuple(d -> 0, N)) where {T, M}
    0 <= N <= M || error("N must be between 0 and $M")
    length(origin) == N || throw(DimensionMismatch("length(origin) = $(length(origin)) is inconsistent with $N interpolating dimensions"))
    sz3 = ntuple(d -> d <= N ? 3 : size(parent, d), Val(M))
    buffer = Array{eltype(parent)}(undef, sz3)
    sztiles = size(parent)[(N + 1):end]  # the tiling dimensions of parent
    # use an impossible initial value (post-offset by origin) to
    # ensure the first access will result in a cache miss
    center = ntuple(d -> -1, N)
    return cachedinterpolators(buffer, parent, origin, center, sztiles)
end

# function-barriered to circumvent type-instability in sztiles
@noinline function cachedinterpolators(buffer::Array{T, M}, parent::Array{T, M}, origin::NTuple{N, Int}, center::NTuple{N, Int}, sztiles::NTuple{K, Int}) where {T, N, M, K}
    itps = Array{CachedInterpolation{T, N, M, origin, K}}(undef, sztiles)
    for tileindex in CartesianIndices(sztiles)
        itps[tileindex] = CachedInterpolation{T, N, M, origin, K}(buffer, parent, center, tileindex)
    end
    return itps
end

@inline function (itp::CachedInterpolation{T, N, M, O, K})(xs::Vararg{Number, N}) where {T, N, M, O, K}
    coefs, parent, center, tileindex = itp.coefs, itp.parent, itp.center, itp.tileindex
    ixs = round.(Int, xs)
    fxs = xs .- ixs .+ 2
    newcenter = ixs .+ O
    sz3 = ntuple(d -> 3, Val(N))
    itpinfo = (ntuple(d -> BSpline(Quadratic(InPlace(OnCell()))), Val(N))..., ntuple(d -> NoInterp(), Val(K))...)
    if newcenter != center
        # Copy the relevant portion from parent into buffer
        offset = CartesianIndex(newcenter .- 2)
        for i in CartesianIndices(sz3)
            coefs[i, tileindex] = parent[i + offset, tileindex]
        end
        itp.center = newcenter
    end
    icoefs = InterpGetindex(coefs)
    wis = weightedindexes((value_weights,), itpinfo, axes(coefs), (fxs..., Tuple(tileindex)...))
    return icoefs[wis...]
end

"""
    Interpolations.gradient(itp::CachedInterpolation, ys...)

Return the gradient of `itp` evaluated at coordinates `ys` as an `SVector`.
The cache is updated automatically, so this can be called without a preceding
`itp(ys...)`.
"""
@inline function Interpolations.gradient(itp::CachedInterpolation{T, N, M, O, K}, ys::Vararg{Number, N}) where {T, N, M, O, K}
    coefs, parent, center, tileindex = itp.coefs, itp.parent, itp.center, itp.tileindex
    iys = round.(Int, ys)
    xs = ys .- iys .+ 2
    newcenter = iys .+ O
    sz3 = ntuple(d -> 3, Val(N))
    if newcenter != center
        offset = CartesianIndex(newcenter .- 2)
        for i in CartesianIndices(sz3)
            coefs[i, tileindex] = parent[i + offset, tileindex]
        end
        itp.center = newcenter
    end
    itpinfo = (ntuple(d -> BSpline(Quadratic(InPlace(OnCell()))), Val(N))..., ntuple(d -> NoInterp(), Val(K))...)
    wis = weightedindexes((value_weights, gradient_weights), itpinfo, axes(coefs), (xs..., Tuple(tileindex)...))
    icoefs = InterpGetindex(coefs)
    return SVector(map(inds -> icoefs[inds...], wis))
end

@inline function Interpolations.gradient!(g::AbstractVector, itp::CachedInterpolation{T, N, M, O, K}, ys::Vararg{Number, N}) where {T, N, M, O, K}
    gs = Interpolations.gradient(itp, ys...)
    return copyto!(g, gs)
end

### Potential deprecations

# if AbstractInterpolation <: AbstractArray goes away, this can be deprecated
getindex(itp::CachedInterpolation{T, N, M, O, K}, xs::Vararg{Integer, N}) where {T, N, M, O, K} = itp(xs...)

end # module
