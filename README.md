# CachedInterpolations.jl

[![CI](https://github.com/HolyLab/CachedInterpolations.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/CachedInterpolations.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/CachedInterpolations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/CachedInterpolations.jl)
[![version](https://juliahub.com/docs/General/CachedInterpolations/stable/version.svg)](https://juliahub.com/ui/Packages/General/CachedInterpolations)

Performance-optimized quadratic B-spline interpolation for large or memory-mapped arrays.

## The problem

When `P` is a large array stored on disk as a memory-mapped file, evaluating
`P[y1, y2, i1, i2]` at many `(i1, i2)` tile positions with slowly-varying
floating-point coordinates `(y1, y2)` incurs repeated disk I/O — even when the
same 3×3 neighborhood of `P` is accessed on consecutive iterations.

## The solution

`CachedInterpolations` wraps `P` in an array of lightweight
`CachedInterpolation` objects — one per tile `(i1, i2)`. Each object caches
the 3×3×... block of `P` centered on the most-recent evaluation point.
When the interpolating coordinates change only slightly between calls (e.g.,
in a gradient-descent loop), the cache is reused and disk I/O is avoided.

## Installation

```julia
using Pkg
Pkg.add("CachedInterpolations")
```

## Usage

```julia
using CachedInterpolations

# A 3×2 array: two tiles of length 3 each
A = [0.0 0.0; 1.0 2.0; 0.0 0.0]   # size (3, 2), N=1 interpolating dim, 1 tiling dim

itps = cachedinterpolators(A, 1)   # returns a length-2 Array of CachedInterpolation

itps[1](2.0)   # quadratic-interpolated value at position 2.0 in tile 1 → 0.75
itps[2](2.0)   # same position in tile 2 → 1.5
```

Repeated calls near the same coordinate reuse the cached 3-element block and
avoid re-reading from `A` (or from disk, if `A` is memory-mapped).

### Centered coordinate axes

Pass an `origin` tuple to shift the visible axis so that a convenient index
(e.g., the center of the array) maps to zero:

```julia
A = reshape([0.0; 1.0; 0.0], (3, 1))
itps = cachedinterpolators(A, 1, (2,))  # index 2 → coordinate 0

axes(itps[1])    # (-1:1,)
itps[1](0.0)     # 0.75 — the peak, now addressed as 0
itps[1](-0.5)    # off-center
```

### Gradient evaluation

`Interpolations.gradient` works on a `CachedInterpolation` with the same call
signature as on a regular `Interpolations` interpolant:

```julia
using Interpolations
g = Interpolations.gradient(itps[1], 0.0)   # SVector of partial derivatives
```

> **Note:** `gradient` assumes you have already called `itp(y...)` at the same
> coordinate to populate the cache. Calling `gradient` without a prior `itp(y...)`
> at the same location returns incorrect results.

### Working with memory-mapped arrays

```julia
using Mmap

# Memory-map a large on-disk array
open("bigdata.bin") do io
    P = Mmap.mmap(io, Array{Float64, 4}, (256, 256, 100, 50))
    itps = cachedinterpolators(P, 2)   # first 2 dims interpolating, last 2 tiling

    # Evaluate at floating-point coords across all tiles — cache avoids re-reading
    # the same 3×3 block when coords change slowly
    result = [itps[i, j](128.0, 128.0) for i in 1:100, j in 1:50]
end
```
