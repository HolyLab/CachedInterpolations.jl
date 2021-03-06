import CachedInterpolations
using Interpolations, Test

A = reshape([0;1;0], (3,1))
C = CachedInterpolations.cachedinterpolators(A, 1)
@test C[1](2.2) ≈ 3/4-0.2^2
@test C[1](1.7) ≈ 3/4-0.3^2

A = rand(7,7,2,2,3)
Q = BSpline(Quadratic(InPlace(OnCell())))
# Note the next line will modify A, and the modified A will be used for C.
# This is what we want to happen.
Ai = interpolate!(A, (Q, Q, NoInterp(), NoInterp(), NoInterp()))
C = CachedInterpolations.cachedinterpolators(A, 2)
@test size(C) == (2,2,3)
c = C[1,1,1]
@test size(c) == (7,7)
@test size(c,1) == 7
@test size(c,2) == 7
@test size(c,3) == 1
@test @inferred(C[1,2,2](3.2, 4.8)) == Ai(3.2,4.8,1,2,2)
@test C[1,2,2](3.2,4.9) == Ai(3.2,4.9,1,2,2)
@test C[1,2,2](3.2,3.8) == Ai(3.2,3.8,1,2,2)

@test Interpolations.gradient(C[1,2,2], 3.2, 3.8) === Interpolations.gradient(Ai, 3.2, 3.8, 1, 2, 2)

# With origin
C = CachedInterpolations.cachedinterpolators(A, 2, (4,4))
@test C[1,2,2](-0.8,0.8) ≈ Ai(3.2,4.8,1,2,2)
@test C[1,2,2](-0.8,0.9) ≈ Ai(3.2,4.9,1,2,2)
@test C[1,2,2](-0.8,-0.2) ≈ Ai(3.2,3.8,1,2,2)
@test Interpolations.gradient(C[1,2,2], -0.8, -0.2) ≈ Interpolations.gradient(Ai, 3.2, 3.8, 1, 2, 2)

# Check for Float32 with Float64 indexes, since that's the
# default mismatch case
A = rand(Float32,7,7,2,2,3)
Ai = interpolate!(A, (Q, Q, NoInterp(), NoInterp(), NoInterp()))
C = CachedInterpolations.cachedinterpolators(A, 2, (4,4))
@test @inferred(C[1,2,2](-0.8, 0.8)) ≈ Ai(3.2,4.8,1,2,2)
