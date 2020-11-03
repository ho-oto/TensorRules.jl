using ChainRulesCore
using ChainRulesTestUtils
using LinearAlgebra
using Random
using TensorOperations
using TensorRules
using Test

rng = MersenneTwister(1234321)

@testset "Theory" begin
    test1(a, b, c, d) = @tensor _[A, C] := a * conj(b[A, B]) * c[B, C] + d[A, C]
    function ChainRulesCore.rrule(::typeof(test1), a, b, c, d)
        f = test1(a, b, c, d)
        function pullback(f̄)
            @tensor ā[] := conj(f̄[A, C]) * conj(b[A, B]) * c[B, C]
            ā = first(ā)
            @tensor b̄[A, B] := a * conj(f̄[A, C]) * c[B, C]
            b̄ = conj(b̄)
            @tensor c̄[B, C] := a * conj(b[A, B]) * conj(f̄[A, C])
            @tensor d̄[A, C] := conj(f̄[A, C])
            return (NO_FIELDS, conj(ā), conj(b̄), conj(c̄), conj(d̄))
        end
        return f, pullback
    end
    for T in (ComplexF64, Float64)
        f = randn(rng, T, 3, 4)
        a, ā = randn(rng, T), randn(rng, T)
        b, b̄ = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
        c, c̄ = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
        d, d̄ = randn(rng, T, 3, 4), randn(rng, T, 3, 4)
        rrule_test(test1, f, (a, ā), (b, b̄), (c, c̄), (d, d̄))
    end

    test2(a, b) = @tensor _[A, D] := a[A, E, B, B, E, C] * b[C, D]
    function ChainRulesCore.rrule(::typeof(test2), a, b)
        f = test2(a, b)
        function pullback(f̄)
            dE, dB = size(a, 2), size(a, 3)
            T = eltype(a)
            δE, δB = Array{T}(I, dE, dE), Array{T}(I, dB, dB)
            @tensor ā[A, E, B, B', E', C] :=
                conj(f̄[A, D]) * b[C, D] * δE[E, E'] * δB[B, B']
            @tensor b̄[C, D] := a[A, E, B, B, E, C] * conj(f̄[A, D])
            return (NO_FIELDS, conj(ā), conj(b̄))
        end
        return f, pullback
    end
    for T in (ComplexF64, Float64)
        f = randn(rng, T, 4, 5)
        a, ā = randn(rng, T, 4, 2, 1, 1, 2, 3), randn(rng, T, 4, 2, 1, 1, 2, 3)
        b, b̄ = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
        rrule_test(test2, f, (a, ā), (b, b̄))
        a, ā = randn(rng, T, 4, 2, 3, 3, 2, 3), randn(rng, T, 4, 2, 3, 3, 2, 3)
        rrule_test(test2, f, (a, ā), (b, b̄))
    end

    test3(a, b) = @tensor _[] := a[A, B] * b[B, A]
    function ChainRulesCore.rrule(::typeof(test3), a, b)
        f = test3(a, b)
        function pullback(f̄)
            @tensor ā[A, B] := conj(first(f̄)) * b[B, A]
            @tensor b̄[B, A] := a[A, B] * conj(first(f̄))
            return (NO_FIELDS, conj(ā), conj(b̄))
        end
        return f, pullback
    end
    for T in (ComplexF64, Float64)
        f = randn(rng, T, 1)
        a, ā = randn(rng, T, 5, 3), randn(rng, T, 5, 3)
        b, b̄ = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
        rrule_test(test3, f, (a, ā), (b, b̄))
    end
end

@testset "RHS parse" begin
    ex = quote
        (A[1:end, :][1, 2][a, b'] + sin(cos(B))[b', a'']) * (C*D+E)[a'', a] * 3 * 2 +
        α * C[a, a] * K[1, 2, 3][a, a] -
        (((L[a, b] * P[b, c]) * (M[c, d] * N[d, e]) * Z[e, f]) * D[f, a]) * π
    end
    @test TensorRules._rhs_to_args(ex)[2] == [
        :(A[1:end, :][1, 2]),
        :(sin(cos(B))),
        :(C * D + E),
        :α,
        :C,
        :(K[1, 2, 3]),
        :L,
        :P,
        :M,
        :N,
        :Z,
        :D,
        :π,
    ]
    @test TensorRules.make_only_product(
        :(
            -(
                2 * (
                    (
                        v[V] +
                        (x[X] - (d[A, B] + (a[A, B] * b[A, B] + c[A, B]) * α + β[A])) +
                        y +
                        z
                    ) - w[W]
                )
            ) * conj(k[K])
        ),
        :a,
    ) == :(-(2 * -((a[A, B] * b[A, B]) * α)) * conj(k[K]))
end
