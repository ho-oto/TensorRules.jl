using ChainRulesCore
using ChainRulesTestUtils
using LinearAlgebra
using MacroTools
using Random
using TensorOperations
using TensorRules
using Test
using Zygote

rng = MersenneTwister(1234321)

@testset "Theory" begin
    test1(a, b, c, d) = @tensor _[A, C] := a * conj(b[A, B]) * c[B, C] + d[A, C]
    function ChainRulesCore.rrule(::typeof(test1), a, b, c, d)
        f = test1(a, b, c, d)
        function pullback(f̄)
            @tensor ā[] := conj(f̄[A, C]) * conj(b[A, B]) * c[B, C]
            ā = first(ā)
            @tensor Δb[A, B] := a * conj(f̄[A, C]) * c[B, C]
            b̄ = conj(Δb)
            @tensor Δc[B, C] := a * conj(b[A, B]) * conj(f̄[A, C])
            @tensor Δd[A, C] := conj(f̄[A, C])
            return (NO_FIELDS, conj(ā), conj(b̄), conj(Δc), conj(Δd))
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
            @tensor Δb[C, D] := a[A, E, B, B, E, C] * conj(f̄[A, D])
            return (NO_FIELDS, conj(ā), conj(Δb))
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
            @tensor Δb[B, A] := a[A, B] * conj(first(f̄))
            return (NO_FIELDS, conj(ā), conj(Δb))
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
        (((L[a, b] * P.P[b, c]) * (M[c, d] * N[d, e]) * Z[e, f]) * D[f, a]) * π
    end
    @test TensorRules.rhs_to_args(ex)[2] == [
        :(A[1:end, :][1, 2]),
        :(sin(cos(B))),
        :(C * D + E),
        :α,
        :C,
        :(K[1, 2, 3]),
        :L,
        :(P.P),
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

    ex = :(-a * (-b + (c + d) + (-e) - f - (g * h)) * i + j)
    @test TensorRules.make_only_product(ex, :a) ==
          :(-a * (-b + (c + d) + (-e) - f - (g * h)) * i)
    @test TensorRules.make_only_product(ex, :b) == :(-a * -b * i)
    @test TensorRules.make_only_product(ex, :c) == :(-a * c * i)
    @test TensorRules.make_only_product(ex, :d) == :(-a * d * i)
    @test TensorRules.make_only_product(ex, :e) == :(-a * -e * i)
    @test TensorRules.make_only_product(ex, :f) == :(-a * -f * i)
    @test TensorRules.make_only_product(ex, :g) == :(-a * -(g * h) * i)
    @test TensorRules.make_only_product(ex, :h) == :(-a * -(g * h) * i)
    @test TensorRules.make_only_product(ex, :i) ==
          :(-a * (-b + (c + d) + (-e) - f - (g * h)) * i)
    @test TensorRules.make_only_product(ex, :j) == :j
end

@testset "gen_rule" begin
    funcname = :test
    T = ComplexF64
    for T in (ComplexF64, Float64)
        a = randn(rng, T, 4, 3)
        b, Δb = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
        b2, Δb2 = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
        c, Δc = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
        c2, Δc2 = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
        c3, Δc3 = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
        d, Δd = randn(rng, T, 4, 4), randn(rng, T, 4, 4)
        α, Δα = randn(rng, T), randn(rng, T)
        β, Δβ = randn(rng, T), randn(rng, T)

        # einsum & conj
        ex = :(@tensor a[B, A] := conj(b[A, C]) * c[C, D] * d[B, D])
        @capture(ex, @tensor lhs_[lhsind__] := rhs_)
        rhsreplace, argsorig, argsdummy, indsall = TensorRules.rhs_to_args(rhs)

        eval(TensorRules.gen_func(funcname, argsdummy, lhsind, rhsreplace, Ref{Expr}()))
        eval(TensorRules.gen_rule(
            funcname,
            argsdummy,
            lhsind,
            rhsreplace,
            indsall,
            Ref{Expr}(),
        ))

        rrule_test(test, a, (b, Δb), (c, Δc), (d, Δd))

        # opt
        for ex in [
            :(@tensoropt (A => 1, C => χ) a[B, A] := b[A, C] * c[C, D] * d[B, D]),
            :(@tensoropt !(A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]),
            :(@tensoropt (A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]),
        ]
            @capture(ex, @tensoropt opt_ lhs_[lhsind__] := rhs_)
            rhsreplace, argsorig, argsdummy, indsall = TensorRules.rhs_to_args(rhs)

            eval(TensorRules.gen_func(funcname, argsdummy, lhsind, rhsreplace, Ref(opt)))
            eval(TensorRules.gen_rule(
                funcname,
                argsdummy,
                lhsind,
                rhsreplace,
                indsall,
                Ref(opt),
            ))

            rrule_test(test, a, (b, Δb), (c, Δc), (d, Δd))
        end

        # scalar in rhs
        ex = :(@tensor a[B, A] := α * conj(b[A, C]) * c[C, B])
        @capture(ex, @tensor lhs_[lhsind__] := rhs_)
        rhsreplace, argsorig, argsdummy, indsall = TensorRules.rhs_to_args(rhs)

        eval(TensorRules.gen_func(funcname, argsdummy, lhsind, rhsreplace, Ref{Expr}()))
        eval(TensorRules.gen_rule(
            funcname,
            argsdummy,
            lhsind,
            rhsreplace,
            indsall,
            Ref{Expr}(),
        ))

        rrule_test(test, a, (α, Δα), (b, Δb), (c, Δc))

        # add
        ex = :(@tensor a[B, A] :=
            -α * conj(b[A, C]) * c[C, B] + β * bb[A, C] * (-cc[C, B] + 2 * ccc[C, B]))
        @capture(ex, @tensor lhs_[lhsind__] := rhs_)
        rhsreplace, argsorig, argsdummy, indsall = TensorRules.rhs_to_args(rhs)

        eval(TensorRules.gen_func(funcname, argsdummy, lhsind, rhsreplace, Ref{Expr}()))
        eval(TensorRules.gen_rule(
            funcname,
            argsdummy,
            lhsind,
            rhsreplace,
            indsall,
            Ref{Expr}(),
        ))

        rrule_test(
            test,
            a,
            (α, Δα),
            (b, Δb),
            (c, Δc),
            (β, Δβ),
            (b2, Δb2),
            (c2, Δc2),
            (c3, Δc3),
        )

        # trace
        a = randn(rng, T, 2)
        b, Δb = randn(rng, T, 3, 2, 3, 3, 2), randn(rng, T, 3, 2, 3, 3, 2)
        c, Δc = randn(rng, T, 3, 3, 3, 2), randn(rng, T, 3, 3, 3, 2)

        for ex in [
            :(@tensor a[C] := b[A, B, B', B', B] * c[A, A', A', C]),
            :(@tensor a[C] := b[A, B, BB, BB, B] * c[A, AA, AA, C]),
        ]
            @capture(ex, @tensor lhs_[lhsind__] := rhs_)
            rhsreplace, argsorig, argsdummy, indsall = TensorRules.rhs_to_args(rhs)

            eval(TensorRules.gen_func(funcname, argsdummy, lhsind, rhsreplace, Ref{Expr}()))
            eval(TensorRules.gen_rule(
                funcname,
                argsdummy,
                lhsind,
                rhsreplace,
                indsall,
                Ref{Expr}(),
            ))

            rrule_test(test, a, (b, Δb), (c, Δc))
        end

        # lhs is scalar
        a = randn(rng, T, 1)
        b, Δb = randn(rng, T, 3, 2), randn(rng, T, 3, 2)
        c, Δc = randn(rng, T, 2, 4), randn(rng, T, 2, 4)
        d, Δd = randn(rng, T, 4, 3), randn(rng, T, 4, 3)

        ex = :(@tensor a[] := b[A, B] * c[B, C] * d[C, A])

        @capture(ex, @tensor lhs_[lhsind__] := rhs_)
        rhsreplace, argsorig, argsdummy, indsall = TensorRules.rhs_to_args(rhs)

        eval(TensorRules.gen_func(funcname, argsdummy, lhsind, rhsreplace, Ref{Expr}()))
        eval(TensorRules.gen_rule(
            funcname,
            argsdummy,
            lhsind,
            rhsreplace,
            indsall,
            Ref{Expr}(),
        ))

        rrule_test(test, a, (b, Δb), (c, Δc), (d, Δd))
    end
end
