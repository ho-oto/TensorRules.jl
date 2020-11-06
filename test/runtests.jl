using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences
using LinearAlgebra
using MacroTools
using Random
using TensorOperations
using TensorRules
using Test
using Zygote

Zygote.refresh()

@testset "Theory" begin
    rng = MersenneTwister(1234321)
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

    t_einsum =
        @∇genedfunc 1 a(b, c, d) = @tensor a[B, A] := conj(b[A, C]) * c[C, D] * d[B, D]
    t_opt1 = @∇genedfunc 1 a(b, c, d) =
        @tensoropt (A => 1, C => χ) a[B, A] := b[A, C] * c[C, D] * d[B, D]
    t_opt2 =
        @∇genedfunc 1 a(b, c, d) = @tensoropt !(A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]
    t_opt3 =
        @∇genedfunc 1 a(b, c, d) = @tensoropt (A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]
    t_rsca = @∇genedfunc 1 a(α, b, c) = @tensor a[B, A] := α * conj(b[A, C]) * c[C, B]
    t_add = @∇genedfunc 1 a(α, b, c, β, bb, cc, ccc) = @tensor a[B, A] :=
        -α * conj(b[A, C]) * c[C, B] + β * bb[A, C] * (-cc[C, B] + 2 * ccc[C, B])
    t_tr1 = @∇genedfunc 1 a(b, c) = @tensor a[C] := b[A, B, B', B', B] * c[A, A', A', C]
    t_tr2 = @∇genedfunc 1 a(b, c) = @tensor a[C] := b[A, B, BB, BB, B] * c[A, AA, AA, C]
    t_lsca = @∇genedfunc 1 a(b, c, d) = @tensor a[] := b[A, B] * c[B, C] * d[C, A]

    rng = MersenneTwister(1234321)

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


        rrule_test(t_einsum, a, (b, Δb), (c, Δc), (d, Δd))
        rrule_test(t_opt1, a, (b, Δb), (c, Δc), (d, Δd))
        rrule_test(t_opt2, a, (b, Δb), (c, Δc), (d, Δd))
        rrule_test(t_opt3, a, (b, Δb), (c, Δc), (d, Δd))
        rrule_test(t_rsca, a, (α, Δα), (b, Δb), (c, Δc))
        rrule_test(
            t_add,
            a,
            (α, Δα),
            (b, Δb),
            (c, Δc),
            (β, Δβ),
            (b2, Δb2),
            (c2, Δc2),
            (c3, Δc3),
        )

        a = randn(rng, T, 2)
        b, Δb = randn(rng, T, 3, 2, 3, 3, 2), randn(rng, T, 3, 2, 3, 3, 2)
        c, Δc = randn(rng, T, 3, 3, 3, 2), randn(rng, T, 3, 3, 3, 2)

        rrule_test(t_tr1, a, (b, Δb), (c, Δc))
        rrule_test(t_tr2, a, (b, Δb), (c, Δc))

        a = randn(rng, T, 1)
        b, Δb = randn(rng, T, 3, 2), randn(rng, T, 3, 2)
        c, Δc = randn(rng, T, 2, 4), randn(rng, T, 2, 4)
        d, Δd = randn(rng, T, 4, 3), randn(rng, T, 4, 3)

        rrule_test(t_lsca, a, (b, Δb), (c, Δc), (d, Δd))
    end
end

@testset "with Zygote" begin
    k = 1

    function foo_(a, b, c, d, e, f)
        @tensoropt !C x[A, B] :=
            k * conj(a[A, C]) * (-sin.(b)[C, D] * c.d[D, B] - c.d[C, B])
        @tensor x[A, B] -= d * conj(e[1])[A, B'', B', B', B'', B]
        x = x + f * k
        @tensor x[A, B] += (@tensor _[A, C] := a[A, B] * a[B, C])[A, C] * (a*a)[C, B]
        return x
    end

    @∇ function foo(a, b, c, d, e, f)
        @tensoropt !C x[A, B] :=
            k * conj(a[A, C]) * (-sin.(b)[C, D] * c.d[D, B] - c.d[C, B])
        @tensor x[A, B] -= d * conj(e[1])[A, B'', B', B', B'', B]
        x = x + f * k
        @tensor x[A, B] += (@tensor _[A, C] := a[A, B] * a[B, C])[A, C] * (a*a)[C, B]
        return x
    end

    rng = MersenneTwister(1234321)
    T = ComplexF64
    a = randn(rng, T, 3, 3)
    b = randn(rng, T, 3, 3)
    c = (d = randn(rng, T, 3, 3),)
    d = randn(rng, T)
    e = [randn(rng, T, 3, 2, 1, 1, 2, 3) for i = 1:2]
    f = randn(rng, T, 3, 3)

    @test foo(a, b, c, d, e, f) ≈ foo_(a, b, c, d, e, f)

    for F in [angle ∘ first, angle ∘ sum, real ∘ sum]
        ad = gradient((a, b, c, d, e, f) -> F(foo(a, b, c, d, e, f)), (a, b, c, d, e, f)...)
        fd = grad(
            central_fdm(5, 1),
            (a, b, c, d, e, f) -> F(foo(a, b, c, d, e, f)),
            (a, b, c, d, e, f)...,
        )

        @test ad[1] ≈ fd[1] atol = cbrt(eps())
        @test ad[2] ≈ fd[2] atol = cbrt(eps())
        @test ad[3].d ≈ fd[3].d atol = cbrt(eps())
        @test ad[4] ≈ fd[4] atol = cbrt(eps())
        @test ad[5][1] ≈ fd[5][1] atol = cbrt(eps())
        @test ad[6] ≈ fd[6] atol = cbrt(eps())
    end
end
