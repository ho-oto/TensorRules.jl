using ChainRulesTestUtils
using FiniteDifferences
using Random
using TensorRules
using Test
using Zygote

Zygote.refresh()

TensorRules.showexpr[] = true

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
