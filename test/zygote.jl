@testset "with Zygote" begin
    k = 1

    function foo_(a, b, c, d, e, f)
        @tensoropt !C x[A, B] :=
            k * conj(a[A, C]) * (-sin.(b)[C, D] * c.d[D, B] - c.d[C, B])
        @tensor x[A, B] -= d * conj(e[1])[A, B'', B', B', B'', B]
        x = x + f * k
        @tensor x[A, B] += (@tensor _[A, C] := a[A, B] * a[B, C])[A, C] * (a * a)[C, B]
        return x
    end

    @∇ function foo(a, b, c, d, e, f)
        @tensoropt !C x[a, b] :=
            k * conj(a[a, c]) * (-sin.(b)[c, d] * c.d[d, b] - c.d[c, b])
        @tensor x[A, B] -= d * conj(e[1])[A, B'', B', B', B'', B]
        x = x + f * k
        @tensor x[A, B] += (@tensor _[A, C] := a[A, B] * a[B, C])[A, C] * (a * a)[C, B]
        return x
    end

    rng = MersenneTwister(1234321)
    T = ComplexF64
    a = randn(rng, T, 3, 3)
    b = randn(rng, T, 3, 3)
    c = (d=randn(rng, T, 3, 3),)
    d = randn(rng, T)
    e = [randn(rng, T, 3, 2, 1, 1, 2, 3) for i in 1:2]
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
