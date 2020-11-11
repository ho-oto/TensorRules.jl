@testset "rules" begin
    rng = MersenneTwister(1234321)

    @testset "einsum" begin
        _esum = @fn∇ 1 a(b, c, d) = @tensor a[B, A] := conj(b[A, C]) * c[C, D] * d[B, D]

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 4, 3)
            b, Δb = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            c, Δc = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            d, Δd = randn(rng, T, 4, 4), randn(rng, T, 4, 4)

            rrule_test(_esum, a, (b, Δb), (c, Δc), (d, Δd))
            frule_test(_esum, (b, Δb), (c, Δc), (d, Δd))
        end
    end

    @testset "tensoropt" begin
        _opt1 = @fn∇ 1 function a(b, c, d)
            @tensoropt (A => 1, C => χ) a[B, A] := b[A, C] * c[C, D] * d[B, D]
        end
        _opt2 = @fn∇ 1 function a(b, c, d)
            @tensoropt !(A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]
        end
        _opt3 = @fn∇ 1 a(b, c, d) = @tensoropt (A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 4, 3)
            b, Δb = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            c, Δc = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            d, Δd = randn(rng, T, 4, 4), randn(rng, T, 4, 4)

            rrule_test(_opt1, a, (b, Δb), (c, Δc), (d, Δd))
            rrule_test(_opt2, a, (b, Δb), (c, Δc), (d, Δd))
            rrule_test(_opt3, a, (b, Δb), (c, Δc), (d, Δd))

            frule_test(_opt1, (b, Δb), (c, Δc), (d, Δd))
            frule_test(_opt2, (b, Δb), (c, Δc), (d, Δd))
            frule_test(_opt3, (b, Δb), (c, Δc), (d, Δd))
        end
    end

    @testset "scalar in rhs" begin
        _scar = @fn∇ 1 a(α, b, c) = @tensor a[B, A] := α * conj(b[A, C]) * c[C, B]

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 4, 3)
            b, Δb = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            c, Δc = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            α, Δα = randn(rng, T), randn(rng, T)

            rrule_test(_scar, a, (α, Δα), (b, Δb), (c, Δc))
            frule_test(_scar, (α, Δα), (b, Δb), (c, Δc))
        end
    end

    @testset "add" begin
        _add = @fn∇ 1 function a(α, b, c, β, d, e, f)
            @tensor a[B, A] :=
                -α * conj(b[A, C]) * c[C, B] + conj(β) * d[A, C] * (-e[C, B] + 2 * f[C, B])
        end

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 4, 3)
            b, Δb = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            d, Δd = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            c, Δc = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            e, Δe = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            f, Δf = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            α, Δα = randn(rng, T), randn(rng, T)
            β, Δβ = randn(rng, T), randn(rng, T)

            args = ((α, Δα), (b, Δb), (c, Δc), (β, Δβ), (d, Δd), (e, Δe), (f, Δf))
            rrule_test(_add, a, args...)
            frule_test(_add, args...)
        end
    end

    @testset "trace" begin
        _tr1 = @fn∇ 1 a(b, c) = @tensor a[C] := b[A, B, B', B', B] * c[A, A', A', C]
        _tr2 = @fn∇ 1 a(b, c) = @tensor a[c] := b[a, b, bb, bb, b] * c[a, aa, aa, c]

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 2)
            b, Δb = randn(rng, T, 3, 2, 3, 3, 2), randn(rng, T, 3, 2, 3, 3, 2)
            c, Δc = randn(rng, T, 3, 3, 3, 2), randn(rng, T, 3, 3, 3, 2)

            rrule_test(_tr1, a, (b, Δb), (c, Δc))
            rrule_test(_tr2, a, (b, Δb), (c, Δc))
            frule_test(_tr1, (b, Δb), (c, Δc))
            frule_test(_tr2, (b, Δb), (c, Δc))
        end
    end

    @testset "lhs is scalar" begin
        _scal = @fn∇ 1 a(b, c, d) = @tensor a[] := b[A, B] * c[B, C] * d[C, A]

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 1)
            b, Δb = randn(rng, T, 3, 2), randn(rng, T, 3, 2)
            c, Δc = randn(rng, T, 2, 4), randn(rng, T, 2, 4)
            d, Δd = randn(rng, T, 4, 3), randn(rng, T, 4, 3)

            rrule_test(_scal, a, (b, Δb), (c, Δc), (d, Δd))
            frule_test(_scal, (b, Δb), (c, Δc), (d, Δd))
        end
    end

    @testset "conj" begin
        _co1 = @fn∇ 1 function a(b, c, d)
            @tensor a[B, A] := conj(b[A, C]) * conj(c[C, D] * d[B, D])
        end
        _co2 = @fn∇ 1 a(b, c, d) = @tensor a[B, A] := conj(b[A, C] * c[C, D] * d[B, D])
        _co3 = @fn∇ 1 function a(b, c, d)
            @tensor a[B, A] := conj(
                1.23 * conj(conj(conj(b[A, C]))) * conj(c[C, D] * d[B, D])
            )
        end

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 4, 3)
            b, Δb = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            c, Δc = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            d, Δd = randn(rng, T, 4, 4), randn(rng, T, 4, 4)
            α, Δα = randn(rng, T), randn(rng, T)
            β, Δβ = randn(rng, T), randn(rng, T)

            rrule_test(_co1, a, (b, Δb), (c, Δc), (d, Δd))
            rrule_test(_co2, a, (b, Δb), (c, Δc), (d, Δd))
            rrule_test(_co3, a, (b, Δb), (c, Δc), (d, Δd))

            frule_test(_co1, (b, Δb), (c, Δc), (d, Δd))
            frule_test(_co2, (b, Δb), (c, Δc), (d, Δd))
            frule_test(_co3, (b, Δb), (c, Δc), (d, Δd))
        end

        _co4 = @fn∇ 1 function a(α, b, c, β, d, e, f)
            @tensor a[B, A] :=
                -α *
                conj(conj(b[A, C]) * c[C, B] + conj(β) * d[A, C] * (-e[C, B] + 2 * f[C, B]))
        end

        for T in (ComplexF64, Float64)
            a = randn(rng, T, 4, 3)
            b, Δb = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            d, Δd = randn(rng, T, 3, 5), randn(rng, T, 3, 5)
            c, Δc = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            e, Δe = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            f, Δf = randn(rng, T, 5, 4), randn(rng, T, 5, 4)
            α, Δα = randn(rng, T), randn(rng, T)
            β, Δβ = randn(rng, T), randn(rng, T)

            args = ((α, Δα), (b, Δb), (c, Δc), (β, Δβ), (d, Δd), (e, Δe), (f, Δf))
            rrule_test(_co4, a, args...)
            frule_test(_co4, args...)
        end
    end
end
