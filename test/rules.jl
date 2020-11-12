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

    @testset "more" begin
        T = ComplexF64

        _tn1 = @fn∇ 1 function a(x, ar, H)
            return @tensoropt !(p, p1, p2) l[l, p, r] :=
                x[l, p1, X] * ar[X, p2, Y] * conj(ar[r, p2', Y]) * H[p, p2', p1, p2]
        end
        _tn2 = @fn∇ 1 function a(x, al, H)
            return @tensoropt !(p, p1, p2) r[l, p, r] :=
                conj(al[X, p1', l]) * al[X, p1, Y] * x[Y, p2, r] * H[p1', p, p1, p2]
        end

        a1, Δa1 = randn(rng, T, 3, 2, 3), randn(rng, T, 3, 2, 3)
        a2, Δa2 = randn(rng, T, 3, 2, 3), randn(rng, T, 3, 2, 3)
        x, Δx = randn(rng, T, 3, 2, 3), randn(rng, T, 3, 2, 3)
        H, ΔH = randn(rng, T, 2, 2, 2, 2), randn(rng, T, 2, 2, 2, 2)
        r = randn(rng, T, 3, 2, 3)

        rrule_test(_tn1, r, (x, Δx), (a1, Δa1), (a2, Δa2), (H, ΔH))
        frule_test(_tn1, (x, Δx), (a1, Δa1), (a2, Δa2), (H, ΔH))
        rrule_test(_tn2, r, (a1, Δa1), (a2, Δa2), (x, Δx), (H, ΔH))
        frule_test(_tn2, (a1, Δa1), (a2, Δa2), (x, Δx), (H, ΔH))

        _tn3 = @fn∇ 1 function a(λ, o, a, b=a)
            return @tensoropt !(p1, p2) _[l, r] :=
                λ *
                conj(b[1][X, p1', Y']) *
                conj(b[2][Y', p2', l]) *
                a[1][X, p1, Y] *
                a[2][Y, p2, r] *
                o[p1', p2', p1, p2]
        end
        _tn4 = @fn∇ 1 function a(x, o, a, b=a)
            return @tensoropt !(p1, p2) _[l, r] :=
                a[1][l, p1, X] *
                a[2][X, p2, Y] *
                x[Y, Y'] *
                conj(b[1][r, p1', X']) *
                conj(b[2][X', p2', Y']) *
                o[p1', p2', p1, p2]
        end

        a1, Δa1 = randn(rng, T, 3, 2, 3), randn(rng, T, 3, 2, 3)
        a2, Δa2 = randn(rng, T, 3, 2, 3), randn(rng, T, 3, 2, 3)
        b1, Δb1 = randn(rng, T, 3, 2, 3), randn(rng, T, 3, 2, 3)
        b2, Δb2 = randn(rng, T, 3, 2, 3), randn(rng, T, 3, 2, 3)
        x, Δx = randn(rng, T, 3, 3), randn(rng, T, 3, 3)
        λ, Δλ = randn(rng, T), randn(rng, T)
        H, ΔH = randn(rng, T, 2, 2, 2, 2), randn(rng, T, 2, 2, 2, 2)
        r = randn(rng, T, 3, 3)

        rrule_test(_tn3, r, (λ, Δλ), (b1, Δb1), (b2, Δb2), (a1, Δa1), (a2, Δa2), (H, ΔH))
        frule_test(_tn3, (λ, Δλ), (b1, Δb1), (b2, Δb2), (a1, Δa1), (a2, Δa2), (H, ΔH))
        rrule_test(_tn4, r, (a1, Δa1), (a2, Δa2), (x, Δx), (b1, Δb1), (b2, Δb2), (H, ΔH))
        frule_test(_tn4, (a1, Δa1), (a2, Δa2), (x, Δx), (b1, Δb1), (b2, Δb2), (H, ΔH))
    end
end
