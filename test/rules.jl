@testset "rules" begin
    @testset "einsum" begin
        _esum = @fn∇ 1 function (b, c, d)
            @tensor a[B, A] := conj(b[A, C]) * c[C, D] * d[B, D]
        end

        for T in (ComplexF64, Float64)
            b = randn(T, 3, 5)
            c = randn(T, 5, 4)
            d = randn(T, 4, 4)

            test_rrule(_esum, b, c, d; check_inferred=false)
            test_frule(_esum, b, c, d)
        end
    end

    @testset "tensoropt" begin
        _opt1 = @fn∇ 1 function (b, c, d)
            @tensoropt (A => 1, C => χ) a[B, A] := b[A, C] * c[C, D] * d[B, D]
        end
        _opt2 = @fn∇ 1 function (b, c, d)
            @tensoropt !(A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]
        end
        _opt3 = @fn∇ 1 function (b, c, d)
            @tensoropt (A, C) a[B, A] := b[A, C] * c[C, D] * d[B, D]
        end

        for T in (ComplexF64, Float64)
            b = randn(T, 3, 5)
            c = randn(T, 5, 4)
            d = randn(T, 4, 4)

            test_rrule(_opt1, b, c, d; check_inferred=false)
            test_rrule(_opt2, b, c, d; check_inferred=false)
            test_rrule(_opt3, b, c, d; check_inferred=false)

            test_frule(_opt1, b, c, d)
            test_frule(_opt2, b, c, d)
            test_frule(_opt3, b, c, d)
        end
    end

    @testset "scalar in rhs" begin
        _scar = @fn∇ 1 function (α, b, c)
            @tensor a[B, A] := α * conj(b[A, C]) * c[C, B]
        end

        for T in (ComplexF64, Float64)
            b = randn(T, 3, 5)
            c = randn(T, 5, 4)
            α = randn(T)

            test_rrule(_scar, α, b, c; check_inferred=false)
            test_frule(_scar, α, b, c)
        end
    end

    @testset "add" begin
        _add = @fn∇ 1 function (α, b, c, β, d, e, f)
            @tensor a[B, A] :=
                -α * conj(b[A, C]) * c[C, B] + conj(β) * d[A, C] * (-e[C, B] + 2 * f[C, B])
        end

        for T in (ComplexF64, Float64)
            α = randn(T)
            b = randn(T, 3, 5)
            c = randn(T, 5, 4)
            β = randn(T)
            d = randn(T, 3, 5)
            e = randn(T, 5, 4)
            f = randn(T, 5, 4)

            test_rrule(_add, α, b, c, β, d, e, f; check_inferred=false)
            test_frule(_add, α, b, c, β, d, e, f)
        end
    end

    @testset "trace" begin
        _tr1 = @fn∇ 1 function (b, c)
            @tensor a[C] := b[A, B, B', B', B] * c[A, A', A', C]
        end
        _tr2 = @fn∇ 1 function (b, c)
            @tensor a[c] := b[a, b, bb, bb, b] * c[a, aa, aa, c]
        end

        for T in (ComplexF64, Float64)
            b = randn(T, 3, 2, 3, 3, 2)
            c = randn(T, 3, 3, 3, 2)

            test_rrule(_tr1, b, c; check_inferred=false)
            test_rrule(_tr2, b, c; check_inferred=false)

            test_frule(_tr1, b, c)
            test_frule(_tr2, b, c)
        end
    end

    @testset "lhs is scalar" begin
        _scal = @fn∇ 1 function (b, c, d)
            @tensor a[] := b[A, B] * c[B, C] * d[C, A]
        end

        for T in (ComplexF64, Float64)
            b = randn(T, 3, 2)
            c = randn(T, 2, 4)
            d = randn(T, 4, 3)

            test_rrule(_scal, b, c, d; check_inferred=false)
            test_frule(_scal, b, c, d)
        end
    end

    @testset "conj" begin
        _co1 = @fn∇ 1 function (b, c, d)
            @tensor a[B, A] := conj(b[A, C]) * conj(c[C, D] * d[B, D])
        end
        _co2 = @fn∇ 1 function (b, c, d)
            @tensor a[B, A] := conj(b[A, C] * c[C, D] * d[B, D])
        end
        _co3 = @fn∇ 1 function (b, c, d)
            @tensor a[B, A] := conj(
                1.23 * conj(conj(conj(b[A, C]))) * conj(c[C, D] * d[B, D])
            )
        end

        for T in (ComplexF64, Float64)
            b = randn(T, 3, 5)
            c = randn(T, 5, 4)
            d = randn(T, 4, 4)
            α = randn(T)
            β = randn(T)

            test_rrule(_co1, b, c, d; check_inferred=false)
            test_rrule(_co2, b, c, d; check_inferred=false)
            test_rrule(_co3, b, c, d; check_inferred=false)

            test_frule(_co1, b, c, d)
            test_frule(_co2, b, c, d)
            test_frule(_co3, b, c, d)
        end

        _co4 = @fn∇ 1 function (α, b, c, β, d, e, f)
            @tensor a[B, A] :=
                -α *
                conj(conj(b[A, C]) * c[C, B] + conj(β) * d[A, C] * (-e[C, B] + 2 * f[C, B]))
        end

        for T in (ComplexF64, Float64)
            α = randn(T)
            b = randn(T, 3, 5)
            c = randn(T, 5, 4)
            β = randn(T)
            d = randn(T, 3, 5)
            e = randn(T, 5, 4)
            f = randn(T, 5, 4)

            test_rrule(_co4, α, b, c, β, d, e, f; check_inferred=false)
            test_frule(_co4, α, b, c, β, d, e, f)
        end
    end

    @testset "more" begin
        T = ComplexF64

        _tn1 = @fn∇ 1 function (x, ar, H)
            return @tensoropt !(p, p1, p2) l[l, p, r] :=
                x[l, p1, X] * ar[X, p2, Y] * conj(ar[r, p2', Y]) * H[p, p2', p1, p2]
        end
        _tn2 = @fn∇ 1 function (x, al, H)
            return @tensoropt !(p, p1, p2) r[l, p, r] :=
                conj(al[X, p1', l]) * al[X, p1, Y] * x[Y, p2, r] * H[p1', p, p1, p2]
        end

        a1 = randn(T, 3, 2, 3)
        a2 = randn(T, 3, 2, 3)
        x = randn(T, 3, 2, 3)
        H = randn(T, 2, 2, 2, 2)
        r = randn(T, 3, 2, 3)

        test_rrule(_tn1, x, a1, a2, H; check_inferred=false)
        test_rrule(_tn2, a1, a2, x, H; check_inferred=false)

        test_frule(_tn1, x, a1, a2, H)
        test_frule(_tn2, a1, a2, x, H)

        _tn3 = @fn∇ 1 function (λ, o, a, b=a)
            return @tensoropt !(p1, p2) _[l, r] :=
                λ *
                conj(b[1][X, p1', Y']) *
                conj(b[2][Y', p2', l]) *
                a[1][X, p1, Y] *
                a[2][Y, p2, r] *
                o[p1', p2', p1, p2]
        end
        _tn4 = @fn∇ 1 function (x, o, a, b=a)
            return @tensoropt !(p1, p2) _[l, r] :=
                a[1][l, p1, X] *
                a[2][X, p2, Y] *
                x[Y, Y'] *
                conj(b[1][r, p1', X']) *
                conj(b[2][X', p2', Y']) *
                o[p1', p2', p1, p2]
        end

        a1 = randn(T, 3, 2, 3)
        a2 = randn(T, 3, 2, 3)
        b1 = randn(T, 3, 2, 3)
        b2 = randn(T, 3, 2, 3)
        x = randn(T, 3, 3)
        λ = randn(T)
        H = randn(T, 2, 2, 2, 2)
        r = randn(T, 3, 3)

        test_rrule(_tn3, λ, b1, b2, a1, a2, H; check_inferred=false)
        test_rrule(_tn4, a1, a2, x, b1, b2, H; check_inferred=false)

        test_frule(_tn3, λ, b1, b2, a1, a2, H)
        test_frule(_tn4, a1, a2, x, b1, b2, H)
    end
end
