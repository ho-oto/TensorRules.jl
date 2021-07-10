@testset "RHS parse" begin
    ex = quote
        (a[1:end, :][1, 2][a, b'] + sin(cos(B))[b', (a')']) * (C * D + E)[(a')', a] * 3 * 2 + α * a[a, a] * K[1, 2, 3][a, a] -
        (((L[a, b] * P.P[b, c]) * (M[c, d] * N[d, e]) * Z[e, f]) * D[f, a]) * π
    end
    @test rhs_to_args(ex)[2] == [
        :(a[1:end, :][1, 2]),
        :(sin(cos(B))),
        :(C * D + E),
        :α,
        :a,
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
    @test make_only_product(ex, :a) == :(-a * (-b + (c + d) + (-e) - f - (g * h)) * i)
    @test make_only_product(ex, :b) == :(-a * -b * i)
    @test make_only_product(ex, :c) == :(-a * c * i)
    @test make_only_product(ex, :d) == :(-a * d * i)
    @test make_only_product(ex, :e) == :(-a * -e * i)
    @test make_only_product(ex, :f) == :(-a * -f * i)
    @test make_only_product(ex, :g) == :(-a * -(g * h) * i)
    @test make_only_product(ex, :h) == :(-a * -(g * h) * i)
    @test make_only_product(ex, :i) == :(-a * (-b + (c + d) + (-e) - f - (g * h)) * i)
    @test make_only_product(ex, :j) == :j

    ex = :(-a * (-conj(conj(b)) + conj(conj(c + d * e) + f) + -g))
    @test make_only_product(ex, :a) ==
          :(-a * (-conj(conj(b)) + conj(conj(c + d * e) + f) + -g))
    @test make_only_product(ex, :b) == :(-a * -conj(conj(b)))
    @test make_only_product(ex, :c) == :(-a * conj(conj(c)))
    @test make_only_product(ex, :d) == :(-a * conj(conj(d * e)))
    @test make_only_product(ex, :e) == :(-a * conj(conj(d * e)))
    @test make_only_product(ex, :f) == :(-a * conj(f))
    @test make_only_product(ex, :g) == :(-a * -(g))
end
