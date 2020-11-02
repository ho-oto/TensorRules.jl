using TensorRules
using Test

ex = quote
    (A[1:end, :][1, 2][a, b'] + sin(cos(B))[b', a'']) * (C*D+E)[a'', a] * 3 * 2 +
    α * C[a, a] * K[1, 2, 3][a, a] -
    (((L[a, b] * P[b, c]) * (M[c, d] * N[d, e]) * Z[e, f]) * D[f, a]) * π
end

@testset "TensorRules.jl" begin
    # Write your tests here.
end
