using ChainRulesTestUtils
using FiniteDifferences
using Random
using TensorRules
using Test
using Zygote

import TensorRules: @fn∇, rhs_to_args, make_only_product

Random.seed!(42)

Zygote.refresh()

# work around for test
function ChainRulesTestUtils.rand_tangent(x::StridedArray{T,0}) where {T}
    return fill(ChainRulesTestUtils.rand_tangent(x[]))
end

@testset "TensorRules" begin
    include("rhsparse.jl")
    include("rules.jl")
    include("zygote.jl")
end
