using ChainRulesTestUtils
using FiniteDifferences
using Random
using TensorRules
using Test
using Zygote

import TensorRules: @fn∇, rhs_to_args, make_only_product

Random.seed!(42)

Zygote.refresh()

@testset "TensorRules" begin
    include("rhsparse.jl")
    include("rules.jl")
    include("zygote.jl")
end
