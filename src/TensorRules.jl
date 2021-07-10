module TensorRules

using ChainRulesCore
using LinearAlgebra
using MacroTools
using TensorOperations

export rrule, frule, NoTangent, Zero, Thunk, InplaceableThunk, ProjectTo
export I
export @tensor, @tensoropt

export @∇

include("parser.jl")
include("generator.jl")
include("macro.jl")

end
