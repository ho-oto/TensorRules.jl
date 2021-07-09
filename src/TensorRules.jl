module TensorRules

using ChainRulesCore
using LinearAlgebra
using MacroTools
using TensorOperations

export rrule, frule, NoTangent, Zero, Thunk, InplaceableThunk
export I
export @tensor, @tensoropt

export @∇

const defaultdifforder = Ref{Int}(1)

include("parser.jl")
include("generator.jl")
include("macro.jl")

end
