# TensorRules.jl

[![Build Status](https://github.com/ho-oto/TensorRules.jl/workflows/CI/badge.svg)](https://github.com/ho-oto/TensorRules.jl/actions)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

`TensorRules.jl` provides a macro `@∇` (you can type `∇` by `\nabla<tab>`), which
enable us to use automatic differentiation (AD) libraries (e.g.,
[`Zygote.jl`](https://github.com/FluxML/Zygote.jl),
[`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl))
with `@tensor` and `@tensoropt` macros in [`TensorOperations.jl`](https://github.com/Jutho/TensorOperations.jl).

`TensorRules.jl` uses [`ChainRulesCore.jl`](https://github.com/JuliaDiff/ChainRulesCore.jl) to define custom adjoints.
So, you can use any AD libraries which supports `ChainRulesCore.jl`.

## How it works

The strategy of `TensorRules.jl` are very similar to [`TensorGrad.jl`](https://github.com/mcabbott/TensorGrad.jl).

`@∇` converts functions which contains `@tensor` or `@tensoropt` macro.
First, `@∇` detects `@tensor` or `@tensoropt` expressions in function definition
and convert them to inlined functions.
Then, `@∇` define custom adjoint rules for the generated functions.

For example, the following definition

```julia
@∇ function foo(a, b, c, d, e, f)
    @tensoropt !C x[A, B] := conj(a[A, C]) * sin.(b)[C, D] * c.d[D, B] + d * e[1, 2][A, B]
    x = x + f
    @tensor x[A, B] += a[A, C] * (a * a)[C, B]
    return x
end
```

will be converted to a code equivalent to

```julia
function foo(a, b, c, d, e, f)
    x = _foo_1(a, sin.(a), c.d, d, e[1, 2])
    x = x + f
    x += _foo_2(a, a * a)
    return x
end

@inline _foo_1(x1, x2, x3, x4, x5) =
    @tensoropt !C _[A, B] := conj(x1[A, C]) * x2[C, D] * x3[D, B] + x4 * x5[A, B]

@inline _foo_2(x1, x2) = @tensor _[A, B] := x1[A, C] * x2[C, B]

function rrule(::typeof(_foo_1), x1, x2, x3, x4, x5)
    f = _foo_1(x1, x2, x3, x4, x5)
    function _foo_1_pullback(Δf)
        Δx1 = InplaceableThunk(
            Thunk(() -> @tensoropt !C Δx1[A, C] := conj(Δf[A, B]) * x2[C, D] * x3[D, B]),
            Δx1 -> @tensoropt !C Δx1[A, C] += conj(Δf[A, B]) * x2[C, D] * x3[D, B]
        )
        Δx2 = InplaceableThunk(
            Thunk(() -> @tensoropt !C Δx2[C, D] := conj(conj(x1[A, C]) * conj(Δf[A, B]) * x3[D, B])),
            Δx2 -> @tensoropt !C Δx2[C, D] += conj(conj(x1[A, C]) * conj(Δf[A, B]) * x3[D, B])
        )
        Δx3 = InplaceableThunk(
            Thunk(() -> @tensoropt !C Δx3[D, B] := conj(conj(x1[A, C]) * x2[C, D] * conj(Δf[A, B]))),
            Δx3 -> @tensoropt !C Δx3[D, B] += conj(conj(x1[A, C]) * x2[C, D] * conj(Δf[A, B]))
        )
        Δx4 = Thunk(() -> first(@tensoropt !C Δx4[] := conj(conj(Δf[A, B]) * x5[A, B])))
        Δx5 = InplaceableThunk(
            Thunk(() -> @tensoropt !C Δx5[A, B] := conj(x4 * conj(Δf[A, B]))),
            Δx5 -> @tensoropt !C Δx5[A, B] := conj(x4 * conj(Δf[A, B]))
        )
        return (NO_FIELDS, Δx1, Δx2, Δx3, Δx4, Δx5)
    end
    return f, _foo_1_pullback
end

function rrule(::typeof(_foo_2), x1, x2)
    ...
end
```

By using `Thunk` and `InplaceableThunk` properly, adjoints will be evaluated only
if they are needed.

## unsupported features

- `@∇` uses `@capture` macro defined in [`MacroTools.jl`](https://github.com/FluxML/MacroTools.jl)
to parse `Expr`. Because of the limitation of `@capture` macro,
index notations based on `:typed_vcat` and `:typed_hcat` (`A[a; b], A[a b]`)
are unsupported. Please use `A[a, b]` style.
- Designations of contraction order based on `ord=(...)` or NCON style are unsupported.
Please use `@tensoropt` and specify costs of each bonds.
- Since `Zygote.jl` does not support inplace operations, we cannot use `@tensor A[] = ...`
in the expression. Please use `:=`, `+=` and `-=` instead.

## TODO

- [x] support `frule`
- [ ] support `@tensor` block (`@tensor begin ... end`)
- [x] support higher order differentiation (by applying `@∇` to `rrule` and `frule` recursively)
    - [ ] add more test (higher order differentiations are not well tested
    since `Zygote.jl` has poor support of higher order differentiation...😞)
    - [ ] better support of `InplaceableThunk` (in this version, when we use `@∇ i foo(...) = ...`
    where `i > 1`, `InplaceableThunk` will be disabled)
- [x] use `@thunk` ?