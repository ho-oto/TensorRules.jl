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

## How to use

```julia
julia> using TensorOperations, TensorRules, Zygote;
julia> function foo(a, b, c) # define function with Einstein summation
           # d_F = \sum_{A,B,C,D} a_{A,B,C} b_{C,D,E,F} c_{A,B,D,E}
           @tensor d[F] := a[A, B, C] * b[C, D, E, F] * c[A, B, D, E]
           return d[1]
       end;
julia> a, b, c = randn(3, 4, 5), randn(5, 6, 7, 8), randn(3, 4, 6, 7);
julia> gradient(foo, a, b, c); # try to obtain gradient of `foo` by Zygote
ERROR: this intrinsic must be compiled to be called
Stacktrace:
...
julia> @∇ function foo(a, b, c) # use @∇
           @tensor d[F] := a[A, B, C] * b[C, D, E, F] * c[A, B, D, E]
           return d[1]
       end;
julia> gradient(foo, a, b, c); # it works!
```

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
    Px1, Px2, Px3, Px4, Px5 = ProjectTo(x1), ProjectTo(x2), ProjectTo(x3), ProjectTo(x4), ProjectTo(x5)
    function _foo_1_pullback(Δf)
        fnΔx1(Δf, x1, x2, x3, x4, x5) = @tensoropt !C _[A, C] := conj(Δf[A, B]) * x2[C, D] * x3[D, B]
        fnΔx2(Δf, x1, x2, x3, x4, x5) = @tensoropt !C _[C, D] := conj(conj(x1[A, C]) * conj(Δf[A, B]) * x3[D, B])
        fnΔx3(Δf, x1, x2, x3, x4, x5) = @tensoropt !C _[D, B] := conj(conj(x1[A, C]) * x2[C, D] * conj(Δf[A, B]))
        fnΔx4(Δf, x1, x2, x3, x4, x5) = first(@tensoropt !C _[] := conj(conj(Δf[A, B]) * x5[A, B]))
        fnΔx5(Δf, x1, x2, x3, x4, x5) = @tensoropt !C _[A, B] := conj(x4 * conj(Δf[A, B]))
        Δx1 = @thunk Px1(fnΔx1(Δf, x1, x2, x3, x4, x5))
        Δx2 = @thunk Px2(fnΔx2(Δf, x1, x2, x3, x4, x5))
        Δx3 = @thunk Px3(fnΔx3(Δf, x1, x2, x3, x4, x5))
        Δx4 = @thunk Px4(fnΔx4(Δf, x1, x2, x3, x4, x5))
        Δx5 = @thunk Px5(fnΔx5(Δf, x1, x2, x3, x4, x5))
        return (NoTangent(), Δx1, Δx2, Δx3, Δx4, Δx5)
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
- [ ] support higher order differentiation (by applying `@∇` to `rrule` and `frule` recursively)
  - [ ] add more test (higher order differentiations are not well tested
    since `Zygote.jl` has poor support of higher order differentiation...😞)
  - [ ] better support of `InplaceableThunk` (in this version, when we use `@∇ i foo(...) = ...`
    where `i > 1`, `InplaceableThunk` will be disabled)
- [x] use `@thunk` ?
