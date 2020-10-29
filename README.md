# TensorRules.jl

[![Build Status](https://github.com/ho-oto/TensorRules.jl/workflows/CI/badge.svg)](https://github.com/ho-oto/TensorRules.jl/actions)


```julia
@∇ function foo(a, b, c, d, e)
    @tensor f[A, B] := conj(a)[A, C] * b[C, B] + e * conj(c[A, B])
    f = f + d
    @tensor g[A, B] := f[A, C] * f[C, B]
    return g
end
```

```julia
function foo(a, b, c, d, e)
    f = _foo_1(e, conj(a), b, conj(c))
    f = f + d
    g = _foo_2(f, f)
    return g
end

@inline _foo_1(x1, x2, x3, x4) = @tensor f[A, B] := x2[A, C] * x3[C, B] + x1 * x4[A, B]

function rrule(::typeof(_foo_1), x1, x2, x3, x4)
    f = _foo_1(x1, x2, x3, x4)
    function _foo_1_pullback(f̄)
        @tensor x̄1 := conj(f)[A, B] * x4[A, B]
        x̄1 = x̄1[]
        @tensor x̄2[A, C] := conj(f)[A, B] * x3[C, B]
        @tensor x̄3[C, B] := x2[A, C] * conj(f)[A, B]
        @tensor x̄4[A, B] := x1 * conj(f)[A, B]
        return (NO_FIELDS, (x̄1, x̄2, x̄3, x̄4))
    end
    return f, _foo_1_pullback
end

@inline _foo_2(x1, x2) = @tensor g[A, B] := x1[A, C] * x2[C, B]

function rrule(::typeof(_foo_2), x1, x2)
    f = _foo_2(x1, x2)
    function _foo_2_pullback(f̄)
        @tensor x̄1[A, C] := conj(f)[A, B] * x2[C, B]
        @tensor x̄2[C, B] := x1[A, C] * conj(f)[A, B]
        return (NO_FIELDS, (x̄1, x̄2))
    end
    return f, _foo_2_pullback
end
```