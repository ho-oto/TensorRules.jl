module TensorRules

using ChainRulesCore
using LinearAlgebra
using MacroTools
using TensorOperations

export rrule, frule, NO_FIELDS, AbstractZero, Zero
export I
export @tensor, @tensoropt

export @∇, @∇genedfunc

const showexpr = Ref(false)

function rhs_to_args(ex::Expr)
    indsall = Union{Symbol,Expr}[]
    symorig, symgend = Union{Symbol,Expr}[], Symbol[]

    exparse(x) =
        if @capture(x, -(rhs__))
            y = exparse.(rhs)
            :(-($(y...)))
        elseif @capture(x, +(rhs__))
            y = exparse.(rhs)
            :(+($(y...)))
        elseif @capture(x, *(rhs__))
            y = exparse.(rhs)
            :(*($(y...)))
        elseif @capture(x, conj(sym_[ind__]))
            # NOTE: :typed_vcat (e.g., A[a; b]) and :typed_hcat (e.g., A[a b]) are
            # unsupported since the limitation of @capture macro
            all(x -> !isa(x, Integer), ind) || error("NCON style is unsupported")
            new = gensym()
            push!(symorig, sym)
            push!(symgend, new)
            append!(indsall, ind)
            :(conj($new[$(ind...)]))
        elseif @capture(x, sym_[ind__])
            all(x -> !isa(x, Integer), ind) || error("NCON style is unsupported")
            new = gensym()
            push!(symorig, sym)
            push!(symgend, new)
            append!(indsall, ind)
            :($new[$(ind...)])
        elseif x isa Number
            x
        else
            new = gensym()
            push!(symorig, x)
            push!(symgend, new)
            new
        end

    exreplaced = exparse(ex)
    return exreplaced, symorig, symgend, indsall
end

function make_only_product(ex::Expr, sym::Symbol)
    hassym(x) =
        if @capture(x, -(y__) | +(y__) | *(y__))
            any(hassym.(y))
        elseif @capture(x, $sym) || @capture(x, $sym[__]) || @capture(x, conj($sym[__]))
            true
        else
            false
        end

    return MacroTools.postwalk(ex) do x
        if @capture(x, -(y__))
            @assert 1 ≤ length(y) ≤ 2
            if length(y) == 1
                x
            elseif hassym(first(y))
                first(y)
            elseif hassym(last(y))
                :(-$(last(y)))
            else
                x
            end
        elseif @capture(x, +(y__))
            @assert 1 ≤ length(y)
            y = filter(hassym, y)
            @assert length(y) ≤ 1
            length(y) == 1 ? first(y) : x
        else
            x
        end
    end
end

function gen_func(
    funcname::Symbol,
    args::Vector{Symbol},
    lhsind::Vector{Any},
    rhs::Expr,
    opt::Ref{Expr},
)
    ex = if isempty(lhsind)
        :($funcname[] := $rhs)
    else
        :($funcname[$(lhsind...)] := $rhs)
    end

    ex = if isassigned(opt)
        :(@inline $funcname($(args...)) = @tensoropt $(opt[]) $ex)
    else
        :(@inline $funcname($(args...)) = @tensor $ex)
    end

    return macroexpand(TensorOperations, ex)
end

function gen_rule(
    funcname::Symbol,
    args::Vector{Symbol},
    lhsind::Vector{Any},
    rhs::Expr,
    indsall::Vector{Union{Symbol,Expr}},
    opt::Ref{Expr},
)
    indsall = [lhsind; indsall]

    @gensym Δlhssym
    Δlhs = if isempty(lhsind)
        :(first($Δlhssym))
    else
        :($Δlhssym[$(lhsind...)])
    end

    Δargs, Δexargs = Symbol[], Expr[]
    for arg in args
        Δarg = gensym()
        rhsarg = make_only_product(rhs, arg)

        ind, isconj = Ref{Vector{Any}}(), Ref{Bool}()
        Δexarg = MacroTools.prewalk(rhsarg) do x # assume to match only once
            if @capture(x, conj($arg[_ind__]))
                ind[], isconj[] = _ind, true
                :(conj($Δlhs))
            elseif @capture(x, $arg[_ind__])
                ind[], isconj[] = _ind, false
                :(conj($Δlhs))
            elseif @capture(x, $arg)
                :(conj($Δlhs))
            else
                x
            end
        end
        @assert (isassigned(ind) && !isempty(ind[])) || !isassigned(ind)
        istensor = isassigned(ind)
        isconj = (isassigned(isconj) ? isconj[] : false)

        shouldtr = istensor ? ind[] ≠ unique(ind[]) : false
        indtr = Union{Symbol,Expr}[]

        if shouldtr
            δs = Expr[]
            for (k, i) in enumerate(ind[])
                if i ∉ indtr
                    push!(indtr, i)
                else
                    j = :($i')
                    while j ∈ [indsall; indtr]
                        j = :($j')
                    end
                    push!(indtr, j)
                    push!(
                        δs,
                        :($Array{eltype($arg)}(I, size($arg, $k), size($arg, $k))[$i, $j]),
                    )
                end
            end
            Δexarg = :(*($Δexarg, $(δs...)))
        elseif istensor
            append!(indtr, ind[])
        end

        Δexarg = if istensor
            if isassigned(opt)
                :(@tensoropt $(opt[]) $Δarg[$(indtr...)] := $Δexarg)
            else
                :(@tensor $Δarg[$(indtr...)] := $Δexarg)
            end
        else
            if isassigned(opt)
                :(@tensoropt $(opt[]) $Δarg[] := $Δexarg;
                $Δarg = first($Δarg))
            else
                :(@tensor $Δarg[] := $Δexarg;
                $Δarg = first($Δarg))
            end
        end
        Δexarg = isconj ? Δexarg : Expr(:block, Δexarg, :($Δarg = conj($Δarg)))

        push!(Δargs, Δarg)
        push!(Δexargs, Δexarg)
    end

    showexpr[] && @show prettify.(rmlines.(Δexargs))
    Δexargs = map(x -> macroexpand(TensorOperations, x), Δexargs)

    @gensym valforw funcback
    backbody = Expr(
        :block,
        :(
            if $Δlhssym isa AbstractZero
                return (NO_FIELDS, $(fill(Zero(), length(Δargs))...))
            end
        ),
        :($Δlhssym = Array($Δlhssym)),
        Δexargs...,
        :(return (NO_FIELDS, $(Δargs...))),
    )

    return quote
        function ChainRulesCore.rrule(::typeof($funcname), $(args...))
            $valforw = $(funcname)($(args...))
            $(funcback)($Δlhssym) = $backbody
            return ($valforw, $funcback)
        end
    end
end

function _nabla(ex::Expr; mod)
    def = splitdef(ex)
    symfuncs, exfuncs, exrules = Symbol[], Expr[], Expr[]

    def[:body] = MacroTools.postwalk(def[:body]) do x
        lhs, lhsind, rhs = Ref{Symbol}(), Ref{Vector{Any}}(), Ref{Expr}()
        which, opt = Ref{Symbol}(), Ref{Expr}()

        if @capture(x, @tensor _lhs_[_lhsind__] := _rhs_) ||
           @capture(x, @tensoropt _lhs_[_lhsind__] := _rhs_)
            lhs[], lhsind[], rhs[] = _lhs, _lhsind, _rhs
            which[] = :assign
        elseif @capture(x, @tensor _lhs_[_lhsind__] += _rhs_) ||
               @capture(x, @tensoropt _lhs_[_lhsind__] += _rhs_)
            lhs[], lhsind[], rhs[] = _lhs, _lhsind, _rhs
            which[] = :pluseq
        elseif @capture(x, @tensor _lhs_[_lhsind__] -= _rhs_) ||
               @capture(x, @tensoropt _lhs_[_lhsind__] -= _rhs_)
            lhs[], lhsind[], rhs[] = _lhs, _lhsind, _rhs
            which[] = :subteq
        elseif @capture(x, @tensoropt _opt_ _lhs_[_lhsind__] := _rhs_)
            lhs[], lhsind[], rhs[] = _lhs, _lhsind, _rhs
            which[], opt[] = :assign, _opt
        elseif @capture(x, @tensoropt _opt_ _lhs_[_lhsind__] += _rhs_)
            lhs[], lhsind[], rhs[] = _lhs, _lhsind, _rhs
            which[], opt[] = :pluseq, _opt
        elseif @capture(x, @tensoropt _opt_ _lhs_[_lhsind__] -= _rhs_)
            lhs[], lhsind[], rhs[] = _lhs, _lhsind, _rhs
            which[], opt[] = :subteq, _opt
        elseif @capture(x, @tensor _[__] = _) ||
               @capture(x, @tensoropt _[__] = _) ||
               @capture(x, @tensoropt _ _[__] = _)
            error("use assignment with `:=` instead of inplace operation with `=` for Zygote")
        else
            return x
        end

        lhs, lhsind, rhs, which = lhs[], lhsind[], rhs[], which[]

        rhsreplace, argsorig, argsdummy, indsall = rhs_to_args(rhs)
        @gensym symfunc

        push!(symfuncs, symfunc)

        @eval mod $(gen_func(symfunc, argsdummy, lhsind, rhsreplace, opt))
        @eval mod $(gen_rule(symfunc, argsdummy, lhsind, rhsreplace, indsall, opt))

        if which == :assign
            return :($lhs = $(Core.eval(mod, symfunc))($(argsorig...)))
        elseif which == :pluseq # use x += y instead of x .+= y for Zygote
            return :($lhs += $(Core.eval(mod, symfunc))($(argsorig...)))
        elseif which == :subteq # use x -= y instead of x .-= y for Zygote
            return :($lhs -= $(Core.eval(mod, symfunc))($(argsorig...)))
        end
    end

    return esc(MacroTools.combinedef(def)), symfuncs
end

macro ∇(ex)
    _nabla(ex; mod = @__MODULE__)[1]
end

macro ∇genedfunc(i::Int, ex)
    _nabla(ex; mod = @__MODULE__)[2][i]
end

end
