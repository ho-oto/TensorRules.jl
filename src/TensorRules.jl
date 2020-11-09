module TensorRules

using ChainRulesCore
using LinearAlgebra
using MacroTools
using TensorOperations

export rrule, frule, NO_FIELDS, Zero, Thunk, InplaceableThunk
export I
export @tensor, @tensoropt

export @∇

const defaultorder = Ref{Int}(1)

function rhs_to_args(ex::Expr)
    indsall = Union{Symbol,Expr}[]
    symorig, symgend, isconjs = Union{Symbol,Expr}[], Symbol[], Bool[]

    exparse(x, isconj) =
        if x isa Number
            x
        elseif @capture(x, -(rhs__))
            y = exparse.(rhs, isconj)
            :(-($(y...)))
        elseif @capture(x, +(rhs__))
            y = exparse.(rhs, isconj)
            :(+($(y...)))
        elseif @capture(x, *(rhs__))
            y = exparse.(rhs, isconj)
            :(*($(y...)))
        elseif @capture(x, conj(rhs_))
            y = exparse(rhs, !isconj)
            :(conj($y))
        elseif @capture(x, sym_[ind__])
            any(x -> isa(x, Integer), ind) && error("NCON style is unsupported")
            @gensym new
            push!(symorig, sym)
            push!(symgend, new)
            push!(isconjs, isconj)
            append!(indsall, ind)
            :($new[$(ind...)])
        else
            @gensym new
            push!(symorig, x)
            push!(symgend, new)
            push!(isconjs, isconj)
            new
        end

    exreplaced = exparse(ex, false)
    return exreplaced, symorig, symgend, isconjs, indsall
end

function make_only_product(ex::Expr, sym::Symbol)
    hassym(x) =
        if @capture(x, -(y__) | +(y__) | *(y__))
            any(hassym.(y))
        elseif @capture(x, conj(y_))
            hassym(y)
        elseif @capture(x, $sym) || @capture(x, $sym[__])
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

function make_scalar_first(ex::Expr)
    return MacroTools.postwalk(ex) do x
        if @capture(x, *(y__))
            tensors = Expr[]
            notensors = Union{Expr,Symbol,Number}[]

            for z in y
                if @capture(z, _[__]) || @capture(z, conj(_[__]))
                    push!(tensors, z)
                else
                    push!(notensors, z)
                end
            end

            isempty(tensors) && return x

            if isempty(notensors)
                return x
            elseif length(notensors) == 1
                return :(*($(notensors[1]), $(tensors...)))
            else
                return :(*(*($(notensors...)), $(tensors...)))
            end
        else
            return x
        end
    end
end

function genfunc(
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

    if isassigned(opt)
        return :(@inline $funcname($(args...)) = @tensoropt $(opt[]) $ex)
    else
        return :(@inline $funcname($(args...)) = @tensor $ex)
    end
end

function genfrule(
    funcname::Symbol,
    args::Vector{Symbol},
    lhsind::Vector{Any},
    rhs::Expr,
    opt::Ref{Expr},
)
    ȧrgs, ṙhss = Symbol[], Expr[]
    for arg in args
        @gensym ȧrg
        ṙhs = make_scalar_first(make_only_product(rhs, arg))

        ṙhs = MacroTools.prewalk(ṙhs) do x # assume to match only once
            @capture(x, $arg) ? ȧrg : x
        end

        push!(ȧrgs, ȧrg)
        push!(ṙhss, ṙhs)
    end
    ṙhs = :(+$(ṙhss...))

    @gensym val v̇al
    lhs = isempty(lhsind) ? :($v̇al[]) : :($v̇al[$(lhsind...)])

    pushbody = if isassigned(opt)
        :(@tensoropt $(opt[]) $lhs := $ṙhs)
    else
        :(@tensor $lhs := $ṙhs)
    end

    return quote
        function ChainRulesCore.frule((_, $(ȧrgs...)), ::typeof($funcname), $(args...))
            $val = $(funcname)($(args...))
            $pushbody
            return ($val, $v̇al)
        end
    end
end

function genrrule(
    funcname::Symbol,
    args::Vector{Symbol},
    lhsind::Vector{Any},
    rhs::Expr,
    opt::Ref{Expr},
    isconjs::Vector{Bool},
    indsall::Vector{Union{Symbol,Expr}};
    useinplace::Bool,
)
    indsall = [lhsind; indsall]

    @gensym Δlhssym
    Δlhs = if isempty(lhsind)
        :(first($Δlhssym))
    else
        :($Δlhssym[$(lhsind...)])
    end

    ∂args, ∂exargs = Symbol[], Expr[]
    for (arg, isconj) in zip(args, isconjs)
        @gensym ∂arg
        rhsarg = make_scalar_first(make_only_product(rhs, arg))

        ind = Ref{Vector{Any}}()
        ∂exrhs = MacroTools.prewalk(rhsarg) do x # assume to match only once
            if @capture(x, $arg[_ind__])
                ind[] = _ind
                isconj ? Δlhs : :(conj($Δlhs))
            elseif @capture(x, $arg)
                isconj ? Δlhs : :(conj($Δlhs))
            else
                x
            end
        end
        @assert (isassigned(ind) && !isempty(ind[])) || !isassigned(ind)
        istensor = isassigned(ind)

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
            ∂exrhs = :(*($∂exrhs, $(δs...)))
        elseif istensor
            append!(indtr, ind[])
        end

        ∂exrhs = isconj ? ∂exrhs : :(conj($∂exrhs))

        ∂exval = if istensor
            if isassigned(opt)
                :(@tensoropt $(opt[]) $∂arg[$(indtr...)] := $∂exrhs)
            else
                :(@tensor $∂arg[$(indtr...)] := $∂exrhs)
            end
        else
            if isassigned(opt)
                :(first(@tensoropt $(opt[]) $∂arg[] := $∂exrhs))
            else
                :(first(@tensor $∂arg[] := $∂exrhs))
            end
        end

        ∂exadd! = if istensor
            if isassigned(opt)
                :($∂arg -> @tensoropt $(opt[]) $∂arg[$(indtr...)] += $∂exrhs)
            else
                :($∂arg -> @tensor $∂arg[$(indtr...)] += $∂exrhs)
            end
        else
            nothing
        end

        ∂exarg = if istensor && useinplace
            :($∂arg = InplaceableThunk(Thunk(() -> $∂exval), $∂exadd!))
        else
            :($∂arg = Thunk(() -> $∂exval))
        end

        push!(∂args, ∂arg)
        push!(∂exargs, ∂exarg)
    end

    @gensym valforw funcback
    backbody = Expr(
        :block,
        :($Δlhssym = Array($Δlhssym)),
        ∂exargs...,
        :(return (NO_FIELDS, $(∂args...))),
    )
    zerobody = :((NO_FIELDS, $(fill(Zero(), length(∂args))...)))

    return quote
        function ChainRulesCore.rrule(::typeof($funcname), $(args...))
            $valforw = $(funcname)($(args...))
            $(funcback)($Δlhssym) = $backbody
            $(funcback)(::Zero) = $zerobody
            return ($valforw, $funcback)
        end
    end
end

function _nabla(ex::Expr, i::Integer = defaultorder[]; mod)
    @assert i ≥ 1

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

        rhsreplace, argsorig, argsdummy, isconjs, indsall = rhs_to_args(rhs)
        @gensym symfunc
        push!(symfuncs, symfunc)

        genargs = (symfunc, argsdummy, lhsind, rhsreplace, opt)

        exfunc = genfunc(genargs...)
        exfrule = genfrule(genargs...)
        exrrule = genrrule(genargs..., isconjs, indsall; useinplace = i == 1)

        @eval mod $(macroexpand(TensorOperations, exfunc))
        if i > 1
            @eval mod $(:(@∇ $(i - 1) $exfrule))
            @eval mod $(:(@∇ $(i - 1) $exrrule))
        else
            @eval mod $(macroexpand(TensorOperations, exfrule))
            @eval mod $(macroexpand(TensorOperations, exrrule))
        end

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
    ex, _ = _nabla(ex; mod = @__MODULE__)
    return ex
end

macro ∇(i::Integer, ex)
    ex, _ = _nabla(ex, i; mod = @__MODULE__)
    return ex
end

macro fn∇(i, ex)
    _, fn = _nabla(ex; mod = @__MODULE__)
    return fn[i]
end

end
