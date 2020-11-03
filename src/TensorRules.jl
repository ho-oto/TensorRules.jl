module TensorRules

using ChainRulesCore
using MacroTools
using TensorOperations

export @∇

function _ex_to_string(ex)
    ex isa Symbol && return string(ex)
    ex = repr(ex)
    str = match(r"^:\((?<str>.+)\)$", ex)
    isnothing(str) ? ex : string(str[:str])
end

function _rhs_to_args(ex::Expr)
    symorig, symgend = Any[], Symbol[]
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
            new = gensym(_ex_to_string(sym))
            push!(symorig, sym)
            push!(symgend, new)
            :(conj($new[$(ind...)]))
        elseif @capture(x, sym_[ind__])
            all(x -> !isa(x, Integer), ind) || error("NCON style is unsupported")
            new = gensym(_ex_to_string(sym))
            push!(symorig, sym)
            push!(symgend, new)
            :($new[$(ind...)])
        elseif x isa Number
            x
        else
            new = gensym(_ex_to_string(x))
            push!(symorig, x)
            push!(symgend, new)
            new
        end
    exreplaced = exparse(ex)
    return exreplaced, symorig, symgend
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
    MacroTools.postwalk(ex) do x
        if @capture(x, -(y__))
            if length(y) == 1
                x
            elseif hassym(first(y))
                first(y)
            elseif hassym(last(y))
                :(-$(last(y)))
            else
                @error y
            end
        elseif @capture(x, +(y__))
            y = filter(hassym, y)
            @assert length(y) == 1
            first(y)
        else
            x
        end
    end
end

function _gen_func(name, args, lhsind, rhs, opt = nothing)
    ex = if isnothing(lhsind)
        :($name[] := $rhs)
    else
        :($name[$(lhsind...)] := $rhs)
    end
    ex = if isnothing(opt)
        :(@inline $name($(args...)) = @tensor $ex)
    else
        :(@inline $name($(args...)) = @tensoropt $opt $ex)
    end
    return macroexpand(TensorOperations, ex)
end

function _gen_rule(name, args, lhsind, rhs, opt = nothing)
    @gensym Δlhssym
    Δlhs = if isnothing(lhsind)
        :($Δlhssym[])
    else
        :($Δlhssym[$(lhsind...)])
    end
    Δargs = []
    Δexargs = []
    for arg in args
        Δarg = gensym(arg)
#        Δsind = nothing
#        Δexarg = MacroTools.prewalk(rhs) do x
#            if @capture(x, conj($arg[Δsind__]))
#                error("TODO")
#            elseif @capture(x, $arg[Δsind__] | $arg)
#                :(conj($Δlhs))
#            else
#                x
#            end
#        end
        Δexarg = if isnothing(Δsind)
            if isnothing(opt)
                quote
                    @tensor $Δarg[] := $Δexarg
                    $Δarg = $Δarg[]
                end
            else
                quote
                    @tensoropt $opt $Δarg[] := $Δexarg
                    $Δarg = $Δarg[]
                end
            end
        else
            if isnothing(opt)
                :(@tensor $Δarg[$(Δsind...)] := $Δexarg)
            else
                :(@tensoropt $opt $Δarg[$(Δsind...)] := $Δexarg)
            end
        end
        push!(Δargs, Δarg)
        push!(Δexargs, macroexpand(TensorOperations, Δexarg))
    end

    @gensym valforw fncback
    backbody = Expr(:block, Δexargs..., :(return (ChainRulesCore.NO_FIELDS, $(Δargs...))))

    return quote
        function ChainRulesCore.rrule(::typeof($(name)), $(args...))
            $valforw = $(name)($(args...))
            $(fncback)($Δlhssym) = $backbody
            return ($valforw, $fncback)
        end
    end
end

function _nabla(ex::Expr)
    def = splitdef(ex)
    exfuncs = []
    exrules = []
    exbody = MacroTools.postwalk(def[:body]) do x
        if @capture(x, @tensor lhs_[lhsind__] := rhs_)
            which, opt = :assign, nothing
        elseif @capture(x, @tensor lhs_[lhsind__] += rhs_)
            which, opt = :pluseq, nothing
        elseif @capture(x, @tensor lhs_[lhsind__] -= rhs_)
            which, opt = :subteq, nothing
        elseif @capture(x, @tensoropt opt_ lhs_[lhsind__] := rhs_)
            which = :assign
        elseif @capture(x, @tensoropt opt_ lhs_[lhsind__] += rhs_)
            which = :pluseq
        elseif @capture(x, @tensoropt opt_ lhs_[lhsind__] -= rhs_)
            which = :subteq
        elseif @capture(x, @tensor lhs_[__] = rhs_ | @tensoropt opt_ lhs_[__] = rhs_)
            error("use assignment with `:=` instead of inplace operation with `=` for Zygote")
        else
            return x
        end
        rhsreplace, argsorig, argsdummy = _rhs_to_args(rhs)
        name = gensym(lhs)
        push!(exfuncs, _gen_func(name, argsdummy, lhsind, rhsreplace, opt))
        push!(exrules, _gen_rule(name, argsdummy, lhsind, rhsreplace, opt))
        if which == :assign
            return :($lhs = $name($(argsorig...)))
        elseif which == :pluseq # use x += y instead of x .+= y for Zygote
            return :($lhs += $name($(argsorig...)))
        elseif which == :subteq # use x -= y instead of x .-= y for Zygote
            return :($lhs -= $name($(argsorig...)))
        end
    end
    def[:body] = exbody
    exout = Expr(:block, combinedef(def), exfuncs..., exrules...)
    return esc(exout)
end

macro ∇(ex)
    _nabla(ex)
end

end
