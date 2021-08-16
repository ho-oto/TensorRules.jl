function genfunc(
    funcname::Symbol, args::Vector{Symbol}, lhsind::Vector{Any}, rhs::Expr, opt::Ref{Expr}
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
    funcname::Symbol, args::Vector{Symbol}, lhsind::Vector{Any}, rhs::Expr, opt::Ref{Expr}
)
    ȧrgs, ṙhss = Symbol[], Expr[]
    for arg in args
        @gensym ȧrg
        ṙhs = make_scalar_first(make_only_product(rhs, arg))

        ṙhs = MacroTools.prewalk(ṙhs) do x  # assume to match only once
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
            ($(ȧrgs...),) = ChainRulesCore.unthunk.(($(ȧrgs...),))
            $val = $(funcname)($(args...))
            $pushbody
            return ($val, $v̇al::typeof($val))
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
    indsall::Vector{Union{Symbol,Expr}},
)
    indsall = [lhsind; indsall]

    @gensym Δlhs
    Δlhsex = isempty(lhsind) ? :(first($Δlhs)) : :($Δlhs[$(lhsind...)])

    projargs, projargexs = Symbol[], Expr[]
    ∂args, ∂argexs, ∂argdefs = Symbol[], Expr[], Expr[]
    for (arg, isconj) in zip(args, isconjs)
        @gensym ∂arg projarg
        push!(∂args, ∂arg)
        push!(projargs, projarg)
        push!(projargexs, :(ProjectTo($arg)))

        rhsarg = make_scalar_first(make_only_product(rhs, arg))

        ind = Ref{Vector{Any}}()
        ∂exrhs = MacroTools.prewalk(rhsarg) do x  # assume to match only once
            if @capture(x, $arg[_ind__])
                ind[] = _ind
                isconj ? Δlhsex : :(conj($Δlhsex))
            elseif @capture(x, $arg)
                isconj ? Δlhsex : :(conj($Δlhsex))
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
                    j = Expr(Symbol("'"), i)
                    while j ∈ [indsall; indtr]
                        j = Expr(Symbol("'"), j)
                    end
                    push!(indtr, j)
                    push!(
                        δs,
                        :(Array{eltype($arg)}(I, size($arg, $k), size($arg, $k))[$i, $j]),
                    )
                end
            end
            ∂exrhs = :(*($∂exrhs, $(δs...)))
        elseif istensor
            append!(indtr, ind[])
        end

        ∂exrhs = isconj ? ∂exrhs : :(conj($∂exrhs))

        @gensym ∂fn
        ∂exdef = if istensor
            if isassigned(opt)
                :(function $∂fn($Δlhs, $(args...))
                    @tensoropt $(opt[]) _[$(indtr...)] := $∂exrhs
                end)
            else
                :($∂fn($Δlhs, $(args...)) = @tensor _[$(indtr...)] := $∂exrhs)
            end
        else
            if isassigned(opt)
                :(function $∂fn($Δlhs, $(args...))
                    return first(@tensoropt $(opt[]) _[] := $∂exrhs)
                end)
            else
                :($∂fn($Δlhs, $(args...)) = first(@tensor _[] := $∂exrhs))
            end
        end
        push!(∂argdefs, ∂exdef)
        if istensor
            @gensym ∂fnadd!! inplaced
            ∂exadd! = if isassigned(opt)
                :(function $∂fnadd!!($inplaced, $Δlhs, $(args...))
                    @tensoropt $(opt[]) $inplaced[$(indtr...)] += $∂exrhs
                end)
            else
                :(function $∂fnadd!!($inplaced, $Δlhs, $(args...))
                    @tensor $inplaced[$(indtr...)] += $∂exrhs
                end)
            end
            push!(∂argdefs, ∂exadd!)
        end

        ∂exarg = if istensor
            :(
                $∂arg = InplaceableThunk(
                    $inplaced -> $∂fnadd!!($inplaced, $Δlhs, $(args...)),
                    Thunk(() -> $projarg($∂fn($Δlhs, $(args...)))),
                )
            )
        else
            :($∂arg = Thunk(() -> $∂fn($Δlhs, $(args...))))
        end

        push!(∂argexs, ∂exarg)
    end

    @gensym lhs pullback
    return quote
        function ChainRulesCore.rrule(::typeof($funcname), $(args...))
            $lhs = $(funcname)($(args...))
            ($(projargs...),) = ($(projargexs...),)
            function $(pullback)($Δlhs)
                return $(Expr(
                    :block,
                    :($Δlhs = Array(ChainRulesCore.unthunk($Δlhs))),
                    ∂argdefs...,
                    ∂argexs...,
                    :(return (ChainRulesCore.NoTangent(), $(∂args...))),
                ))
            end
            return ($lhs, $pullback)
        end
    end
end
