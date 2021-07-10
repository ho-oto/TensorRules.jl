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
    indsall::Vector{Union{Symbol,Expr}},
)
    indsall = [lhsind; indsall]

    @gensym Δlhssym Δlhssymproj
    Δlhs = isempty(lhsind) ? :(first($Δlhssym)) : :($Δlhssym[$(lhsind...)])
    projargsyms, projargdefs = Symbol[], Expr[]
    for arg in args
        @gensym projarg
        push!(projargsyms, projarg)
        push!(projargdefs, :(ProjectTo($arg)))
    end

    ∂args, ∂exargs, ∂exdefs = Symbol[], Expr[], Expr[]
    for (arg, isconj, projsym) in zip(args, isconjs, projargsyms)
        @gensym ∂arg
        rhsarg = make_scalar_first(make_only_product(rhs, arg))

        ind = Ref{Vector{Any}}()
        ∂exrhs = MacroTools.prewalk(rhsarg) do x  # assume to match only once
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
                :(function $∂fn($Δlhssym, $(args...))
                        @tensoropt $(opt[]) _[$(indtr...)] := $∂exrhs
                    end)
            else
                :($∂fn($Δlhssym, $(args...)) = @tensor _[$(indtr...)] := $∂exrhs)
            end
        else
            if isassigned(opt)
                :(function $∂fn($Δlhssym, $(args...))
                        return first(@tensoropt $(opt[]) _[] := $∂exrhs)
                    end)
            else
                :($∂fn($Δlhssym, $(args...)) = first(@tensor _[] := $∂exrhs))
            end
        end
        push!(∂exdefs, ∂exdef)
        if istensor
            @gensym ∂fnadd inplaced
            ∂exadd! = if isassigned(opt)
                :(
                    function $∂fnadd($inplaced, $Δlhssym, $(args...))
                        @tensoropt $(opt[]) $inplaced[$(indtr...)] += $∂exrhs
                    end
                )
            else
                :(
                    function $∂fnadd($inplaced, $Δlhssym, $(args...))
                        @tensor $inplaced[$(indtr...)] += $∂exrhs
                    end
                )
            end
            push!(∂exdefs, ∂exadd!)
        end

        ∂exarg = if istensor
            :(
                $∂arg = InplaceableThunk(
                    Thunk(() -> $projsym($∂fn($Δlhssym, $(args...)))),
                    $inplaced -> $∂fnadd($inplaced, $Δlhssym, $(args...)),
                )
            )
        else
            :($∂arg = Thunk(() -> $∂fn($Δlhssym, $(args...))))
        end

        push!(∂args, ∂arg)
        push!(∂exargs, ∂exarg)
    end

    @gensym valforw funcback
    backbody = Expr(
        :block,
        :($Δlhssym = $Δlhssymproj($Δlhssym)),
        ∂exdefs...,
        ∂exargs...,
        :(return (NoTangent(), $(∂args...))),
    )

    projargsym = Expr(:tuple, projargsyms...)
    projargdef = Expr(:tuple, projargdefs...)
    return quote
        function ChainRulesCore.rrule(::typeof($funcname), $(args...))
            $valforw = $(funcname)($(args...))
            $Δlhssymproj = ProjectTo($valforw)
            $projargsym = $projargdef
            $(funcback)($Δlhssym) = $backbody
            return ($valforw, $funcback)
        end
    end
end
