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
                    j = :($i')
                    while j ∈ [indsall; indtr]
                        j = :($j')
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
        :(return (NoTangent(), $(∂args...))),
    )
    zerobody = :((NoTangent(), $(fill(ZeroTangent(), length(∂args))...)))

    return quote
        function ChainRulesCore.rrule(::typeof($funcname), $(args...))
            $valforw = $(funcname)($(args...))
            $(funcback)($Δlhssym) = $backbody
            $(funcback)(::AbstractZero) = $zerobody
            return ($valforw, $funcback)
        end
    end
end
