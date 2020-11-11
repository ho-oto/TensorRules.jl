function rhs_to_args(ex::Expr)
    indsall = Union{Symbol,Expr}[]
    symorig, symgend, isconjs = Union{Symbol,Expr}[], Symbol[], Bool[]

    function exparse(x, isconj)
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
