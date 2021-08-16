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
            error("use assignment with `:=` instead of inplace operation with `=`")
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
        exrrule = genrrule(genargs..., isconjs, indsall)

        @eval mod $(macroexpand(TensorOperations, exfunc))
        @eval mod $(macroexpand(TensorOperations, exfrule))
        @eval mod $(macroexpand(TensorOperations, exrrule))

        if which == :assign
            return :($lhs = $(Core.eval(mod, symfunc))($(argsorig...)))
        elseif which == :pluseq  # use x += y instead of x .+= y for Zygote
            return :($lhs += $(Core.eval(mod, symfunc))($(argsorig...)))
        elseif which == :subteq  # use x -= y instead of x .-= y for Zygote
            return :($lhs -= $(Core.eval(mod, symfunc))($(argsorig...)))
        end
    end

    return esc(MacroTools.combinedef(def)), symfuncs
end

macro ∇(ex)
    ex, _ = _nabla(ex; mod=@__MODULE__)
    return ex
end

macro fn∇(i, ex)
    _, fn = _nabla(ex; mod=@__MODULE__)
    return fn[i]
end
