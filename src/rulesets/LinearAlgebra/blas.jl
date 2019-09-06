#=
These implementations were ported from the wonderful DiffLinearAlgebra
package (https://github.com/invenia/DiffLinearAlgebra.jl).
=#

using LinearAlgebra: BlasFloat

_zeros(x) = fill!(similar(x), zero(eltype(x)))

#####
##### `BLAS.dot`
#####

frule(::typeof(BLAS.dot), x, y) = frule(dot, x, y)

rrule(::typeof(BLAS.dot), x, y) = rrule(dot, x, y)

function rrule(::typeof(BLAS.dot), n, X, incx, Y, incy)
    Ω = BLAS.dot(n, X, incx, Y, incy)
    function blas_dot_pullback(ΔΩ)
        if ΔΩ isa Zero
            ∂X = Zero()
            ∂Y = Zero()
        else
            ΔΩ = extern(ΔΩ)
            ∂X = @thunk scal!(n, ΔΩ, blascopy!(n, Y, incy, _zeros(X), incx), incx)
            ∂Y = @thunk scal!(n, ΔΩ, blascopy!(n, X, incx, _zeros(Y), incy), incy)
        end
        return (NO_FIELDS, DNE(), ∂X, DNE(), ∂Y, DNE())
    end
    return Ω, blas_dot_pullback
end

#####
##### `BLAS.nrm2`
#####

function frule(::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    function nrm2_pushforward(_, Δx)
        return sum(Δx * cast(@thunk(x * inv(Ω))))
    end
    return Ω, nrm2_pushforward
end

function rrule(::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    function nrm2_pullback(ΔΩ)
        return NO_FIELDS, @thunk(ΔΩ * @thunk(x * inv(Ω)))
    end
    return Ω, nrm2_pullback
end

function rrule(::typeof(BLAS.nrm2), n, X, incx)
    Ω = BLAS.nrm2(n, X, incx)
    function nrm2_pullback(ΔΩ)
        if ΔΩ isa Zero
            ∂X = Zero()
        else
            ΔΩ = extern(ΔΩ)
            ∂X = scal!(n, ΔΩ / Ω, blascopy!(n, X, incx, _zeros(X), incx), incx)
        end
        return (NO_FIELDS, DNE(), ∂X, DNE())
    end

    return Ω, nrm2_pullback
end

#####
##### `BLAS.asum`
#####

function frule(::typeof(BLAS.asum), x)
    return BLAS.asum(x), (_, Δx) -> sum(cast(sign, x) * Δx)
end

function rrule(::typeof(BLAS.asum), x)
    return BLAS.asum(x), ΔΩ -> (NO_FIELDS, @thunk(ΔΩ * cast(sign, x)))
end

function rrule(::typeof(BLAS.asum), n, X, incx)
    Ω = BLAS.asum(n, X, incx)
    function asum_pullback(ΔΩ)
        if ΔΩ isa Zero
            ∂X = Zero()
        else
            ΔΩ = extern(ΔΩ)
            ∂X = @thunk scal!(
                n, ΔΩ,
                blascopy!(n, sign.(X), incx, _zeros(X), incx),
                incx
            )
        end
        return (NO_FIELDS, DNE(), ∂X, DNE())
    end
    return Ω, asum_pullback
end

#####
##### `BLAS.gemv`
#####

function rrule(::typeof(gemv), tA::Char, α::T, A::AbstractMatrix{T},
               x::AbstractVector{T}) where T<:BlasFloat
    y = gemv(tA, α, A, x)
    function gemv_pullback(ȳ)
        if uppercase(tA) === 'N'
            ∂A = @thunk(α * ȳ * x', (Ā, ȳ) -> ger!(α, ȳ, x, Ā))
            ∂x = @thunk(gemv('T', α, A, ȳ), (x̄, ȳ) -> gemv!('T', α, A, ȳ, one(T), x̄))
        else
            ∂A = @thunk(α * x * ȳ', (Ā, ȳ) -> ger!(α, x, ȳ, Ā))
            ∂x = @thunk(gemv('N', α, A, ȳ), (x̄, ȳ) -> gemv!('N', α, A, ȳ, one(T), x̄))
        end
        return (NO_FIELDS, DNE(), @thunk(dot(ȳ, y) / α), ∂A, ∂x)
    end
    return y, gemv_pullback
end

function rrule(::typeof(gemv), tA::Char, A::AbstractMatrix{T},
               x::AbstractVector{T}) where T<:BlasFloat
    y, inner_pullback = rrule(gemv, tA, one(T), A, x)
    function gemv_pullback(Ȳ)
        (_, dtA, _, dA, dx) = inner_pullback(Ȳ)
        return (NO_FIELDS, dtA, dA, dx)
    end
    return y, gemv_pullback
end

#####
##### `BLAS.gemm`
#####

function rrule(::typeof(gemm), tA::Char, tB::Char, α::T,
               A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:BlasFloat
    C = gemm(tA, tB, α, A, B)
    function gemv_pullback(C̄)
        β = one(T)
        if uppercase(tA) === 'N'
            if uppercase(tB) === 'N'
                ∂A = @thunk(gemm('N', 'T', α, C̄, B))
                ∂A_update = Ā -> gemm!('N', 'T', α, C̄, B, β, Ā)
                ∂B = @thunk(gemm('T', 'N', α, A, C̄))
                ∂B_update = B̄ -> gemm!('T', 'N', α, A, C̄, β, B̄)
            else
                ∂A = @thunk(gemm('N', 'N', α, C̄, B))
                ∂A_update = Ā -> gemm!('N', 'N', α, C̄, B, β, Ā)
                ∂B = @thunk(gemm('T', 'N', α, C̄, A))
                ∂B_update = B̄ -> gemm!('T', 'N', α, C̄, A, β, B̄)
            end
        else
            if uppercase(tB) === 'N'
                ∂A = @thunk(gemm('N', 'T', α, B, C̄))
                ∂A_update = Ā -> gemm!('N', 'T', α, B, C̄, β, Ā)
                ∂B = @thunk(gemm('N', 'N', α, A, C̄))
                ∂B_update = B̄ -> gemm!('N', 'N', α, A, C̄, β, B̄)
            else
                ∂A = @thunk(gemm('T', 'T', α, B, C̄))
                ∂A_update = Ā -> gemm!('T', 'T', α, B, C̄, β, Ā)
                ∂B = @thunk(gemm('T', 'T', α, C̄, A))
                ∂A_update = B̄ -> gemm!('T', 'T', α, C̄, A, β, B̄)
            end
        end
        # TODO: ∂A_update and ∂B_update. Requires working out update rules in the post #30 world
        return (NO_FIELDS, DNE(), DNE(), @thunk(dot(C̄, C) / α), ∂A, ∂B)
    end
    return C, gemv_pullback
end

function rrule(::typeof(gemm), tA::Char, tB::Char,
               A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:BlasFloat
    C, inner_pullback = rrule(gemm, tA, tB, one(T), A, B)
    function gemv_pullback(Ȳ)
        (_, dtA, dtB, _, dA, dB) = inner_pullback(Ȳ)
        return (NO_FIELDS, dtA, dtB, dA, dB)
    end
    return C, gemm_pullback
end
