#####
##### `@rule`s
#####

@rule(dot(x, y), (cast(y), cast(x)))
@rule(sum(x), One())

#####
##### custom rules
#####

# inv

function frule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @memoize(-Ω)
    return Ω, Chain((Ω̇, ẋ) -> Ω̇ + m * ẋ * Ω)
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @memoize(-Ω)
    return Ω, Chain((x̄, Ω̄) -> x̄ + m' * Ω̄ * Ω')
end

# det

function frule(::typeof(det), x)
    Ω, m = det(x), @memoize(inv(x))
    return Ω, Chain((Ω̇, ẋ) -> Ω̇ + Ω * tr(m * ẋ))
end

function rrule(::typeof(det), x)
    Ω, m = det(x), @memoize(inv(x)')
    return Ω, Chain((x̄, Ω̄) -> x̄ + Ω * Ω̄ * m)
end

# logdet

function frule(::typeof(LinearAlgebra.logdet), x)
    Ω, m = logdet(x), @memoize(inv(x))
    return Ω, Chain((Ω̇, ẋ) -> Ω̇ + tr(m * ẋ))
end

function rrule(::typeof(LinearAlgebra.logdet), x)
    Ω, m = logdet(x), @memoize(inv(x)')
    return Ω, Chain((x̄, Ω̄) -> x̄ + Ω̄ * m)
end

# trace

frule(::typeof(tr), x) = (tr(x), (Ω̇, ẋ) -> add(Ω̇, Diagonal(materialize(ẋ))))

rrule(::typeof(tr), x) = (tr(x), (x̄, Ω̄) -> add(x̄, Diagonal(materialize(Ω̄))))

function ChainRules.frule(::typeof(fft), A)
    Ω = fft(A)
    return Ω, Chain((Ω̇, Ȧ) -> Ω̇ + fft(Ȧ))
end

# function rrule(::typeof(fft), A)
#     Ω = fft(A)
#     return Ω, Chain((Ā, Ω̄) -> Ā + ????)
# end
