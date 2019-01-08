#####
##### `Chain`
#####

Cassette.@context ChainContext

const CHAIN_CONTEXT = Cassette.disablehooks(ChainContext())

Cassette.overdub(::ChainContext, ::typeof(+), a, b) = add(a, b)
Cassette.overdub(::ChainContext, ::typeof(*), a, b) = mul(a, b)

Cassette.overdub(::ChainContext, ::typeof(add), a, b) = add(a, b)
Cassette.overdub(::ChainContext, ::typeof(mul), a, b) = mul(a, b)

struct Chain{F}
    f::F
end

@inline (chain::Chain{F})(args...) where {F} = Cassette.overdub(CHAIN_CONTEXT, chain.f, args...)

#####
##### `@chain`
#####

#=
Here are some examples using `@chain` to implement forward- and reverse-mode
chain rules for an intermediary function of the form:

    y₁, y₂ = f(x₁, x₂)

Forward-Mode:

    @chain(∂y₁_∂x₁, ∂y₁_∂x₂)
    @chain(∂y₂_∂x₁, ∂y₂_∂x₂)

    # expands to:
    (ẏ₁, ẋ₁, ẋ₂) -> ẏ₁ + @thunk(∂y₁_∂x₁) * ẋ₁ + @thunk(∂y₁_∂x₂) * ẋ₂
    (ẏ₂, ẋ₁, ẋ₂) -> ẏ₂ + @thunk(∂y₂_∂x₁) * ẋ₁ + @thunk(∂y₂_∂x₂) * ẋ₂

Reverse-Mode:

    @chain(adjoint(∂y₁_∂x₁), adjoint(∂y₂_∂x₁))
    @chain(adjoint(∂y₁_∂x₂), adjoint(∂y₂_∂x₂))

    # expands to:
    (x̄₁, ȳ₁, ȳ₂) -> add(x̄₁, mul(@thunk(adjoint(∂y₁_∂x₁)), ȳ₁), mul(@thunk(adjoint(∂y₂_∂x₁)), ȳ₂))
    (x̄₂, ȳ₁, ȳ₂) -> add(x̄₂, mul(@thunk(adjoint(∂y₁_∂x₂)), ȳ₁), mul(@thunk(adjoint(∂y₂_∂x₂)), ȳ₂))

Some notation used here:
- `Δᵢ`: a seed (perturbation/sensitivity), i.e. the result of a chain rule evaluation
- `∂ᵢ`: a partial derivative to be multiplied by `Δᵢ` as part of chain rule evaluation
=#
macro chain(∂s...)
    Δs = [Symbol(string(:Δ, i)) for i in 1:length(∂s)]
    ∂Δs = [:(*(@thunk($(esc(∂s[i]))), $(Δs[i]))) for i in 1:length(∂s)]
    return :(Chain((Δ₀, $(Δs...)) -> +(Δ₀, $(∂Δs...))))
end
