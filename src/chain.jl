# TODO: turn various comments in this file into docstrings

#####
##### `Chain`/`AbstractChainable`
#####
#=
This file defines a custom algebra for chain rule evaluation that factors
complex support, bundle support, zero-elision, etc. into nicely separated
parts.

The main two operations of this algebra are:

`add`: linearly combine partial derivatives (i.e. the `+` part of the
       multivariate chain rule)

`mul`: multiply partial derivatives by a perturbation/sensitivity coefficient
       (i.e. the `*` part of the multivariate chain rule)

Valid arguments to these operations are `T` where `T<:AbstractChainable`, or
where `T` has `broadcast`, `+`, and `*` implementations.

A bunch of the operations in this file have kinda monad-y "fallthrough"
implementations; each step handles an element of the algebra before dispatching
to the next step. This way, we don't need to implement extra machinery just to
resolve ambiguities (e.g. a promotion mechanism).
=#

Cassette.@context ChainableContext

const CHAINABLE_CONTEXT = ChainableContext()

struct Chain{F}
    f::F
end

@inline (c::Chain)(args...) = Cassette.overdub(CHAINABLE_CONTEXT, c.f, args...)

#=
Here are some examples using `@chain` to implement forward- and reverse-mode
chain rules for an intermediary function of the form:

    y₁, y₂ = f(x₁, x₂)

Forward-Mode:

    @chain(∂y₁_∂x₁, ∂y₁_∂x₂)
    @chain(∂y₂_∂x₁, ∂y₂_∂x₂)

    # expands to:
    Chain((ẏ₁, ẋ₁, ẋ₂) -> add(ẏ₁, mul(@thunk(∂y₁_∂x₁), ẋ₁), mul(@thunk(∂y₁_∂x₂), ẋ₂)))
    Chain((ẏ₂, ẋ₁, ẋ₂) -> add(ẏ₂, mul(@thunk(∂y₂_∂x₁), ẋ₁), mul(@thunk(∂y₂_∂x₂), ẋ₂)))

Reverse-Mode:

    @chain(adjoint(∂y₁_∂x₁), adjoint(∂y₂_∂x₁))
    @chain(adjoint(∂y₁_∂x₂), adjoint(∂y₂_∂x₂))

    # expands to:
    Chain((x̄₁, ȳ₁, ȳ₂) -> add(x̄₁, mul(@thunk(adjoint(∂y₁_∂x₁)), ȳ₁), mul(@thunk(adjoint(∂y₂_∂x₁)), ȳ₂)))
    Chain((x̄₂, ȳ₁, ȳ₂) -> add(x̄₂, mul(@thunk(adjoint(∂y₁_∂x₂)), ȳ₁), mul(@thunk(adjoint(∂y₂_∂x₂)), ȳ₂)))
=#
macro chain(∂s...)
    δs = [Symbol(string(:δ, i)) for i in 1:length(∂s)]
    Δs = Any[]
    for i in 1:length(∂s)
        ∂ = esc(∂s[i])
        push!(Δs, :(mul(@thunk($∂), $(δs[i]))))
    end
    return :(Chain((δ₀, $(δs...)) -> add(δ₀, $(Δs...))))
end

abstract type AbstractChainable end

Cassette.execute(::ChainableCtx, ::typeof(*), a::AbstractChainable, b::AbstractChainable) = mul(a, b)
Cassette.execute(::ChainableCtx, ::typeof(*), a::AbstractChainable, b) = mul(a, b)
Cassette.execute(::ChainableCtx, ::typeof(*), a, b::AbstractChainable) = mul(a, b)
Cassette.execute(::ChainableCtx, ::typeof(mul), a, b) = mul(a, b)

@inline mul(a, b) = mul_zero(a, b)
@inline mul_zero(a, b) = mul_one(a, b)
@inline mul_one(a, b) = mul_thunk(a, b)
@inline mul_thunk(a, b) = mul_wirtinger(a, b)
@inline mul_wirtinger(a, b) = mul_cast(a, b)
@inline mul_cast(a, b) = mul_fallback(a, b)
@inline mul_fallback(a, b) = a * b

Cassette.execute(::ChainableCtx, ::typeof(+), a::AbstractChainable, b::AbstractChainable) = add(a, b)
Cassette.execute(::ChainableCtx, ::typeof(+), a::AbstractChainable, b) = add(a, b)
Cassette.execute(::ChainableCtx, ::typeof(+), a, b::AbstractChainable) = add(a, b)
Cassette.execute(::ChainableCtx, ::typeof(add), a, b) = add(a, b)

@inline add(a, b) = add_zero(a, b)
@inline add_zero(a, b) = add_one(a, b)
@inline add_one(a, b) = add_thunk(a, b)
@inline add_thunk(a, b) = add_wirtinger(a, b)
@inline add_wirtinger(a, b) = add_cast(a, b)
@inline add_cast(a, b) = add_fallback(a, b)
@inline add_fallback(a, b) = broadcasted(+, a, b)

Cassette.execute(::ChainableCtx, ::typeof(adjoint), x) = _adjoint(x)
Cassette.execute(::ChainableCtx, ::typeof(_adjoint), x) = _adjoint(x)

_adjoint(x) = adjoint(x)
_adjoint(x::Base.Broadcast.Broadcasted) = broadcasted(adjoint, x)

#####
##### `Thunk`
#####

struct Thunk{F} <: AbstractChainable
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

(t::Thunk{F})() where {F} = (t.f)()

_adjoint(t::Thunk) = @thunk(_adjoint(t()))

Base.Broadcast.materialize(t::Thunk) = @thunk(materialize(t()))

unthunk(x) = x
unthunk(t::Thunk) = t()

#####
##### `Zero`/`DNE`
#####

struct Zero <: AbstractChainable end

_adjoint(::Zero) = Zero()

Base.Broadcast.materialize(::Zero) = false

mul_zero(::Zero, ::Zero) = Zero()
mul_zero(::Zero, ::Any) = Zero()
mul_zero(::Any, ::Zero) = Zero()

add_zero(::Zero, ::Zero) = Zero()
add_zero(::Zero, b) = unthunk(b)
add_zero(a, ::Zero) = unthunk(a)

#=
Equivalent to `Zero` for the purposes of propagation (i.e. partial
derivatives which don't exist simply do not contribute to a rule's total
derivative).
=#

const DNE = Zero

#=
TODO: How should we really handle the above? This is correct w.r.t. propagator
algebra; even if an actual new type `DNE <: AbstractChainable` was defined,
all the rules would be the same. Furthermore, users wouldn't be able to detect
many differences, since `DNE` must materialize to `materialize(Zero())`. Thus,
it seems like a derivative's `DNE`-ness should be exposed to users in a way
that's just unrelated to the chain rule algebra. Conversely, we want to
minimize the amount of special-casing needed for users writing higher-level
rule definitions/fallbacks, or else things will get unwieldy...
=#

#####
##### `One`
#####

struct One <: AbstractChainable end

_adjoint(::One) = One()

Base.Broadcast.materialize(::One) = true

mul_one(::One, ::One) = One()
mul_one(::One, b) = unthunk(b)
mul_one(a, ::One) = unthunk(a)

add_one(a::One, b::One) = add(materialize(a), materialize(b))
add_one(a::One, b) = add(materialize(a), b)
add_one(a, b::One) = add(a, materialize(b), b)

#####
##### `Wirtinger`
#####

struct Wirtinger{P, C} <: AbstractChainable
    primal::P
    conjugate::C
end

# TODO: check this against conjugation rule in notes
_adjoint(w::Wirtinger) = Wirtinger(adjoint(w.primal), adjoint(w.conjugate))

function Base.Broadcast.materialize(w::Wirtinger)
    return Wirtinger(materialize(w.primal), materialize(w.conjugate))
end

# TODO: document derivation that leads to this rule (see notes)
function _mul_wirtinger(a::Wirtinger, b::Wirtinger)
    new_primal = add(mul(a.primal, b.primal), mul(a.conjugate, adjoint(b.conjugate)))
    new_conjugate = add(mul(a.primal, b.conjugate), mul(a.conjugate, adjoint(b.primal)))
    return Wirtinger(new_primal, new_conjugate)
end

mul_wirtinger(a::Wirtinger, b) = Wirtinger(mul(a.primal, b), mul(a.conjugate, b))
mul_wirtinger(a, b::Wirtinger) = Wirtinger(mul(a, b.primal), mul(a, b.conjugate))

function add_wirtinger(a::Wirtinger, b::Wirtinger)
    return Wirtinger(add(a.primal, b.primal), add(a.conjugate, b.conjugate))
end

add_wirtinger(a::Wirtinger, b) = Wirtinger(add(a.primal, b), a.conjugate)
add_wirtinger(a, b::Wirtinger) = Wirtinger(add(a, b.primal), b.conjugate)

#####
##### `Cast`
#####

struct Cast{V} <: AbstractChainable
    value::V
end

_adjoint(c::Cast) = Cast(broadcasted(adjoint, c.value))

Base.Broadcast.materialize(c::Cast) = materialize(c.value)

mul_cast(a::Cast, b::Cast) = broadcasted(*, a.value, b.value)
mul_cast(a::Cast, b) = broadcasted(*, a.value, b)
mul_cast(a, b::Cast) = broadcasted(*, a, b.value)

#####
##### `MaterializeInto`
#####

struct MaterializeInto{S} <: AbstractChainable
    storage::S
    increment::Bool
    function MaterializeInto(storage, increment::Bool = true)
        return new{typeof(storage)}(storage, increment)
    end
end

function add(a::MaterializeInto, b)
    _materialize!(a.storage, a.increment ? add(a.storage, b) : b)
    return a
end

function _materialize!(a::Wirtinger, b::Wirtinger)
    materialize!(a.primal, b.primal)
    materialize!(a.conjugate, b.conjugate)
    return a
end

function _materialize!(a::Wirtinger, b)
    materialize!(a.primal, b)
    materialize!(a.conjugate, Zero())
    return a
end

function _materialize!(a, b::Wirtinger)
    return error("cannot `materialize!` `Wirtinger` into non-`Wirtinger`")
end

_materialize!(a, b) = materialize!(a, b)
