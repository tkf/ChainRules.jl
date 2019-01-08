# TODO: turn various comments in this file into docstrings

#####
##### `AbstractDifferential`
#####
#=
This file defines a custom algebra for chain rule evaluation that factors
complex support, bundle support, zero-elision, etc. into nicely separated
parts.

The main three operations of this algebra are:

`add`: linearly combine partial derivatives (i.e. the `+` part of the
        multivariate chain rule)

`mul`: multiply partial derivatives by a perturbation/sensitivity coefficient
        (i.e. the `*` part of the multivariate chain rule)

`adjoint`: complex conjugate transpose

`dechainify`: convert an `AbstractDifferential`

Valid arguments to these operations are `T` where `T<:AbstractDifferential`, or
where `T` has proper `broadcast`, `+`, `*`, and `adjoint` implementations.

This `AbstractDifferential` algebra has a monad-y "fallthrough" implementation;
each step handles an element of the algebra before dispatching to the next step.
This way, we don't need to implement promotion/conversion rules between subtypes
of `AbstractDifferential` to resolve potential ambiguities.
=#

abstract type AbstractDifferential end

const PRECEDENCE_LIST = [:accumulated, :wirtinger, :casted, :zero, :dne,
                         :one, :thunk, :memoize, :fallback]

global defs = Expr(:block)

let previous_add_name = :add, previous_mul_name = :mul
    for name in PRECEDENCE_LIST
        next_add_name = Symbol(string(:add_, name))
        next_mul_name = Symbol(string(:mul_, name))
        push!(defs.args, quote
            @inline $(previous_add_name)(a, b) = $(next_add_name)(a, b)
            @inline $(previous_mul_name)(a, b) = $(next_mul_name)(a, b)
        end)
        previous_add_name = next_add_name
        previous_mul_name = next_mul_name
    end
end

eval(defs)

@inline add_fallback(a, b) = a + b

@inline mul_fallback(a, b) = a * b

#####
##### `Accumulated`
#####

struct Accumulated{S} <: AbstractDifferential
    storage::S
    increment::Bool
    function Accumulated(storage, increment::Bool = true)
        return new{typeof(storage)}(storage, increment)
    end
end

accumulated_materialize!(x, y) = materialize!(x, y)

function add_accumulated(a::Accumulated, b::Accumulated)
    error("")
end

function add_accumulated(a::Accumulated, b)
    accumulated_materialize!(a.storage, broadcastable(a.increment ? add(cast(a.storage), b) : b))
    return a
end

function add_accumulated(a, b::Accumulated)
    accumulated_materialize!(b.storage, broadcastable(b.increment ? add(a, cast(b.storage)) : a))
    return b
end

function mul_accumulated(a::Accumulated, b::Accumulated)
    error("")
end

function mul_accumulated(a::Accumulated, b)
    error("")
end

function mul_accumulated(a, b::Accumulated)
    error("")
end

#####
##### `Wirtinger`
#####
# TODO: Document the derivations that lead to all of these rules (see notes)

struct Wirtinger{P,C} <: AbstractDifferential
    primal::P
    conjugate::C
end

# TODO: is this "optimization" correct?
function Wirtinger(primal::Union{Real,AbstractArray{<:Real}},
                   conjugate::Union{Real,AbstractArray{<:Real}})
    return add(primal, conjugate)
end

Base.Broadcast.broadcastable(w::Wirtinger) = Wirtinger(broadcastable(w.primal),
                                                       broadcastable(w.conjugate))

Base.adjoint(w::Wirtinger) = Wirtinger(adjoint(w.primal), adjoint(w.conjugate))

function accumulated_materialize!(a, b::Wirtinger)
    error("")
end

function accumulated_materialize!(a::Wirtinger, b::Wirtinger)
    materialize!(a.primal, b.primal)
    materialize!(a.conjugate, b.conjugate)
end

function accumulated_materialize!(a::Wirtinger, b)
    error("")
end

function add_wirtinger(a::Wirtinger, b::Wirtinger)
    return Wirtinger(add(a.primal, b.primal), add(a.conjugate, b.conjugate))
end

add_wirtinger(a::Wirtinger, b) = add(a, Wirtinger(b, Zero()))
add_wirtinger(a, b::Wirtinger) = add(Wirtinger(a, Zero()), b)

function mul_wirtinger(a::Wirtinger, b::Wirtinger)
    new_primal = add(mul(a.primal, b.primal), mul(a.conjugate, adjoint(b.conjugate)))
    new_conjugate = add(mul(a.primal, b.conjugate), mul(a.conjugate, adjoint(b.primal)))
    return Wirtinger(new_primal, new_conjugate)
end

mul_wirtinger(a::Wirtinger, b) = mul(a, Wirtinger(b, Zero()))
mul_wirtinger(a, b::Wirtinger) = mul(Wirtinger(a, Zero()), b)

#####
##### `Thunk`
#####

struct Thunk{F} <: AbstractDifferential
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

@inline (t::Thunk{F})() where {F} = t.f()

Base.Broadcast.broadcastable(x::Thunk) = broadcastable(x())

Base.adjoint(x::Thunk) = @thunk(adjoint(x()))

add_thunk(a::Thunk, b::Thunk) = add(a(), b())
add_thunk(a::Thunk, b) = add(a(), b)
add_thunk(a, b::Thunk) = add(a, b())

mul_thunk(a::Thunk, b::Thunk) = mul(a(), b())
mul_thunk(a::Thunk, b) = mul(a(), b)
mul_thunk(a, b::Thunk) = mul(a, b())

#####
##### `Memoize`
#####

struct Memoize{F,R} <: AbstractDifferential
    thunk::Thunk{F}
    ret::Ref{R}
end

function Memoize(thunk::Thunk)
    R = Core.Compiler.return_type(thunk, ()) # XXX danger zone!
    return Memoize(thunk, Ref{R}())
end

macro memoize(body)
    return :(Memoize(@thunk($(esc(body)))))
end

function (m::Memoize{F,R})()::R where {F, R}
    if !isassigned(m.ret)
        m.ret[] = m.thunk()
    end
    return m.ret[]::R
end

Base.Broadcast.broadcastable(x::Memoize) = broadcastable(x())

Base.adjoint(x::Memoize) = @thunk(adjoint(x()))

add_memoize(a::Memoize, b::Memoize) = add(a(), b())
add_memoize(a::Memoize, b) = add(a(), b)
add_memoize(a, b::Memoize) = add(a, b())

mul_memoize(a::Memoize, b::Memoize) = mul(a(), b())
mul_memoize(a::Memoize, b) = mul(a(), b)
mul_memoize(a, b::Memoize) = mul(a, b())

#####
##### `Zero`
#####

struct Zero <: AbstractDifferential end

Base.Broadcast.broadcastable(::Zero) = Ref(Zero())

Base.adjoint(::Zero) = Zero()

add_zero(::Zero, ::Zero) = Zero()
add_zero(::Zero, b) = b
add_zero(a, ::Zero) = a

mul_zero(::Zero, ::Zero) = Zero()
mul_zero(::Zero, ::Any) = Zero()
mul_zero(::Any, ::Zero) = Zero()

Base.convert(::Type{T}, ::Zero) where {T<:Number} = convert(T, false)

#####
##### `DNE`
#####

struct DNE <: AbstractDifferential end

Base.Broadcast.broadcastable(::DNE) = Ref(DNE())

Base.adjoint(::DNE) = DNE()

add_dne(::DNE, ::DNE) = DNE()
add_dne(::DNE, b) = b
add_dne(a, ::DNE) = a

mul_dne(::DNE, ::DNE) = DNE()
mul_dne(::DNE, ::Any) = DNE()
mul_dne(::Any, ::DNE) = DNE()

Base.convert(::Type{T}, ::DNE) where {T<:Number} = convert(T, false)

#####
##### `One`
#####

struct One <: AbstractDifferential end

Base.Broadcast.broadcastable(::One) = Ref(One())

Base.adjoint(::One) = One()

add_one(a::One, b::One) = add(true, true)
add_one(a::One, b) = add(true, b)
add_one(a, b::One) = add(a, true)

mul_one(::One, ::One) = One()
mul_one(::One, b) = b
mul_one(a, ::One) = a

Base.convert(::Type{T}, ::One) where {T<:Number} = convert(T, true)

#####
##### `Casted`
#####

struct Casted{V} <: AbstractDifferential
    value::V
end

cast(x) = Casted(x)
cast(f, args...) = Casted(broadcasted(f, args...))

Base.Broadcast.broadcastable(x::Casted) = x.value

Base.adjoint(x::Casted) = cast(adjoint, x.value)

add_casted(a::Casted, b::Casted) = Casted(broadcasted(add, a.value, b.value))
add_casted(a::Casted, b) = Casted(broadcasted(add, a.value, b))
add_casted(a, b::Casted) = Casted(broadcasted(add, a, b.value))

mul_casted(a::Casted, b::Casted) = Casted(broadcasted(mul, a.value, b.value))
mul_casted(a::Casted, b) = Casted(broadcasted(mul, a.value, b))
mul_casted(a, ::Casted) = Casted(broadcasted(mul, a, b.value))
