
#####
##### `Wirtinger`
#####
# TODO: Document the derivations that lead to all of these rules (see notes)

"""
    Wirtinger(primal::Union{Number,AbstractDifferential},
              conjugate::Union{Number,AbstractDifferential})

Return a `Wirtinger` instance with two directly accessible fields:

- `primal`: the value corresponding to `∂f/∂z * dz`
- `conjugate`: the value corresponding to `∂f/∂z̄ * dz̄`

This `Wirtinger` instance, as a whole, represents the complex differential `df`,
defined in the Wirtinger calculus as:

```
df = ∂f/∂z * dz + ∂f/∂z̄ * dz̄
```

This representation allows convenient derivative definitions for nonholomorphic
functions of complex variables. For example, consider the `@scalar_rule` for
`abs2`:

```
@scalar_rule(abs2(x), Wirtinger(x', x))
```
"""
struct Wirtinger{P,C} <: AbstractDifferential
    primal::P
    conjugate::C
    function Wirtinger(primal::Union{Number,AbstractDifferential},
                       conjugate::Union{Number,AbstractDifferential})
        return new{typeof(primal),typeof(conjugate)}(primal, conjugate)
    end
    function Wirtinger(primal, conjugate)
        error("`Wirtinger` only supports elements of type <: Union{Number,AbstractDifferential} for now")
    end
end

"""
    Wirtinger(primal::Real, conjugate::Real)

Return `add(primal, conjugate)`.

The Wirtinger calculus generally requires that downstream propagation mechanisms
have access to `∂f/∂z * dz` and `∂f/∂z̄ * dz` separately. However, if both of
these terms are real-valued, then downstream Wirtinger propagation mechanisms
resolve to the same mechanisms as real-valued calculus. In this case, the sum
in the differential `df = ∂f/∂z * dz + ∂f/∂z̄ * dz` can be computed eagerly and
a special `Wirtinger` representation is not needed.

Thus, this method primarily exists as an optimization.
"""
Wirtinger(primal::Real, conjugate::Real) = add(primal, conjugate)

extern(x::Wirtinger) = error("`Wirtinger` cannot be converted into an external type.")

Base.Broadcast.broadcastable(w::Wirtinger) = Wirtinger(broadcastable(w.primal),
                                                       broadcastable(w.conjugate))

Base.iterate(x::Wirtinger) = (x, nothing)
Base.iterate(::Wirtinger, ::Any) = nothing

function add_wirtinger(a::Wirtinger, b::Wirtinger)
    return Wirtinger(add(a.primal, b.primal), add(a.conjugate, b.conjugate))
end

add_wirtinger(a::Wirtinger, b) = add(a, Wirtinger(b, Zero()))
add_wirtinger(a, b::Wirtinger) = add(Wirtinger(a, Zero()), b)

function mul_wirtinger(a::Wirtinger, b::Wirtinger)
    new_primal = add(mul(a.primal, b.primal), mul(a.conjugate, conj(b.conjugate)))
    new_conjugate = add(mul(a.primal, b.conjugate), mul(a.conjugate, conj(b.primal)))
    return Wirtinger(new_primal, new_conjugate)
end

mul_wirtinger(a::Wirtinger, b) = mul(a, Wirtinger(b, Zero()))
mul_wirtinger(a, b::Wirtinger) = mul(Wirtinger(a, Zero()), b)
