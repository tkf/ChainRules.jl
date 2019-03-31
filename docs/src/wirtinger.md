# Complex-Valued Derivatives

Step-by-step primal calculation:

```julia
z₁ = g(z₀)
z₂ = f(z₁)
```

Wirtinger transformation:

```julia
# define:

g(z) = G(z, z̄)
f(z) = F(z, z̄)

# thus:

z₁ = g(z₀)
z₂ = f(z₁)

# so that the primal calculation can be rewritten as:

z₁, z̄₁ = G(z₀, z̄₀)
z₂, z̄₂ = F(z₁, z̄₁)
```

Step-by-step forward-mode differential propagation (derived by mechanically following the usual "real-valued scalar calculus"):

```julia
dz₀ = rand(Complex{Float64}) # from hypothetical upstream calculation
dz̄₀ = rand(Complex{Float64}) # from hypothetical upstream calculation

dz₁ = ∂z₁_∂z₀ * dz₀ + ∂z₁_∂z̄₀ * dz̄₀
dz̄₁ = ∂z̄₁_∂z₀ * dz₀ + ∂z̄₁_∂z̄₀ * dz̄₀

dz₂ = ∂z₂_∂z₁ * dz₁ + ∂z₂_∂z̄₁ * dz̄₁
dz̄₂ = ∂z̄₂_∂z₁ * dz₁ + ∂z̄₂_∂z̄₁ * dz̄₁
```

<!-- Proposed rule:

```julia
Base.conj(x::Wirtinger) = Wirtinger(conj(x.conjugate), conj(x.primal))

# (∂/∂z, ∂/∂z̄)(a∘b) = ∂a/∂b * (∂b/∂z, ∂b/∂z̄) + ∂a/∂b̄ * (∂b̄/∂z, ∂b̄/∂z̄)
function mul_wirtinger(a::Wirtinger, b::Wirtinger)
    return add(mul(a.primal, b), mul(a.conjugate, conj(b)))
end
```

In action:

```julia
w₀ = Wirtinger(dz₀, dz̄₀)

# doesn't work
w₁ = Wirtinger(∂z₁_∂z₀, ∂z₁_∂z̄₀) * w₀
   = ∂z₁_∂z₀ * w₀ + ∂z₁_∂z̄₀ * conj(w₀)
   = Wirtinger(∂z₁_∂z₀ * dz₀,       ∂z₁_∂z₀ * dz̄₀) +
     Wirtinger(∂z₁_∂z̄₀ * conj(dz̄₀), ∂z₁_∂z̄₀ * conj(dz₀))
   = Wirtinger(∂z₁_∂z₀ * dz₀ + ∂z₁_∂z̄₀ * conj(dz̄₀),
               ∂z₁_∂z₀ * dz̄₀ + ∂z₁_∂z̄₀ * conj(dz₀))
   != Wirtinger(dz₁, dz̄₁)
``` -->


<!--
```julia
dz₀ = rand(Complex{Float64}) # from hypothetical upstream calculation
dz̄₀ = rand(Complex{Float64}) # from hypothetical upstream calculation

dz₁ = ∂z₁/∂z₀ * dz₀ + ∂z₁/∂z̄₀ * dz̄₀
dz̄₁ = ∂z̄₁/∂z₀ * dz₀ + ∂z̄₁/∂z̄₀ * dz̄₀
```

```julia
Base.conj(x::Wirtinger) = Wirtinger(conj(x.conjugate), conj(x.primal))

# (∂/∂z, ∂/∂z̄)(a∘b) = ∂a/∂b * (∂b/∂z, ∂b/∂z̄) + ∂a/∂b̄ * (∂b̄/∂z, ∂b̄/∂z̄)
function mul_wirtinger(a::Wirtinger, b::Wirtinger)
    return add(mul(a.primal, b), mul(a.conjugate, conj(b)))
end
```

```julia


# commutative?
w₁ = w₀ * Wirtinger(∂z₁/∂z₀, ∂z₁/∂z̄₀)
   = Wirtinger(w₀.primal * ∂z₁/∂z₀ + w₀.conjugate * ∂z₁/∂z̄₀,
               conj(w₀.conjugate) * ∂z₁/∂z₀ + conj(w₀.primal) * ∂z₁/∂z̄₀)
   = Wirtinger(dz₀ * ∂z₁/∂z₀ + dz̄₀ * ∂z₁/∂z̄₀,
               conj(dz̄₀) * ∂z₁/∂z₀ + conj(dz₀) * ∂z₁/∂z̄₀)
```


```julia
function *(a::Wirtinger, b::Wirtinger)
    new_primal = a.primal * b.primal + a.conjugate * conj(b.conjugate)
    new_conjugate = a.primal * b.conjugate + a.conjugate * conj(b.primal)
    return Wirtinger(new_primal, new_conjugate)
end

# function *(a::Wirtinger, b::Wirtinger)
#     new_primal = a.primal * b.primal + a.conjugate * b.conjugate
#     new_conjugate = conj(a.conjugate) * b.primal + conj(a.primal) * b.conjugate
#     return Wirtinger(new_primal, new_conjugate)
# end
```

```julia
w₀ = Wirtinger(dz₀, dz̄₀)

w₁ = wirt(∂z₁_∂z₀, ∂z₁_∂z̄₀) * w₀
   = Wirtinger(∂z₁_∂z₀, ∂z̄₁_∂z₀) * w₀
   = Wirtinger(∂z₁_∂z₀ * dz₀ + conj(∂z̄₁_∂z₀) * dz̄₀,
               ∂z̄₁_∂z₀ * dz₀ + conj(∂z₁_∂z₀) * dz̄₀)
   = Wirtinger(∂z₁_∂z₀ * dz₀ + ∂z₁_∂z̄₀ * dz̄₀,
               ∂z̄₁_∂z₀ * dz₀ + ∂z̄₁_∂z̄₀ * dz̄₀)
   = Wirtinger(dz₁, dz̄₁)

# commutative?
w₁ = w₀ * Wirtinger(∂z₁/∂z₀, ∂z₁/∂z̄₀)
   = Wirtinger(w₀.primal * ∂z₁/∂z₀ + w₀.conjugate * ∂z₁/∂z̄₀,
               conj(w₀.conjugate) * ∂z₁/∂z₀ + conj(w₀.primal) * ∂z₁/∂z̄₀)
   = Wirtinger(dz₀ * ∂z₁/∂z₀ + dz̄₀ * ∂z₁/∂z̄₀,
               conj(dz̄₀) * ∂z₁/∂z₀ + conj(dz₀) * ∂z₁/∂z̄₀)
```

```julia
w₀ = Wirtinger(dz₀, conj(dz̄₀))

w₁ = wirt(∂z₁_∂z₀, ∂z₁_∂z̄₀) * w₀
   = Wirtinger(∂z₁_∂z₀, ∂z̄₁_∂z₀) * w₀
   = wirt(∂z₁_∂z₀ * dz₀ + conj(∂z̄₁_∂z₀ * conj(dz̄₀)),
          conj(∂z₁_∂z₀ * conj(dz̄₀)) + ∂z̄₁_∂z₀ * dz₀)
   = Wirtinger(∂z₁_∂z₀ * dz₀ + ∂z₁_∂z̄₀ * dz̄₀,
               conj(∂z̄₁_∂z̄₀ * dz̄₀ + ∂z̄₁_∂z₀ * dz₀))
   = Wirtinger(dz₁, conj(dz̄₁))

   = w₀ * wirt(∂z₁_∂z₀, ∂z₁_∂z̄₀)
   = w₀ * Wirtinger(∂z₁_∂z₀, ∂z̄₁_∂z₀)
   = wirt(dz₀ * ∂z₁_∂z₀ + conj(conj(dz̄₀) * ∂z̄₁_∂z₀),
          conj(dz₀ * ))

   # = Wirtinger(∂z₁_∂z₀ * dz₀ + ∂z₁_∂z̄₀ * dz̄₀,
   #             conj(∂z̄₁_∂z₀ * dz₀ + ∂z̄₁_∂z̄₀ * dz̄₀))
   # = Wirtinger(dz₁, conj(dz̄₁))
```

```julia
function *(a::Wirtinger, b::Wirtinger)
    new_primal = a.primal * b.primal + conj(a.conjugate * b.conjugate)
    new_conjugate = conj(a.primal * b.conjugate) + a.conjugate * b.primal
    return wirt(new_primal, new_conjugate)
end
```

```julia
# frule(::typeof(abs2), x) = abs2(x), WirtingerChain((Δz) ->


@scalar_rule(f(x, y), Wirtinger(x, x'), ∂f_∂y)

function ChainRules.frule(::typeof(abs2), x)
    Ω = abs2(x)
    return Ω, make_Wirtinger_Chain(Wirtinger(x', x))
end

f(x) = abs2(x)



frule(::typeof(f), x, x̄, y) = f(x, x̄, y), (Chain((Δx, Δx̄) -> ∂f_∂x * Δx + ∂f_∂x̄ * Δx̄), # ΔΩ
                                           Chain((Δx, Δx̄) -> ∂f̄_∂x * Δx + ∂f̄_∂x̄ * Δx̄)) # ΔΩ̄


function make_Wirtinger_Chain(∂₁::Wirtinger, ∂₂)
    WirtingerChain(Chain(Δx -> primal(∂x) * primal(Δx) + conjugate(∂x) * conjugate(Δx)), # f
                   Chain(Δx -> primal(∂x) * primal(Δx) + conjugate(∂x) * conjugate(Δx))) # f̄
end

(c::WirtingerChain)(args...) = Wirtinger(c.f(args...), c.f̄(args...))

# function WirtingerChain(Wirtinger(∂f_∂x, ∂f_∂x̄), y)
#     WirtingerChain(Chain((Δx, Δx̄) -> ∂f_∂x * Δx + ∂f_∂x̄ * Δx̄ + y * ),
#                    Chain((Δx, Δx̄) -> ∂f̄_∂x * Δx + ∂f̄_∂x̄ * Δx̄))
# end
#
# function WirtingerChain(∂f_∂x::Real, ∂f_∂x̄::Real)
#     Chain((ΔΩ, Δx, Δx̄) -> ΔΩ + ∂f_∂x * Δx + ∂f_∂x̄ * Δx̄,
#           (ΔΩ̄, Δx, Δx̄) -> ΔΩ̄ + ∂f̄_∂x * Δx + ∂f̄_∂x̄ * Δx̄)
# end
```
-->
