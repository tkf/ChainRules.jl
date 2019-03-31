using Documenter, ChainRules

makedocs(modules=[ChainRules],
         doctest = false,
         format = :html,
         sitename = "ChainRules",
         pages = ["Introduction" => "index.md",
                  "Complex-Valued Derivatives" => "wirtinger.md"])

deploydocs(repo = "github.com/JuliaDiff/ChainRules.jl.git",
           osname = "linux",
           julia = "1.0",
           target = "build",
           deps = nothing,
           make = nothing)
