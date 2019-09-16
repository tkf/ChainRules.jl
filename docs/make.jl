using ChainRules
using ChainRulesCore
using Documenter

makedocs(
    modules=[ChainRules, ChainRulesCore],
    format=Documenter.HTML(prettyurls=false, assets = ["assets/chainrules.css"]),
    sitename="ChainRules",
    authors="Jarrett Revels and other contributors",
    pages=[
        "Introduction" => "index.md",
        "Getting Started" => "getting_started.md",
        "API" => "api.md",
    ],
)

deploydocs(repo="github.com/JuliaDiff/ChainRules.jl.git")
