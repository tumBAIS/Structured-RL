using DataDeps
using DecisionFocusedLearningBenchmarks
using InferOpt
using Flux
using Flux.Optimise
using Random
using StatsBase
using Distributions
using LinearAlgebra
using WarcraftShortestPaths
using JLD2
using Plots

include("../../utils.jl")

cost(y; c_true, kwargs...) = dot(y, c_true)

function reward_comparison(train_rew, val_rew)
    means = [(train_rew[i] + val_rew[i]) / 2 for i in eachindex(train_rew)]
    last_mean = means[end]  # Mean of the last two elements
    return last_mean == minimum(means)  # Check if it's the smallest mean
end
