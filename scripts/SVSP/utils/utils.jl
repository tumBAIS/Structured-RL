using DecisionFocusedLearningBenchmarks
using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling:
    evaluate_solution, compact_mip, local_search, deterministic_mip
using Gurobi
using JLD2
using StatsBase: StatsBase
using Statistics: mean
using MLUtils: splitobs, numobs
using ProgressMeter: @showprogress

include("../../utils.jl")

dataset_path = joinpath(logdir, "datasets.jld2")
test_dataset_path = joinpath(logdir, "test_datasets.jld2")
results_path = joinpath(logdir, "results.jld2")

# Model initialization
b = StochasticVehicleSchedulingBenchmark()
model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)

# Reward comparison
function reward_comparison(train_rew, val_rew)
    means = [(train_rew[i] + val_rew[i]) / 2 for i in eachindex(train_rew)]
    last_mean = means[end]  # Mean of the last two elements
    return last_mean == minimum(means)  # Check if it's the smallest mean
end
