using DecisionFocusedLearningBenchmarks
using DynamicVehicleRouting
using DensityInterface: logdensityof
using DataDeps
using MLUtils
using Random
using Flux
using Distributions: MvNormal
using DynamicVehicleRouting
using Flux
using Flux.Optimise
using GraphNeuralNetworks
using Gurobi
using DecisionFocusedLearningBenchmarks: grb_model
using JLD2
using LinearAlgebra: I, dot
using Random
using StatsBase
using Plots

include("../../utils.jl")

## Environment definition

# Environment structure
mutable struct RLDVSPEnv{D<:DVSPEnv,S}
    env::D
    is_deterministic::Bool
    scaling_features::S
end

function RLDVSPEnv(
    instance_path::String;
    max_requests_per_epoch=10,
    seed=67,
    is_deterministic=true,
    scaling_features=nothing,
)
    static_instance = read_vsp_instance(instance_path)
    env = DVSPEnv(static_instance; seed, max_requests_per_epoch)
    next_epoch!(env)
    return RLDVSPEnv(env, is_deterministic, scaling_features)
end

state(env::RLDVSPEnv) = DynamicVehicleRouting.get_state(env.env)

is_terminated(env::RLDVSPEnv) = DynamicVehicleRouting.is_terminated(env.env)

# Actor and critic features
function embedding(env::RLDVSPEnv)
    x = Float32.(compute_features(env.env))
    if isnothing(env.scaling_features)
        return x
    else
        return StatsBase.transform(env.scaling_features, x)
    end
end

function embedding_critic(env::RLDVSPEnv)
    x = Float32.(compute_critic_features(env.env))
    if isnothing(env.scaling_features)
        return x
    else
        return StatsBase.transform(env.scaling_features, x)
    end
end

# Step and reset environment
function apply_action!(env::RLDVSPEnv, routes::Vector{Vector{Int}})
    env_routes = env_routes_from_state_routes(env.env, routes)

    # reward = apply_decision!(env.env, env_routes; check_feasibility=false)
    reward = apply_decision!(env.env, env_routes)
    next_epoch!(env.env)
    return state(env), -reward
end

function reset_env!(env::RLDVSPEnv)
    DynamicVehicleRouting.reset!(env.env; reset_seed=env.is_deterministic)
    next_epoch!(env.env)
    return nothing
end

# Policy evaluation
function evaluate_policy(
    π::DynamicVehicleRouting.AbstractDynamicPolicy,
    envs::AbstractVector{<:RLDVSPEnv};
    nb_episodes::Int=1_000,
    kwargs...,
)
    score_per_trajectory = Float64[]
    for env in envs
        for _ in 1:nb_episodes
            trajectory_score, _ = DynamicVehicleRouting.run_policy!(π, env.env; kwargs...)
            push!(score_per_trajectory, trajectory_score)
        end
    end
    mean_reward = sum(score_per_trajectory) / (nb_episodes * length(envs))
    return mean_reward, score_per_trajectory
end

# Environment setup
function _euro_neurips_unpack(local_filepath)
    directory = dirname(local_filepath)
    unpack(local_filepath)
    # Move instances and delete the rest
    instance_dir = joinpath(directory, "euro-neurips-vrp-2022-quickstart-main", "instances")
    instance_files = filter(f -> endswith(f, ".txt"), readdir(instance_dir; join=true))

    # Split instances into train, validation, and test
    train_instances, validation_instances, test_instances = splitobs(
        MersenneTwister(0), instance_files; at=(0.34, 0.34), shuffle=true
    )

    # Create directories for train, validation, and test
    train_dir = joinpath(directory, "train")
    validation_dir = joinpath(directory, "validation")
    test_dir = joinpath(directory, "test")
    mkpath(train_dir)
    mkpath(validation_dir)
    mkpath(test_dir)

    # Move files to respective directories
    for filepath in train_instances
        mv(filepath, joinpath(train_dir, basename(filepath)))
    end
    for filepath in validation_instances
        mv(filepath, joinpath(validation_dir, basename(filepath)))
    end
    for filepath in test_instances
        mv(filepath, joinpath(test_dir, basename(filepath)))
    end

    # Remove the original unpacked directory
    rm(joinpath(directory, "euro-neurips-vrp-2022-quickstart-main"); recursive=true)
    return nothing
end

## Baseline policies

# Greedy policy
struct Greedy end

function (π::Greedy)(env::RLDVSPEnv; rng=nothing, kwargs...)
    # nb_requests = sum(.!env.state.is_must_dispatch[2:end])
    nb_requests = sum(state(env).is_postponable)
    θ = ones(nb_requests) * 1e9
    routes = prize_collecting_vsp(θ; instance=state(env), kwargs...)
    return nothing, nothing, routes
end

function apply_policy!(π::Greedy, env::RLDVSPEnv; kwargs...)
    _, _, routes = π(env; kwargs...)   # sample an action
    return apply_action!(env, routes)  # apply the action
end

# Expert policy
function expert_evaluation(envs; model_builder)
    total = []
    for e in envs
        final_env = e.env
        routes_expert = anticipative_solver(final_env; model_builder)
        duration = final_env.config.static_instance.duration[
            final_env.customer_index, final_env.customer_index
        ]
        expert_costs = [cost(routes, duration) for routes in routes_expert]
        push!(total, -sum(expert_costs))
    end
    return sum(total) / length(envs), total
end

function cost(routes::Vector{Vector{Int}}, duration::AbstractMatrix)
    total = zero(eltype(duration))
    for route in routes
        current_location = 1
        for r in route
            total += duration[current_location, r]
            current_location = r
        end
        total += duration[current_location, 1]
    end
    return total
end

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

register(
    DataDep(
        "euro-neurips-2022",
        "EURO-NeurIPs challenge 2022 dataset",
        "https://github.com/ortec/euro-neurips-vrp-2022-quickstart/archive/refs/heads/main.zip";
        post_fetch_method=_euro_neurips_unpack,
    ),
)

dataset_path = datadep"euro-neurips-2022"

train_instances = joinpath(dataset_path, "train")
val_instances = joinpath(dataset_path, "validation")
test_instances = joinpath(dataset_path, "test")

nb_train_instances = 10
nb_val_instances = 10
nb_test_instances = 10
max_requests_per_epoch = 10

model_builder = grb_model # highs_model if you do not have gurobi

# Train instances
scaling_envs = map(2:nb_train_instances) do i
    instance_path = [joinpath(train_instances, e) for e in readdir(train_instances)][i]
    return RLDVSPEnv(instance_path; max_requests_per_epoch, seed=0, is_deterministic=true)
end;

X = reduce(hcat, [embedding(env) for env in scaling_envs])
dt = fit(ZScoreTransform, X; dims=2)
StatsBase.transform(dt, X)

train_envs = map(1:nb_train_instances) do i # Start at index 2, because MacOS adds a .DS_Store to the folder
    instance_path = [joinpath(train_instances, e) for e in readdir(train_instances)][i]
    return RLDVSPEnv(
        instance_path;
        max_requests_per_epoch,
        seed=0,
        is_deterministic=true,
        # scaling_features=dt,
    )
end;

# Validation instances
scaling_envs = map(1:nb_val_instances) do i
    instance_path = [joinpath(val_instances, e) for e in readdir(val_instances)][i]
    return RLDVSPEnv(instance_path; max_requests_per_epoch, seed=0, is_deterministic=true)
end;

X = reduce(hcat, [embedding(env) for env in scaling_envs])
dt = fit(ZScoreTransform, X; dims=2)
StatsBase.transform(dt, X)

val_envs = map(2:nb_val_instances) do i
    instance_path = [joinpath(val_instances, e) for e in readdir(val_instances)][i]
    return RLDVSPEnv(
        instance_path;
        max_requests_per_epoch,
        seed=0,
        is_deterministic=true,
        # scaling_features=dt,
    )
end;

# Test instances
scaling_envs = map(1:nb_test_instances) do i
    instance_path = readdir(test_instances; join=true)[i]
    return RLDVSPEnv(instance_path; max_requests_per_epoch, seed=0, is_deterministic=true)
end;

X = reduce(hcat, [embedding(env) for env in scaling_envs])
dt = fit(ZScoreTransform, X; dims=2)
StatsBase.transform(dt, X)

test_envs = map(1:nb_test_instances) do i
    instance_path = readdir(test_instances; join=true)[i]
    return RLDVSPEnv(
        instance_path;
        max_requests_per_epoch,
        seed=0,
        is_deterministic=true,
        # scaling_features=dt,
    )
end;
