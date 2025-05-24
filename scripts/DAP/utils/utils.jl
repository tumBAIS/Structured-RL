using Flux
using Flux.Optimise
using InferOpt
using Random
using JLD2
using Plots
using Distributions
using LinearAlgebra
using Combinatorics

include("../../utils.jl")

## Set up the environment

# Initialize the hidden customer model
customer_model = Chain(Dense([0.3 0.5 0.6 -0.4 -0.8 0.0]), vec)

Random.seed!(0)

# Environment structure
mutable struct DAP
    N::Int                      # Number of items
    d::Int                      # Dimension of feature vectors (in addition to hype, satisfaction, and price)
    K::Int                      # Assortment size constraint
    J::Int                      # Initial inventory
    seed::Int                   # Random seed
    max_steps::Int              # Number of steps per episode
    step::Int                   # Current step
    purchase_hist::Vector{Int}  # Purchase history
    dynamic::Bool               # Flag: endogeneity of the customer model

    function DAP(N, d, K, J; max_steps=80, seed=0, dynamic=true)
        purchase_hist = Int[]
        return new(N, d, K, J, seed, max_steps, 0, purchase_hist, dynamic)
    end
end

# Feature 1: Random static feature
# Feature 2: Random static feature
# Feature 3: Hype
# Feature 4: Satisfaction
# Feature 5: Price

## Basic operations of environment

# Reset the environment
function reset!(env::DAP; seed=0)
    env.seed = seed
    features = rand(MersenneTwister(env.seed), Uniform(1.0, 10.0), (env.d + 3, env.N))
    features = vcat(features, ones(1, env.N))
    d_features = zeros(2, env.N)
    inventory = ones(env.N)
    env.step = 1
    return features, d_features, inventory
end

# Environment seeds:
# Training: 1:100
# Validation: 1001:1100
# Testing: 2001:2100

# Update the hype vector
function hype_update(env::DAP)
    hype_vector = ones(env.N)
    env.purchase_hist[end] != 0 ? hype_vector[env.purchase_hist[end]] += 0.02 : nothing
    if length(env.purchase_hist) >= 2
        if env.purchase_hist[end - 1] != 0
            hype_vector[env.purchase_hist[end - 1]] -= 0.005
        else
            nothing
        end
        if length(env.purchase_hist) >= 3
            if env.purchase_hist[end - 2] != 0
                hype_vector[env.purchase_hist[end - 2]] -= 0.005
            else
                nothing
            end
            if length(env.purchase_hist) >= 4
                if env.purchase_hist[end - 3] != 0
                    hype_vector[env.purchase_hist[end - 3]] -= 0.005
                else
                    nothing
                end
                if length(env.purchase_hist) >= 5
                    if env.purchase_hist[end - 4] != 0
                        hype_vector[env.purchase_hist[end - 4]] -= 0.005
                    else
                        nothing
                    end
                end
            end
        end
    end
    return hype_vector
end

# Step function
function step!(env::DAP, features, inventory, item)
    old_features = copy(features)
    push!(env.purchase_hist, item)
    if env.dynamic
        hype_vector = hype_update(env)
        features[3, :] .*= hype_vector
        item != 0 ? features[4, item] *= 1.01 : nothing
        features[6, :] .+= 9 / env.max_steps
    end
    d_features = features[3:4, :] - old_features[3:4, :]
    item != 0 ? inventory[item] -= 1 / env.J : nothing
    inventory = round.(inventory, digits=4)
    env.step += 1
    return features, d_features, inventory
end

# Choice probabilities
function choice_probabilities(env::DAP, S, θ)
    exp_values = [exp(θ[i]) * S[i] for i in 1:(env.N)]
    denominator = 1 + sum(exp_values)
    probs = [exp_values[i] / denominator for i in 1:(env.N)]
    push!(probs, 1 / denominator) # Probability of no purchase
    return probs
end

# Purchase decision
function purchase(env::DAP, S, r, θ_0)
    probs = choice_probabilities(env, S, θ_0)
    sampled_item = rand(MersenneTwister(env.seed + env.step), Multinomial(1, probs))
    # @show probs sampled_item
    item = findfirst(==(1), sampled_item)
    item == env.N + 1 ? item = 0 : item
    item != 0 ? revenue = r[item] : revenue = 0.0
    return item, revenue
end

# Perfect solution
function expert_solution(env::DAP, r, inventory, θ_0)
    best_S = []
    best_revenue = 0.0
    for S in combinations(1:(env.N), env.K)
        skip = false
        for i in S
            inventory[i] <= 0.0 ? skip = true : nothing
        end
        skip ? continue : nothing
        S_vec = zeros(env.N)
        S_vec[S] .= 1
        probs = choice_probabilities(env, S_vec, θ_0)
        pop!(probs)
        expected_revenue = sum(probs .* r)
        if expected_revenue > best_revenue
            best_S, best_revenue = S_vec, expected_revenue
        end
    end
    return best_S
end

# DAP CO-layer
function DAP_optimization(θ; env::DAP)
    solution = partialsortperm(θ, 1:(env.K); rev=true) # It never makes sense not to show k items
    S = zeros(env.N)
    S[solution] .= 1
    return S
end

## Solution functions

# Anticipative (fixed)
function expert_policy(
    env::DAP, customer_model, episodes; first_seed=1, create_dataset=false, use_oracle=false
)
    dataset = []
    rev_global = Float64[]
    for i in 1:episodes
        rev_episode = 0.0
        start_features, d_features, inventory = reset!(env; seed=first_seed - 1 + i)
        features = copy(start_features)
        done = false
        training_instances = []
        while !done
            r = features[5, :]
            θ_0 = customer_model(features)
            if use_oracle
                θ_adj = θ_0 .* r
                S = DAP_optimization(θ_adj; env=env)
            else
                S = expert_solution(env, r, inventory, θ_0)
            end
            item, revenue = purchase(env, S, r, θ_0)
            rev_episode += revenue
            delta_features = features[3:4, :] .- start_features[3:4, :]
            feature_vector = vcat(features, d_features, delta_features)
            push!(training_instances, (features=feature_vector, S_t=S))
            features, d_features, inventory = step!(env, features, inventory, item)
            count(!iszero, inventory) < env.K ? break : nothing
            env.step > env.max_steps ? done = true : done = false
        end
        push!(rev_global, rev_episode)
        push!(dataset, training_instances)
    end
    if create_dataset
        return mean(rev_global), rev_global, dataset
    else
        return mean(rev_global), rev_global
    end
end

# Validation/testing function
function val_test(env::DAP, model, customer_model, episodes; first_seed=1)
    rev_global = Float64[]
    for i in 1:episodes
        rev_episode = 0.0
        start_features, d_features, inventory = reset!(env; seed=first_seed - 1 + i)
        features = copy(start_features)
        done = false
        while !done
            delta_features = features[3:4, :] .- start_features[3:4, :]
            r = features[5, :]
            feature_vector = vcat(features, d_features, delta_features)
            θ = model(feature_vector)
            S = DAP_optimization(θ; env=env)
            θ_0 = customer_model(features)
            item, revenue = purchase(env, S, r, θ_0)
            rev_episode += revenue
            features, d_features, inventory = step!(env, features, inventory, item)
            count(!iszero, inventory) < env.K ? break : nothing
            env.step > env.max_steps ? done = true : done = false
        end
        push!(rev_global, rev_episode)
    end
    return mean(rev_global), rev_global
end

# Greedy heuristic
function model_greedy(features)
    model = Chain(Dense([0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0]), vec)
    return model(features)
end

# Random heuristic
function model_random(features)
    rand_seed = Int(round(sum(features)))
    return rand(MersenneTwister(rand_seed), Uniform(0.0, 1.0), size(features)[2])
end

## Model definitions

# Fixed critic
function critic_reward(features, S, critic_rew)
    features = features[1:6, :]
    features = features[:, S .!= 0]
    return mean(critic_rew(features))
end

# Dynamic critic
function critic_future(features, S, critic_fut)
    features = vcat(features, transpose(S))
    return mean(critic_fut(features))
end

# Episode generation
function generate_episode(
    env::DAP, model, critic_rew, critic_fut, customer_model, sigma, random_seed
)
    buffer = []
    start_features, d_features, inventory = reset!(env; seed=random_seed)
    features = copy(start_features)
    done = false
    while !done
        delta_features = features[3:4, :] .- start_features[3:4, :]
        r = features[5, :]
        feature_vector = vcat(features, d_features, delta_features)
        θ = model(feature_vector)
        η = rand(MersenneTwister(random_seed * env.step), p(θ, sigma), 1)[:, 1]
        S = DAP_optimization(η; env=env)
        θ_0 = customer_model(features)
        item, revenue = purchase(env, S, r, θ_0)
        features, d_features, inventory = step!(env, features, inventory, item)
        feat_next = vcat(features, d_features, features[3:4, :] .- start_features[3:4, :])
        push!(
            buffer,
            (
                t=env.step - 1,
                feat_t=feature_vector,
                theta=θ,
                eta=η,
                S_t=S,
                a_T=item,
                rev_t=revenue,
                ret_t=0.0,
                feat_next=feat_next,
            ),
        )
        count(!iszero, inventory) < env.K ? break : nothing
        env.step > env.max_steps ? done = true : done = false
    end
    for i in (length(buffer) - 1):-1:1
        if i == length(buffer) - 1
            ret = buffer[i].rev_t
        else
            ret = buffer[i].rev_t + 0.99 * buffer[i + 1].ret_t
        end
        traj = buffer[i]
        traj_updated = (; traj..., ret_t=ret)
        buffer[i] = traj_updated
    end
    return buffer
end

# Replay buffer
function rb_add(replay_buffer, experience, rb_capacity, rb_position, rb_size)
    for transition in experience
        if rb_size < rb_capacity
            push!(replay_buffer, transition)
        else
            replay_buffer[rb_position] = transition
        end
        rb_position = (rb_position % rb_capacity) + 1
        rb_size = min(rb_size + 1, rb_capacity)
    end
    return replay_buffer, rb_position, rb_size
end

# Sample from the replay buffer
function rb_sample(replay_buffer, batch_size)
    idxs = rand(eachindex(replay_buffer), batch_size)
    return [replay_buffer[i] for i in idxs]
end

# Reward comparison function
function reward_comparison(train_rew, val_rew; minim=1)
    means = [(train_rew[i] + val_rew[i]) / 2 for i in eachindex(train_rew)]
    length(means) >= minim ? means = means[minim:end] : nothing
    last_mean = means[end]  # Mean of the last two elements
    return last_mean == maximum(means)  # Check if it's the largest mean
end
