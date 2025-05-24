using ChainRulesCore
using DynamicVehicleRouting
using IterTools: partition
using InferOpt: GeneralizedMaximizer, PerturbedAdditive, Pushforward, FenchelYoungLoss
using Statistics
using Distributions
using Graphs
using GraphNeuralNetworks

## Policy definition

# Policy structure
p(θ, stdev) = MvNormal(θ, stdev * I)

mutable struct CombinatorialACPolicy{M1,M2,P,CO,R<:AbstractRNG,S<:Union{Int,Nothing}}
    actor_model::M1
    critic_model::M2
    p::P # for the actor_model
    CO_layer::CO
    rng::R
    seed::S
end

function CombinatorialACPolicy(; actor_model, critic_model, p, CO_layer, seed=0)
    return CombinatorialACPolicy(
        actor_model, critic_model, p, CO_layer, MersenneTwister(seed), seed
    )
end

# Policy function (apply policy to environment)
function (π::CombinatorialACPolicy)(env::RLDVSPEnv; rng=nothing, perturb=true, kwargs...)
    (; actor_model, p, CO_layer) = π
    s = embedding(env)
    θ = actor_model(s)
    if !perturb
        η = θ
        a = CO_layer(θ; instance=state(env), kwargs...)
    else
        η = rand(isnothing(rng) ? π.rng : rng, p(θ, sigmaF_dvsp))
        a = CO_layer(η; instance=state(env), kwargs...)
    end
    s_c = embedding_critic(env)
    return (; state=deepcopy(state(env)), s, θ, η, a, s_c)
end

function reset_seed!(π::CombinatorialACPolicy)
    (; seed, rng) = π
    return Random.seed!(rng, seed)
end

## Critic GNN

begin
    struct critic_GNN
        layers::NamedTuple
    end

    Flux.@layer critic_GNN

    function critic_GNN(node_features, edge_features)
        layers = (
            g1=NNConv(node_features => 15, Dense(edge_features, node_features), celu),
            g2=NNConv(15 => 10, Dense(edge_features, 15), celu),
            g3=NNConv(10 => 10, Dense(edge_features, 10), celu),
            g_out=GraphConv(10 => 10, celu),
            c1=GraphConv(node_features => 15, celu),
            c2=GraphConv(15 => 10, celu),
            c3=GraphConv(10 => 10, celu),
            c_out=GraphConv(10 => 10, celu),
            pool=GlobalPool(+),
            l1=Dense(10, 15, celu),
            l2=Dense(15, 10, celu),
            l3=Dense(10, 10, celu),
            l4=Dense(10, 5, celu),
            l_out=Dense(5, 1, celu),
        )
        return critic_GNN(layers)
    end

    function (critic_GNN::critic_GNN)(g::GNNGraph, x::AbstractMatrix, e::AbstractMatrix)
        l = critic_GNN.layers
        if isassigned(e, 1)
            h1 = l.g1(g, x, e)
            h2 = l.g2(g, h1, e)
            h3 = l.g3(g, h2, e)
            h_out = l.g_out(g, h3)
        else
            h1 = l.c1(g, x)
            h2 = l.c2(g, h1)
            h3 = l.c3(g, h2)
            h_out = l.c_out(g, h3)
        end
        pool = l.pool(g, h_out)
        k1 = l.l1(pool)
        k2 = l.l2(k1)
        k3 = l.l3(k2)
        k4 = l.l4(k3)
        out = l.l_out(k4)
        return out
    end
end

## Episode generation

# General case
function apply_policy!(π::CombinatorialACPolicy, env::RLDVSPEnv; perturb=true, kwargs...)
    _, _, _, _, a = π(env; perturb, kwargs...)  # sample an action
    return apply_action!(env, a)     # apply the action
end

function generate_episode(π::CombinatorialACPolicy, env::RLDVSPEnv; kwargs...)
    reset_env!(env)
    local trajectory
    while !is_terminated(env)
        state, s, θ, η, a, s_c = π(env; kwargs...)
        next_state, reward = apply_action!(env, a)
        next_s = embedding(env)
        next_s_c = embedding_critic(env)
        if @isdefined trajectory
            push!(
                trajectory,
                (;
                    state,
                    s,
                    θ,
                    η,
                    a,
                    next_state,
                    next_s,
                    reward,
                    s_c,
                    next_s_c,
                    Rₜ=0.0,
                    adv=0.0,
                ),
            )
        else
            trajectory = [(;
                state,
                s,
                θ,
                η,
                a,
                next_state,
                next_s,
                reward,
                s_c,
                next_s_c,
                Rₜ=0.0,
                adv=0.0,
            )]
        end
    end

    return trajectory
end

# PPO-specific
function PPO_episodes(
    p::CombinatorialACPolicy,
    target_critic,
    envs,
    nb_episodes::Int,
    γ,
    V_method,
    adv_method;
    kwargs...,
)
    training_envs = sample(envs, nb_episodes)
    local episodes
    (; critic_model) = p
    for e in 1:nb_episodes
        trajectories = generate_episode(p, training_envs[e]; kwargs...)
        pop!(trajectories) # Remove the last element (the last state is not used)
        for i in length(trajectories):-1:1
            rew_new = trajectories[i].reward / 100 # Scaling to improve learning stability
            # Calculate cumulative discounted returns
            if i == length(trajectories) # Account for future value of second-last state
                ret =
                    trajectories[i].reward +
                    γ * V_value_GNN(
                        p,
                        trajectories[i].next_s,
                        trajectories[i].next_s_c,
                        target_critic,
                        V_method;
                        instance=trajectories[i].next_state,
                        kwargs...,
                    )
            else
                ret = trajectories[i].reward + γ * trajectories[i + 1].Rₜ
            end
            # Calculate advantages
            if adv_method == "TD_n" # TD(n)
                advantage =
                    ret - V_value_GNN(
                        p,
                        trajectories[i].s,
                        trajectories[i].s_c,
                        critic_model,
                        V_method;
                        instance=trajectories[i].state,
                        kwargs...,
                    )
            else # TD(1)
                advantage =
                    trajectories[i].reward +
                    γ * V_value_GNN(
                        p,
                        trajectories[i].next_s,
                        trajectories[i].next_s_c,
                        target_critic,
                        V_method;
                        instance=trajectories[i].next_state,
                        kwargs...,
                    ) - V_value_GNN(
                        p,
                        trajectories[i].s,
                        trajectories[i].s_c,
                        critic_model,
                        V_method;
                        instance=trajectories[i].state,
                        kwargs...,
                    )
            end
            # Update trajectories with returns and advantages
            traj = trajectories[i]
            traj_updated = (; traj..., reward=rew_new, Rₜ=ret, adv=advantage)
            trajectories[i] = traj_updated
        end
        if e == 1
            episodes = deepcopy(trajectories)
        else
            append!(episodes, trajectories)
        end
    end
    return episodes
end

# SRL-specific
function SRL_episodes(p::CombinatorialACPolicy, envs, nb_episodes::Int; kwargs...)
    training_envs = sample(envs, nb_episodes)
    local episodes
    for e in 1:nb_episodes
        trajectories = generate_episode(p, training_envs[e]; perturb=true, kwargs...)
        pop!(trajectories)
        if e == 1
            episodes = deepcopy(trajectories)
        else
            append!(episodes, trajectories)
        end
    end
    return episodes
end

## Replay buffer

function rb_add(replay_buffer, rb_capacity, rb_position, rb_size, episodes)
    for episode in episodes
        if rb_size < rb_capacity
            push!(replay_buffer, episode)
        else
            replay_buffer[rb_position] = episode
        end
        rb_position = (rb_position % rb_capacity) + 1
        rb_size = min(rb_size + 1, rb_capacity)
    end
    return replay_buffer, rb_position, rb_size
end

function rb_sample(replay_buffer, batch_size)
    idxs = rand(eachindex(replay_buffer), batch_size)
    return [replay_buffer[i] for i in idxs]
end

## Actor losses

# PPO-specific
function J_PPO(π::CombinatorialACPolicy, batch, clip, sigmaF_average)
    (; p) = π
    embeddings = getfield.(batch, :s)
    thetas = getfield.(batch, :θ)
    advantages = getfield.(batch, :adv)
    etas = getfield.(batch, :η)

    # Policy ratio
    old_probs = [
        logdensityof(p(thetas[j], sigmaF_average), etas[j]) for j in eachindex(batch)
    ]
    new_probs = [
        logdensityof(p(π.actor_model(embeddings[j]), sigmaF_average), etas[j]) for
        j in eachindex(batch)
    ]

    # Actor loss
    ratio_unclipped = [exp(new_probs[j] - old_probs[j]) for j in eachindex(batch)]
    ratio_clipped = clamp.(ratio_unclipped, 1 - clip, 1 + clip)
    return mean(min.(ratio_unclipped .* advantages, ratio_clipped .* advantages))
end

# SRL-specific
function SRL_actions(
    π::CombinatorialACPolicy, batch; sigmaB=0.05, no_samples=20, temp=1.0, kwargs...
)
    # Load data
    (; actor_model, critic_model, p) = π
    embeddings = getfield.(batch, :s)
    embeds_c = getfield.(batch, :s_c)
    states = getfield.(batch, :state)
    best_solutions = []
    for j in eachindex(batch)
        # Perturb and sample candidate actions
        θ = actor_model(embeddings[j])
        η = rand(π.rng, p(θ, sigmaB), no_samples - 1)
        route = prize_collecting_vsp(θ; instance=states[j], kwargs...)
        solutions = [VSPSolution(route; max_index=nb_locations(states[j])).edge_matrix]
        values = [Q_value_GNN(route, embeds_c[j], critic_model; instance=states[j])]
        for i in 1:(no_samples - 1)
            route = prize_collecting_vsp(η[:, i]; instance=states[j], kwargs...)
            push!(
                solutions, VSPSolution(route; max_index=nb_locations(states[j])).edge_matrix
            )
            push!(values, Q_value_GNN(route, embeds_c[j], critic_model; instance=states[j]))
        end
        # Calculate target action (called solutions for better distinction)
        values = values ./ temp
        lse = logsumexp(values)
        probs = exp.(values .- lse)
        best_action = sum(probs .* solutions)
        any(isnan.(best_action)) ? best_action = solutions[argmax(values)] : nothing
        push!(best_solutions, best_action)
    end
    return best_solutions
end

function optimization_fyl(θ; instance, kwargs...) # CO-layer
    routes = prize_collecting_vsp(θ; instance=instance, kwargs...)
    return VSPSolution(routes; max_index=nb_locations(instance.instance)).edge_matrix
end

function g_fyl(y; instance, kwargs...) # Helper for Fenchel-Young loss
    return vec(sum(y[:, instance.is_postponable]; dims=1))
end

function h_fyl(y, duration) # Helper for Fenchel-Young loss
    value = 0.0
    N = size(duration, 1)
    for i in 1:N
        for j in 1:N
            value -= y[i, j] * duration[i, j]
        end
    end
    return value
end

function h_fyl(y; instance, kwargs...) # Helper for Fenchel-Young loss
    return h_fyl(y, instance.instance.duration)
end

function J_SRL(π::CombinatorialACPolicy, batch, best_solutions; kwargs...)
    # Definition of Fenchel-Young loss
    gm = GeneralizedMaximizer(optimization_fyl, g_fyl, h_fyl)
    perturbed_layer = PerturbedAdditive(gm; ε=1e-2, nb_samples=20, is_parallel=true)
    fyl = FenchelYoungLoss(perturbed_layer)

    embeddings = getfield.(batch, :s)
    states = getfield.(batch, :state)

    # Update actor using Fenchel-Young loss
    l = 0.0
    for j in eachindex(batch)
        l += fyl(
            π.actor_model(embeddings[j]), best_solutions[j]; instance=states[j], kwargs...
        )
    end
    return l
end

## Critic losses

# Loss preparation
function grads_prep_GNN(batch)
    states = getfield.(batch, :state)
    routes = getfield.(batch, :a)
    s_c = getfield.(batch, :s_c)

    graphs = []
    edge_features = []

    for j in eachindex(batch)
        # Create adjacency matrix of action
        adj_matrix =
            VSPSolution(routes[j]; max_index=nb_locations(states[j].instance)).edge_matrix
        dist_matrix = states[j].instance.duration

        # Create graph of action
        n = size(adj_matrix, 1)
        g = DiGraph(n)
        for i in 1:n
            for j in 1:n
                if adj_matrix[i, j] == 1
                    add_edge!(g, i, j)
                end
            end
        end
        push!(graphs, GNNGraph(g))

        # Add adjacency matrix as edge features to graph
        push!(
            edge_features,
            Float32.(Matrix([dist_matrix[src(e), dst(e)] for e in edges(g)]')),
        )
    end

    return graphs, s_c, edge_features
end

# Huber loss
function huber_GNN(π, graphs, s_c, edge_features, critic_target, δ; kwargs...)
    new_critic = [
        -sum(π.critic_model(graphs[j], Float32.(s_c[j]), edge_features[j])) for
        j in eachindex(critic_target)
    ]

    # Calculate loss
    error = new_critic .- critic_target
    quadratic = 0.5 .* error .^ 2
    linear = δ .* (abs.(error) .- 0.5 .* δ)
    return mean(ifelse.(abs.(error) .<= δ, quadratic, linear))
end

## Q-value definition

# Q-values (GNN)
function Q_value_GNN(routes, s_c, critic_model; instance)
    adj_matrix = VSPSolution(routes; max_index=nb_locations(instance.instance)).edge_matrix
    dist_matrix = instance.instance.duration

    # Construct graph
    n = size(adj_matrix, 1)
    g = DiGraph(n)
    for i in 1:n
        for j in 1:n
            if adj_matrix[i, j] == 1
                add_edge!(g, i, j)
            end
        end
    end
    graph = GNNGraph(g)

    # Add edge features (distance matrix) to graph and pass graph to critic
    edge_features = Float32.(Matrix([dist_matrix[src(e), dst(e)] for e in edges(g)]'))
    return -sum(critic_model(graph, Float32.(s_c), edge_features))
end

# V-values (GNN)
function V_value_GNN(
    π::CombinatorialACPolicy, s, s_c, critic_model, method; instance, kwargs...
)
    (; actor_model, CO_layer, p) = π
    θ = actor_model(s)
    if method == "on_policy" # With perturbation
        η = rand(π.rng, p(θ, sigmaF_dvsp))
        action = CO_layer(η; instance, kwargs...)
    else # Without perturbation
        action = CO_layer(θ; instance, kwargs...)
    end
    return Q_value_GNN(action, s_c, critic_model; instance)
end

function evaluate_policy(
    π::CombinatorialACPolicy,
    envs::AbstractVector;
    nb_episodes::Int=1000,
    return_scores=false,
    kwargs...,
)
    score_per_trajectory = Float64[]
    for env in envs
        for _ in 1:nb_episodes
            reset_env!(env)
            trajectory_score = 0
            while !is_terminated(env)
                _, reward = apply_policy!(π, env; kwargs...)
                trajectory_score += reward
            end
            push!(score_per_trajectory, trajectory_score)
        end
    end
    mean_reward = sum(score_per_trajectory) / (nb_episodes * length(envs))
    # @info "std",
    sqrt(
        sum(
            (score_per_trajectory .- mean_reward) .* (score_per_trajectory .- mean_reward)
        ) / (nb_episodes * length(envs)),
    )
    if !return_scores
        return mean_reward
    else
        return mean_reward, score_per_trajectory
    end
end
