using LinearAlgebra
using Random
using Plots
using GridGraphs
using Flux
using Flux.Optimise
using InferOpt
using Statistics
using Distributions
using Flux
using InferOpt
using JLD2

include("../../utils.jl")

## GridWorld Environment

# Environment structure
mutable struct GridWorld
    width::Int
    height::Int
    num_features::Int
    features::Array{Float64,3}  # size: height x width x num_features
    cell_cost_weights::Array{Float64,1}  # size: num_features
    cost_param_weights::Array{Float64,1}  # size: num_features
    cell_costs::Array{Float64,2}  # size: height x width
    cell_params::Array{Float64,2}  # size: height x width
    cost_param::Float64
    robot_position::Tuple{Int,Int}
    target_position::Tuple{Int,Int}
    seed::Int
    max_steps::Int
    step::Int
end
# Initialize a new GridWorld environment
function GridWorld(;
    width=20,
    height=20,
    num_features=6,
    cell_cost_weights=[-1.0, -1.2, -0.8],
    cost_param_weights=[-0.03, 0.02, 0.01],
    seed=0,
)
    features = rand(num_features, height, width)
    cell_costs = zeros(height, width)
    cell_params = zeros(height, width)
    cost_param = 1.0
    robot_position = (rand(1:height), rand(1:width))
    target_position = (rand(1:height), rand(1:width))
    max_steps = 100
    step = 0

    return GridWorld(
        width,
        height,
        num_features,
        features,
        cell_cost_weights,
        cost_param_weights,
        cell_costs,
        cell_params,
        cost_param,
        robot_position,
        target_position,
        seed,
        max_steps,
        step,
    )
end

# Reset the environment at the beginning of an episode
function reset!(env::GridWorld; seed=0)
    env.seed = seed
    env.features = rand(MersenneTwister(seed), env.num_features, env.height, env.width)
    env.features = cat(
        env.features, fill(1 / env.max_steps, 1, env.height, env.width); dims=1
    )
    env.cell_costs = sum([env.features[i, :, :] * env.cell_cost_weights[i] for i in 1:3])
    env.cell_params = sum([
        env.features[i + 3, :, :] * env.cost_param_weights[i] for i in 1:3
    ])
    env.cost_param = 1.0
    env.robot_position = (1, 1)
    env.step = 1
    return env.target_position = (
        rand(MersenneTwister(env.step), 1:(env.height)),
        rand(MersenneTwister(env.max_steps + env.step), 1:(env.width)),
    )
end

# Environment seeds:
# Training: 1:100
# Validation: 1001:1100
# Testing: 2001:2100

# Get a random target cell
function random_target(env::GridWorld)
    target = (
        rand(MersenneTwister(env.step), 1:(env.height)),
        rand(MersenneTwister(env.max_steps + env.step), 1:(env.width)),
    )
    target == env.robot_position ? target = (env.height, env.width) : nothing
    target == env.robot_position ? target = (1, 1) : nothing
    return target
end

# Compute cost of a path
function compute_path_cost(env::GridWorld, path::Matrix{Int})
    # cost = 0.0
    # for (i, j) in path
    #     cost += env.cell_costs[i, j]
    # end
    cost = dot(path, env.cell_costs)
    return cost * env.cost_param
end

# Update the robot’s cost parameter based on the path
function update_cost_param!(env::GridWorld, path::Matrix{Int})
    delta = dot(path, env.cell_params)
    env.cost_param *= 1 + delta
    env.cost_param = max(env.cost_param, 0.01)
    return env.cost_param = min(env.cost_param, 100.0)
end

# Step function: input a path, update position and cost parameter, return cost
function step!(env::GridWorld, path::Matrix{Int})
    total_cost = compute_path_cost(env, path)
    update_cost_param!(env, path)
    env.robot_position = env.target_position
    env.target_position = random_target(env)
    env.step += 1
    env.features[7, :, :] = fill(env.step / env.max_steps, 1, env.height, env.width)
    return total_cost
end

## Path functions

# Greedy (straight) path function
function path_straight(
    start::Tuple{Int,Int}, goal::Tuple{Int,Int}; matrix=true, dims=[20, 20]
)
    x0, y0 = start[2], start[1]  # column, row
    x1, y1 = goal[2], goal[1]

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = x0 < x1 ? 1 : -1
    sy = y0 < y1 ? 1 : -1
    err = dx + dy

    path = Tuple{Int,Int}[]

    while true
        push!(path, (y0, x0))  # store as (row, col)
        if x0 == x1 && y0 == y1
            break
        end
        e2 = 2 * err
        if e2 >= dy
            err += dy
            x0 += sx
        end
        if e2 <= dx
            err += dx
            y0 += sy
        end
    end
    popfirst!(path)  # remove start node

    if matrix
        matrix = zeros(Int, dims[1], dims[2])
        for (i, j) in path
            matrix[i, j] = 1
        end
        return matrix
    else
        return path
    end
end

# Shortest path function using Dijkstra's algorithm
function path_dijkstra(θ::AbstractMatrix{R}; start, goal) where {R<:Real}
    g = GridGraph(.-θ; directions=QUEEN_DIRECTIONS)
    path = grid_dijkstra(
        g, coord_to_index(g, start[1], start[2]), coord_to_index(g, goal[1], goal[2])
    )
    popfirst!(path)  # remove start node
    return path_to_matrix(g, path)
end

# Alternative version of Dijkstra's algorithm for plotting
function dijkstra_plots(θ::AbstractMatrix{R}; start, goal) where {R<:Real}
    g = GridGraph(.-θ; directions=QUEEN_DIRECTIONS)
    path = grid_dijkstra(
        g, coord_to_index(g, start[1], start[2]), coord_to_index(g, goal[1], goal[2])
    )
    popfirst!(path)  # remove start node
    return [index_to_coord(g, i) for i in path]
end

# Expert path function using Dijkstra's algorithm
function path_expert(env::GridWorld; matrix=true)
    if matrix # For normal use
        return path_dijkstra(
            env.cell_costs; start=env.robot_position, goal=env.target_position
        )
    else # For plotting paths
        return dijkstra_plots(
            env.cell_costs; start=env.robot_position, goal=env.target_position
        )
    end
end

# Plot the path on the grid
function plot_grid_with_path(
    grid_scores::Matrix{Float64},
    path::Vector{Tuple{Int,Int}},
    start::Tuple{Int,Int},
    goal::Tuple{Int,Int},
)
    heatmap_data = grid_scores

    # Transpose the grid to match (i,j) ↔ (row, col)
    heatmap(
        heatmap_data;
        color=cgrad(:greys; rev=false), # viridis, greys, batlow, magma, cividis
        yflip=true,
        aspect_ratio=1,
        legend=false,
        axis=nothing,  # hides axes
        ticks=false,   # hides tick labels
        grid=false,    # hides grid lines
        border=:none,
    )

    # Plot path
    path_plot = vcat(start, path)
    xs = [pos[2] for pos in path_plot]
    ys = [pos[1] for pos in path_plot]
    plot!(xs, ys; linecolor=:red, linewidth=5)

    # Mark start and goal
    scatter!(
        [start[2]],
        [start[1]];
        markersize=10,
        markercolor=:lime,
        markerstrokecolor=:black,
        markerstrokewidth=3,
    )

    scatter!(
        [goal[2]],
        [goal[1]];
        markersize=10,
        markercolor=:orange,
        markerstrokecolor=:black,
        markerstrokewidth=3,
    )

    return (plot!())
end

# Plot path of model
function plot_paths(env::GridWorld, model; seed=1, step=1)
    reset!(env; seed=seed)
    if step != 1
        env.robot_position = (
            rand(MersenneTwister(step - 1), 1:(env.height)),
            rand(MersenneTwister(env.max_steps + step - 1), 1:(env.width)),
        )
    else
        nothing
    end
    if step != 1
        env.target_position = (
            rand(MersenneTwister(step), 1:(env.height)),
            rand(MersenneTwister(env.max_steps + step - 1), 1:(env.width)),
        )
    else
        nothing
    end
    if model == "opt"
        path = path_expert(env; matrix=false)
    elseif model == "greedy"
        path = path_straight(
            env.robot_position,
            env.target_position;
            matrix=false,
            dims=[env.height, env.width],
        )
    else
        scores = model(env.features)
        path = dijkstra_plots(scores; start=env.robot_position, goal=env.target_position)
    end
    plt = plot_grid_with_path(env.cell_costs, path, env.robot_position, env.target_position)
    return plt
end

## Model definitions

# Define the initial actor model
p(θ, stdev) = MvNormal(θ, stdev * I)

function GSPP_model(; num_features=7, seed=0)
    Random.seed!(seed)
    return Chain(
        Dense(num_features => 1; bias=true), x -> -abs.(x), z -> dropdims(z; dims=1)
    )
end

# Define the initial critic model
function GSPP_critic(; num_features=8, seed=0)
    Random.seed!(seed)
    return Chain(Dense(num_features => 1; bias=true), x -> sum(x))
end

# Compute the expert solution for the GridWorld environment
function expert_solution(env::GridWorld, episodes; first_seed=1, create_dataset=false)
    costs = zeros(episodes)
    dataset = []
    for e in 1:episodes
        reset!(env; seed=first_seed + e - 1)
        cost_eps = 0.0
        training_instances = []
        while env.step <= env.max_steps
            path = path_expert(env)
            push!(
                training_instances,
                (
                    feat=env.features,
                    start=env.robot_position,
                    goal=env.target_position,
                    sol=path,
                ),
            )
            cost_eps += step!(env, path)
        end
        costs[e] = cost_eps
        push!(dataset, training_instances)
    end
    if create_dataset
        return mean(costs), costs, dataset
    else
        return mean(costs), costs
    end
end

# Testing function for the greedy heuristic
function greedy_heuristic(env::GridWorld, episodes; first_seed=1)
    costs = zeros(episodes)
    for e in 1:episodes
        reset!(env; seed=first_seed + e - 1)
        cost_eps = 0.0
        while env.step <= env.max_steps
            path = path_straight(
                env.robot_position,
                env.target_position;
                matrix=true,
                dims=[env.height, env.width],
            )
            cost_eps += step!(env, path)
        end
        costs[e] = cost_eps
    end
    return mean(costs), costs
end

# Testing function for the learned models
function val_test(env::GridWorld, model, episodes; first_seed=1)
    costs = zeros(episodes)
    for e in 1:episodes
        reset!(env; seed=first_seed + e - 1)
        cost_eps = 0.0
        while env.step <= env.max_steps
            θ = model(env.features)
            path = path_dijkstra(θ; start=env.robot_position, goal=env.target_position)
            cost_eps += step!(env, path)
        end
        costs[e] = cost_eps
    end
    return mean(costs), costs
end

## Helper functions for algorithms

# Reward comparison function
function reward_comparison(train_rew, val_rew; minim=1)
    means = [(train_rew[i] + val_rew[i]) / 2 for i in eachindex(train_rew)]
    length(means) >= minim ? means = means[minim:end] : nothing
    last_mean = means[end]  # Mean of the last two elements
    return last_mean == maximum(means)  # Check if it's the smallest mean
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

# Perturbation function
function perturber(θ, sigma, no_samples)
    return -abs.(
        reshape(rand(p(reshape(θ, length(θ)), sigma), no_samples), (no_samples, size(θ)...))
    )
end

# Compute features for critic
function features_critic(env::GridWorld, features, param, action)
    return cat(features, fill(param, 1, env.height, env.width); dims=1) .*
           reshape(action, 1, env.height, env.width)
end

# Episode generation
function generate_episode(env::GridWorld, model, sigma, seed; rew_factor=1)
    reset!(env; seed=seed)
    buffer = []
    while env.step <= env.max_steps
        θ = model(env.features)
        η = perturber(θ, sigma, 1)[1, :, :]
        path = path_dijkstra(η; start=env.robot_position, goal=env.target_position)
        traj_1 = (
            feat=env.features,
            start=env.robot_position,
            goal=env.target_position,
            theta=θ,
            eta=η,
            param=env.cost_param,
        )
        cost = step!(env, path) * rew_factor
        traj_2 = (
            next_feat=env.features,
            next_start=env.robot_position,
            next_goal=env.target_position,
            next_param=env.cost_param,
            rew=cost[end],
            ret=0.0,
        )
        push!(buffer, merge(traj_1, traj_2))
    end
    for i in length(buffer):-1:1
        if i == length(buffer)
            ret = buffer[i].rew
        else
            ret = buffer[i].rew + 0.99 * buffer[i + 1].ret
        end
        traj = buffer[i]
        traj_updated = (; traj..., ret=ret)
        buffer[i] = traj_updated
    end
    return buffer
end

# Critic update
function critic_update!(
    env::GridWorld, model, critic, c_target, batch, opt_c; critic_mode="TD_0"
)
    c_features = []
    targets = []
    for tr in batch
        action = path_dijkstra(model(tr.feat); start=tr.start, goal=tr.goal)
        push!(c_features, features_critic(env, tr.feat, tr.param, action))
        if critic_mode == "TD_0"
            next_action = path_dijkstra(
                model(tr.next_feat); start=tr.next_start, goal=tr.next_goal
            )
            nextc_features = features_critic(env, tr.next_feat, tr.next_param, next_action)
            push!(targets, tr.rew + 0.99 * c_target(nextc_features))
        else
            push!(targets, tr.ret)
        end
    end
    grads = gradient(Flux.params(critic)) do
        error = [critic(c_features[j]) for j in eachindex(batch)] .- targets
        quadratic = 0.5 .* error .^ 2
        linear = 1.0 .* (abs.(error) .- 0.5 .* 1.0)
        return mean(ifelse.(abs.(error) .<= 1.0, quadratic, linear))
    end
    return Flux.update!(opt_c, Flux.params(critic), grads)
end
