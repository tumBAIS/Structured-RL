include("utils/utils.jl")
include("utils/policy.jl")

# Training preparation
Random.seed!(0)
actor_model = Chain(Dense(14 => 1; bias=false), vec)
critic_model = critic_GNN(15, 1)

PPO_policy = CombinatorialACPolicy(;
    actor_model, critic_model, p, CO_layer=prize_collecting_vsp, seed=0
)

model_builder = grb_model

# PPO training function
function PPO_training(
    π::CombinatorialACPolicy,
    train_envs,
    val_envs;
    episodes=400,
    collection_steps=20,
    epochs=100,
    batch_size=1,
    ntests=1,
    clip=0.2,
    use_rb=true,
    sigmaF_values=[0.5, 0.05],
    lr_values=[1e-3, 5e-4],
    V_method="off_policy",
    adv_method="TD_n",
    critic_factor=10,
    kwargs...,
)
    # Initialize training
    (; actor_model, critic_model) = π
    opt_actor = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_critic = Optimiser(ClipValue(1e-3), Adam(lr_values[1] * critic_factor))
    train_reward_history = Float64[]
    val_reward_history = Float64[]
    actor_weights_history = Matrix{Float32}[]
    loss_history = Float64[] # Can be used to store actor or critic losses

    γ = 1.0
    best_model = deepcopy(π.actor_model)
    best_performance = -Inf
    best_episode = 0
    target_critic = deepcopy(critic_model)

    global sigmaF_dvsp = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / episodes
    sigmaF_average = 2
    lr_step_a = (lr_values[1] - lr_values[2]) / episodes
    lr_step_c = (lr_values[1] * critic_factor - lr_values[2]) / episodes

    # Initialize replay buffer if it is used
    if use_rb
        replay_buffer = []
        rb_capacity = collection_steps * 6 * 100
        rb_position = 1
        rb_size = 0
        iterations = epochs
        epochs = 1
    end

    reset_seed!(π)
    for e in 1:episodes
        # Test model
        push!(
            train_reward_history,
            evaluate_policy(
                π,
                train_envs;
                nb_episodes=ntests,
                rng=MersenneTwister(0),
                perturb=false,
                kwargs...,
            ),
        )
        push!(
            val_reward_history,
            evaluate_policy(
                π,
                val_envs;
                nb_episodes=ntests,
                rng=MersenneTwister(0),
                perturb=false,
                kwargs...,
            ),
        )
        push!(actor_weights_history, copy(π.actor_model[1].weight))

        # Save currently best model
        if val_reward_history[end] >= best_performance
            best_performance = val_reward_history[end]
            best_model = deepcopy(π.actor_model)
            best_episode = e
        end

        @info e,
        "sigmaF:",
        sigmaF_dvsp,
        "lr",
        opt_actor.os[2].eta,
        "train:",
        train_reward_history[end],
        "val:",
        val_reward_history[end]

        # Collect experience
        episodes = PPO_episodes(
            π,
            target_critic,
            train_envs,
            collection_steps,
            γ,
            V_method,
            adv_method;
            kwargs...,
        )
        if use_rb
            replay_buffer, rb_position, rb_size = rb_add(
                replay_buffer, rb_capacity, rb_position, rb_size, episodes
            )
            batches = [rb_sample(replay_buffer, batch_size) for j in 1:iterations]
        else
            batches = Flux.DataLoader(
                episodes; batchsize=batch_size, shuffle=true, rng=MersenneTwister(0)
            )
        end

        for i in epochs
            for batch in batches
                # Train critic
                critic_target = getfield.(batch, :Rₜ)
                graphs, s_c, edge_features = grads_prep_GNN(batch)
                grads_critic = Flux.gradient(Flux.params(π.critic_model)) do
                    return huber_GNN(
                        π, graphs, s_c, edge_features, critic_target, 1.0; kwargs...
                    )
                end
                Flux.update!(opt_critic, Flux.params(π.critic_model), grads_critic)

                # Train actor
                grads_actor = Flux.gradient(Flux.params(π.actor_model)) do
                    return -J_PPO(π, batch, clip, sigmaF_average)
                end
                Flux.update!(opt_actor, Flux.params(π.actor_model), grads_actor)
            end
            # Update target critic
            target_critic = deepcopy(π.critic_model)
        end

        # Update sigmaF and learning rates
        global sigmaF_dvsp = max(sigmaF_dvsp - sigmaF_step, sigmaF_values[2])
        lr_a = opt_actor.os[2].eta
        lr_c = opt_critic.os[2].eta
        opt_actor.os[2].eta = max(lr_a - lr_step_a, lr_values[2])
        opt_critic.os[2].eta = max(lr_c - lr_step_c, lr_values[2])
    end
    # Final tests

    π.actor_model = deepcopy(best_model)
    final_test = evaluate_policy(
        π, train_envs; nb_episodes=ntests, rng=MersenneTwister(0), perturb=false, kwargs...
    )
    final_val = evaluate_policy(
        π, val_envs; nb_episodes=ntests, rng=MersenneTwister(0), perturb=false, kwargs...
    )
    @info "final train:", final_test, "final val:", final_val, "best_episode:", best_episode
    push!(train_reward_history, final_test)
    push!(val_reward_history, final_val)
    push!(actor_weights_history, copy(π.actor_model[1].weight))

    return π.actor_model, train_reward_history, val_reward_history, loss_history
end

# Train PPO
PPO_model, PPO_train, PPO_val, PPO_losses = PPO_training(
    PPO_policy,
    train_envs,
    val_envs;
    episodes=400,
    collection_steps=20,
    epochs=100,
    batch_size=1,
    ntests=1,
    clip=0.2,
    use_rb=true,
    sigmaF_values=[0.5, 0.05],
    lr_values=[1e-3, 5e-4],
    V_method="off_policy",
    adv_method="TD_n",
    critic_factor=10,
    model_builder,
);

# Test the trained model
PPO_policy_evaluation = KleopatraVSPPolicy(PPO_model)
PPO_final_train, PPO_final_train_rew =
    .-evaluate_policy(PPO_policy_evaluation, train_envs; nb_episodes=10, model_builder)
PPO_final_test, PPO_final_test_rew =
    .-evaluate_policy(PPO_policy_evaluation, test_envs; nb_episodes=10, model_builder)

# Plot the train and validation rewards
dvsp_PPO_rew_line = plot(PPO_train; label="train history", title="DVSP PPO", marker=:o)
plot!(dvsp_PPO_rew_line, PPO_val; label="val history", marker=:o)
savefig(dvsp_PPO_rew_line, "plots/dvsp_PPO_rew_line.pdf")

# Save the model and rewards
jldsave(
    joinpath(logdir, "dvsp_PPO_training_results.jld2");
    model=PPO_model,
    train_rew=PPO_train,
    val_rew=PPO_val,
    train_final=PPO_final_train_rew,
    test_final=PPO_final_test_rew,
)
