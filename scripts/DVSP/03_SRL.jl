include("utils/utils.jl")
include("utils/policy.jl")

# Training preparation
Random.seed!(0)
actor_model = Chain(Dense(14 => 1; bias=false), vec)
critic_model = critic_GNN(15, 1)

policy = CombinatorialACPolicy(;
    actor_model, critic_model, p, CO_layer=prize_collecting_vsp, seed=0
)

model_builder = grb_model

# SRL training function
function SRL_training(
    π::CombinatorialACPolicy,
    train_envs,
    val_envs;
    grad_steps=400,
    collection_steps=20,
    iterations=100,
    batch_size=4,
    ntests=1,
    sigmaF=0.1,
    sigmaB_values=[1.0, 0.1],
    lr_values=[1e-3, 2e-4],
    critic_factor=2,
    temp_values=[10.0, 10.0],
    kwargs...,
)
    # Initialize training
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
    target_critic = deepcopy(π.critic_model)

    global sigmaF_dvsp = sigmaF
    sigmaB = sigmaB_values[1]
    sigmaB_step = (sigmaB_values[1] - sigmaB_values[2]) / grad_steps
    lr_step_a = (lr_values[1] - lr_values[2]) / grad_steps
    lr_step_c = (lr_values[1] * critic_factor - lr_values[2]) / grad_steps
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / grad_steps

    # Initialize replay buffer
    replay_buffer = []
    rb_capacity = collection_steps * 6 * 1000
    rb_position = 1
    rb_size = 0

    reset_seed!(π)
    for e in 1:grad_steps

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
        "sigmaB:",
        sigmaB,
        "lr:",
        opt_actor.os[2].eta,
        "train:",
        train_reward_history[end],
        "val:",
        val_reward_history[end]

        # Collect experience
        episodes = SRL_episodes(π, train_envs, collection_steps; kwargs...)
        replay_buffer, rb_position, rb_size = rb_add(
            replay_buffer, rb_capacity, rb_position, rb_size, episodes
        )
        batches = [rb_sample(replay_buffer, batch_size) for j in 1:iterations]

        for batch in batches
            # Train critic
            rewards = getfield.(batch, :reward)
            next_states = getfield.(batch, :next_state)
            next_embeddings = getfield.(batch, :next_s)
            next_embeds_c = getfield.(batch, :next_s_c)
            critic_target =
                rewards .+
                γ .* [
                    V_value_GNN(
                        π,
                        next_embeddings[j],
                        next_embeds_c[j],
                        target_critic,
                        "off_policy";
                        instance=next_states[j],
                        kwargs...,
                    ) for j in eachindex(batch)
                ]
            graphs, s_c, edge_features = grads_prep_GNN(batch)
            grads_critic = Flux.gradient(Flux.params(π.critic_model)) do
                return huber_GNN(π, graphs, s_c, edge_features, critic_target, 1.0; kwargs...)
            end
            Flux.update!(opt_critic, Flux.params(π.critic_model), grads_critic)

            # Train actor
            opt_actions = SRL_actions(π, batch; sigmaB, no_samples=40, temp=temp, kwargs...)
            grads_actor = Flux.gradient(Flux.params(π.actor_model)) do
                return J_SRL(π, batch, opt_actions; kwargs...)
            end
            Flux.update!(opt_actor, Flux.params(π.actor_model), grads_actor)
        end
        # Update target critic
        target_critic = deepcopy(π.critic_model)
        # Update sigmaB, learning rates, and temperature
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[2])
        lr_a = opt_actor.os[2].eta
        lr_c = opt_critic.os[2].eta
        opt_actor.os[2].eta = max(lr_a - lr_step_a, lr_values[2])
        opt_critic.os[2].eta = max(lr_c - lr_step_c, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
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

# Train SRL
SRL_model, SRL_train, SRL_val, SRL_losses = SRL_training(
    policy,
    train_envs,
    val_envs;
    grad_steps=400,
    collection_steps=20,
    iterations=100,
    batch_size=4,
    ntests=1,
    sigmaF=0.1,
    sigmaB_values=[1.0, 0.1],
    lr_values=[1e-3, 2e-4],
    critic_factor=2,
    temp_values=[10.0, 10.0],
    model_builder,
);

# Test the trained model
SRL_policy_evaluation = KleopatraVSPPolicy(SRL_model)
SRL_final_train, SRL_final_train_rew =
    .-evaluate_policy(SRL_policy_evaluation, train_envs; nb_episodes=10, model_builder)
SRL_final_test, SRL_final_test_rew =
    .-evaluate_policy(SRL_policy_evaluation, test_envs; nb_episodes=10, model_builder)

# Plot the train and validation rewards
dvsp_SRL_rew_line = plot(SRL_train; label="train history", title="DVSP SRL", marker=:o)
plot!(dvsp_SRL_rew_line, SRL_val; label="val history", marker=:o)
savefig(dvsp_SRL_rew_line, "plots/dvsp_SRL_rew_line.pdf")

# Save the model and rewards
jldsave(
    joinpath(logdir, "dvsp_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    val_rew=SRL_val,
    train_final=SRL_final_train_rew,
    test_final=SRL_final_test_rew,
)
