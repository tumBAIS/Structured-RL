include("utils/utils.jl")

# PPO update function
function PPO_update!(
    env::GridWorld,
    model,
    critic_1,
    critic_2,
    batch,
    opt_a;
    sigmaF_avg=0.5,
    clip=0.2,
    adv_mode="TD_n",
)
    # Calculate advantages
    advantages = []
    for tr in batch
        action = path_dijkstra(tr.theta; start=tr.start, goal=tr.goal)
        c_features = features_critic(env, tr.feat, tr.param, action)
        V_value = middle(critic_1(c_features), critic_2(c_features))
        if adv_mode == "TD_n"
            push!(advantages, tr.ret - V_value)
        else
            action = path_dijkstra(tr.eta; start=tr.start, goal=tr.goal)
            c_features = features_critic(env, tr.feat, tr.param, action)
            Q_value = middle(critic_1(c_features), critic_2(c_features))
            push!(advantages, Q_value - V_value)
        end
    end
    # Calculate loss
    grads = gradient(Flux.params(model)) do
        old_probs = [
            pdf(
                p(reshape(tr.theta, length(tr.theta)), sigmaF_avg),
                reshape(tr.eta, length(tr.eta)),
            ) for tr in batch
        ]
        new_probs = [
            pdf(
                p(reshape(model(tr.feat), length(tr.theta)), sigmaF_avg),
                reshape(tr.eta, length(tr.eta)),
            ) for tr in batch
        ]
        ratio_unclipped = new_probs ./ old_probs
        ratio_clipped = clamp.(ratio_unclipped, 1 - clip, 1 + clip)
        return -mean(min.(ratio_unclipped .* advantages, ratio_clipped .* advantages))
    end
    # Update model
    return Flux.update!(opt_a, Flux.params(model), grads)
end

# PPO training function
function PPO_GSPP(
    env::GridWorld;
    episodes=200,
    iterations=100,
    batch_size=1,
    critic_eps=40,
    sigmaF_values=[0.05, 0.05],
    lr_values=[5e-4, 5e-4],
    critic_mode="TD_0",
    seed=0,
)
    # Initialize training
    model = GSPP_model(; seed=seed)
    critic_1 = GSPP_critic(; seed=seed)
    critic_2 = GSPP_critic(; seed=seed + 1)
    c1_target = deepcopy(critic_1)
    c2_target = deepcopy(critic_2)
    opt_a = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c1 = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c2 = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))

    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / episodes
    sigmaF_avg = (sigmaF_values[1] + sigmaF_values[2]) / 2
    sigmaF_avg = 2
    lr_step = (lr_values[1] - lr_values[2]) / episodes
    clip = 0.2

    train_rewards = Float64[]
    val_rewards = Float64[]
    losses = Float64[] # Can be used to store actor or critic losses
    best_model = deepcopy(model)
    best_episode = 0

    replay_buffer = []
    rb_position = 1
    rb_size = 0
    rb_capacity = 2000
    adv_mode = "TD_0"

    for e in 1:episodes
        # Test model
        push!(train_rewards, val_test(env, model, 50; first_seed=1)[1])
        push!(val_rewards, val_test(env, model, 50; first_seed=1001)[1])
        @info e,
        "sigmaF:", sigmaF, "lr:", opt_a.os[2].eta, "train:", train_rewards[end], "val:",
        val_rewards[end]
        if reward_comparison(train_rewards, val_rewards)
            best_model = deepcopy(model)
            best_episode = e
        end

        # Collect experience
        experience = generate_episode(env, model, sigmaF, e % 100 + 1; rew_factor=1)
        replay_buffer, rb_position, rb_size = rb_add(
            replay_buffer, experience, rb_capacity, rb_position, rb_size
        )
        batches = [rb_sample(replay_buffer, batch_size) for j in 1:iterations]
        shuffled_exp = Flux.DataLoader(
            experience; batchsize=batch_size, shuffle=true, rng=MersenneTwister(0)
        )

        # Train actor
        for batch in batches
            if e > critic_eps
                PPO_update!(
                    env,
                    model,
                    critic_1,
                    critic_2,
                    batch,
                    opt_a;
                    sigmaF_avg=sigmaF_avg,
                    clip=clip,
                    adv_mode=adv_mode,
                )
            end
        end

        # Train critics
        if critic_mode == "TD_0"
            for batch in batches
                critic_update!(
                    env, model, critic_1, c1_target, batch, opt_c1; critic_mode=critic_mode
                )
                critic_update!(
                    env, model, critic_2, c2_target, batch, opt_c2; critic_mode=critic_mode
                )
            end
        else
            for batch in shuffled_exp
                critic_update!(
                    env, model, critic_1, c1_target, batch, opt_c1; critic_mode=critic_mode
                )
                critic_update!(
                    env, model, critic_2, c2_target, batch, opt_c2; critic_mode=critic_mode
                )
            end
        end

        # Update target critics
        c1_target = deepcopy(critic_1)
        c2_target = deepcopy(critic_2)
        # Update sigmaF and learning rates
        sigmaF = max(sigmaF - sigmaF_step, sigmaF_values[2])
        lr = opt_a.os[2].eta
        opt_a.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c1.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c2.os[2].eta = max(lr - lr_step, lr_values[2])
    end

    # Final tests
    push!(train_rewards, val_test(env, best_model, 100; first_seed=1)[1])
    push!(val_rewards, val_test(env, best_model, 100; first_seed=1001)[1])
    @info "final train:",
    train_rewards[end], "final val:", val_rewards[end], "best_episode:",
    best_episode
    return best_model, train_rewards, val_rewards, losses
end

# Train PPO
PPO_model, PPO_train, PPO_val, PPO_losses = PPO_GSPP(
    env;
    episodes=200,
    iterations=100,
    batch_size=1,
    critic_eps=40,
    sigmaF_values=[0.05, 0.05],
    lr_values=[5e-4, 5e-4],
    critic_mode="TD_0",
)

# Test the trained model
PPO_final_train_mean, PPO_final_train_rew = val_test(env, PPO_model, 100; first_seed=1)
PPO_final_test_mean, PPO_final_test_rew = val_test(env, PPO_model, 100; first_seed=2001)

# Plot the train and validation rewards
gspp_PPO_rew_line = plot(PPO_train; label="train history", title="GSPP PPO", marker=:o)
plot!(gspp_PPO_rew_line, PPO_val; label="val history", marker=:o)
savefig(gspp_PPO_rew_line, joinpath(plotdir, "gspp_PPO_rew_line.pdf"))

# Plot the PPO path
plot_paths(env, PPO_model)

# Save the model and rewards
jldsave(
    joinpath(logdir, "gspp_PPO_training_results.jld2");
    model=PPO_model,
    train_rew=PPO_train,
    val_rew=PPO_val,
    train_final=PPO_final_train_rew,
    test_final=PPO_final_test_rew,
)
