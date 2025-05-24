include("utils/utils.jl")

initial_model = load(joinpath(logdir, "DAP_initial_model.jld2"))["actor"]
critic_1 = load(joinpath(logdir, "DAP_critic_rew.jld2"))["critic"]
critic_2 = load(joinpath(logdir, "DAP_critic_fut.jld2"))["critic"]

# PPO setup
critic_rew = deepcopy(critic_1)
critic_fut = deepcopy(critic_2)
p(θ, stdev) = MvNormal(θ, stdev * I)
PPO_model = deepcopy(initial_model)

# PPO training function
function PPO_dynamic(
    env::DAP,
    model,
    ritic_rew,
    critic_fut,
    customer_model;
    episodes=200,
    iterations=100,
    batch_size=4,
    sigmaF_values=[0.1, 0.05],
    lr_values=[5e-3, 5e-3],
    clip=0.2,
    dynamic=false,
)
    # Initialize training
    opt_a = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c1 = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c2 = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / episodes
    sigmaF_avg = (sigmaF_values[1] + sigmaF_values[2]) / 2
    lr_step = (lr_values[1] - lr_values[2]) / episodes

    train_rewards = Float64[]
    val_rewards = Float64[]
    losses = Float64[] # Can be used to store actor or critic losses
    best_model = deepcopy(model)
    best_episode = 0
    replay_buffer = []
    rb_position = 1
    rb_size = 0

    for e in 1:episodes
        # Test model
        push!(train_rewards, val_test(env, model, customer_model, 30; first_seed=1)[1])
        push!(val_rewards, val_test(env, model, customer_model, 30; first_seed=1001)[1])
        @info e,
        "sigmaF:", sigmaF, "lr:", opt_a.os[2].eta, "train:", train_rewards[end], "val:",
        val_rewards[end]
        if reward_comparison(train_rewards, val_rewards)
            best_model = deepcopy(model)
            best_episode = e
        end

        # Collect experience
        experience = generate_episode(
            env, model, ritic_rew, critic_fut, customer_model, sigmaF, e % 100 + 1
        )
        replay_buffer, rb_position, rb_size = rb_add(
            replay_buffer, experience, 1600, rb_position, rb_size
        )
        batches = [rb_sample(replay_buffer, batch_size) for j in 1:iterations]
        shuffled_exp = Flux.DataLoader(
            experience; batchsize=batch_size, shuffle=true, rng=MersenneTwister(0)
        )

        for i in 1:iterations
            # Train actor
            batch = batches[i]
            Q_values = [
                critic_reward(batch[j].feat_t, batch[j].S_t, critic_rew) for
                j in 1:batch_size
            ]
            S_θ = [DAP_optimization(batch[j].theta; env=env) for j in 1:batch_size]
            V_values = [
                critic_reward(batch[j].feat_t, S_θ[j], critic_rew) for j in 1:batch_size
            ]
            advantages = Q_values .- V_values
            advantages = getfield.(batch, :ret_t) .- V_values
            grads = gradient(Flux.params(model)) do
                old_probs = [
                    pdf(p(batch[j].theta, sigmaF_avg), batch[j].eta) for j in 1:batch_size
                ]
                new_probs = [
                    pdf(p(model(batch[j].feat_t), sigmaF_avg), batch[j].eta) for
                    j in 1:batch_size
                ]
                ratio_unclipped = new_probs ./ old_probs
                ratio_clipped = clamp.(ratio_unclipped, 1 - clip, 1 + clip)
                return -mean(
                    min.(ratio_unclipped .* advantages, ratio_clipped .* advantages)
                )
            end
            Flux.update!(opt_a, Flux.params(model), grads)

            # Train static critic
            grads = gradient(Flux.params(critic_rew)) do
                error =
                    [
                        critic_reward(batch[j].feat_t, batch[j].S_t, critic_rew) for
                        j in 1:batch_size
                    ] .- getfield.(batch, :rev_t)
                quadratic = 0.5 .* error .^ 2
                linear = 1.0 .* (abs.(error) .- 0.5 .* 1.0)
                return mean(ifelse.(abs.(error) .<= 1.0, quadratic, linear))
            end
            Flux.update!(opt_c1, Flux.params(critic_rew), grads)
        end

        # Train dynamic critic
        if dynamic
            for i in 1:Int(round(iterations / 2))
                for batch in shuffled_exp
                    if length(batch) == batch_size
                        critic_target = [
                            batch[j].ret_t - batch[j].rev_t for j in 1:batch_size
                        ]
                        grads = gradient(Flux.params(critic_fut)) do
                            error =
                                [
                                    critic_future(
                                        batch[j].feat_t, batch[j].S_t, critic_fut
                                    ) for j in 1:batch_size
                                ] .- critic_target
                            quadratic = 0.5 .* error .^ 2
                            linear = 1.0 .* (abs.(error) .- 0.5 .* 1.0)
                            return mean(ifelse.(abs.(error) .<= 1.0, quadratic, linear))
                        end
                        Flux.update!(opt_c2, Flux.params(critic_fut), grads)
                    end
                end
            end
        end

        # Update sigmaF and learning rates
        sigmaF = max(sigmaF - sigmaF_step, sigmaF_values[2])
        lr = opt_a.os[2].eta
        opt_a.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c1.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c2.os[2].eta = max(lr - lr_step, lr_values[2])
    end

    # Final tests
    push!(train_rewards, val_test(env, best_model, customer_model, 30; first_seed=1)[1])
    push!(val_rewards, val_test(env, best_model, customer_model, 30; first_seed=1001)[1])
    @info "final train:",
    train_rewards[end], "final val:", val_rewards[end], "best_episode:",
    best_episode
    return best_model, train_rewards, val_rewards, losses
end

# Train PPO
PPO_model, train_hist, val_hist, losses = PPO_dynamic(
    env,
    PPO_model,
    critic_rew,
    critic_fut,
    customer_model;
    episodes=200,
    iterations=100,
    batch_size=4,
    sigmaF_values=[0.1, 0.05],
    lr_values=[5e-3, 5e-3],
    clip=0.2,
    dynamic=false,
);

# Test the trained model
PPO_final_train, PPO_final_train_rew = val_test(env, PPO_model, customer_model, 100; first_seed=1)
PPO_final_test, PPO_final_test_rew = val_test(env, PPO_model, customer_model, 100; first_seed=2001)

# Plot the train and validation rewards
dap_PPO_rew_line = plot(PPO_train; label="train history", title="DAP PPO", marker=:o)
plot!(dap_PPO_rew_line, PPO_val; label="val history", marker=:o)
savefig(dap_PPO_rew_line, joinpath(plotdir, "dap_PPO_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "dap_PPO_training_results.jld2");
    model=PPO_model,
    train_rew=PPO_train,
    val_rew=PPO_val,
    train_final=PPO_final_train_rew,
    test_final=PPO_final_test_rew,
)
