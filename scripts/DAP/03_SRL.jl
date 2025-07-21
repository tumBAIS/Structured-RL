include("utils/utils.jl")

# Env initialization
env = DAP(20, 2, 4, 200)

initial_model = load(joinpath(logdir, "DAP_initial_model.jld2"))["actor"]
critic_1 = load(joinpath(logdir, "DAP_critic_rew.jld2"))["critic"]
critic_2 = load(joinpath(logdir, "DAP_critic_fut.jld2"))["critic"]

# SRL setup
critic_rew = deepcopy(critic_1)
critic_fut = deepcopy(critic_2)
p(θ, stdev) = MvNormal(θ, stdev * I)
regularized_predictor = PerturbedAdditive(
    DAP_optimization; ε=1.0, nb_samples=20, is_parallel=true
)
loss = FenchelYoungLoss(regularized_predictor)
SRL_model = deepcopy(initial_model)

# SRL training function
function SRL_dynamic(
    env::DAP,
    model,
    critic_rew,
    critic_fut,
    customer_model;
    episodes=200,
    iterations=100,
    batch_size=4,
    sigmaF_values=[2.0, 1.0],
    sigmaB_values=[2.0, 1.0],
    lr_values=[1e-3, 5e-4],
    dynamic=true,
    temp_values=[1.0, 1.0],
)
    # Initialize training
    opt_a = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c1 = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    opt_c2 = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / episodes
    sigmaB = sigmaB_values[1]
    sigmaB_step = (sigmaB_values[1] - sigmaB_values[2]) / episodes
    lr_step = (lr_values[1] - lr_values[2]) / episodes
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / episodes

    train_rewards = Float64[]
    val_rewards = Float64[]
    losses = Float64[] # Can be used to store actor or critic losses
    best_model = deepcopy(model)
    best_episode = 0
    replay_buffer = []
    rb_position = 1
    rb_size = 0
    no_samples = 40

    for e in 1:episodes
        # Test model
        push!(train_rewards, val_test(env, model, customer_model, 30; first_seed=1)[1])
        push!(val_rewards, val_test(env, model, customer_model, 30; first_seed=1001)[1])
        @info e, "sigmaF:", sigmaF,
        "lr:", opt_a.os[2].eta,
        "temp:", temp,
        "train:", train_rewards[end],
        "val:", val_rewards[end]
        if reward_comparison(train_rewards, val_rewards)
            best_model = deepcopy(model)
            best_episode = e
        end

        # Collect experience
        experience = generate_episode(
            env, model, critic_rew, critic_fut, customer_model, sigmaF, e % 100 + 1
        )
        replay_buffer, rb_position, rb_size = rb_add(
            replay_buffer, experience, 8000, rb_position, rb_size
        )
        batches = [rb_sample(replay_buffer, batch_size) for j in 1:iterations]
        shuffled_exp = Flux.DataLoader(
            experience; batchsize=batch_size, shuffle=true, rng=MersenneTwister(0)
        )

        for i in 1:iterations
            # Train actor
            S_best = []
            for j in 1:batch_size
                # Perturb and sample candidate actions
                data = batches[i][j]
                θ = model(data.feat_t)
                η = rand(MersenneTwister(e * i), p(θ, sigmaB), no_samples)
                sequences = [DAP_optimization(θ; env=env)]
                values = if dynamic
                    [
                        critic_reward(data.feat_t, sequences[end], critic_rew) +
                        critic_future(data.feat_t, sequences[end], critic_fut),
                    ]
                else
                    [critic_reward(data.feat_t, sequences[end], critic_rew)]
                end
                for k in 1:no_samples
                    push!(sequences, DAP_optimization(η[:, k]; env=env))
                    if dynamic
                        push!(
                            values,
                            critic_reward(data.feat_t, sequences[end], critic_rew) +
                            critic_future(data.feat_t, sequences[end], critic_fut),
                        )
                    else
                        push!(
                            values, critic_reward(data.feat_t, sequences[end], critic_rew)
                        )
                    end
                end
                # Compute target action
                values = values ./ temp
                lse = logsumexp(values)
                probs = exp.(values .- lse)
                best_action = sum(probs .* sequences)
                if any(isnan.(best_action))
                    best_action = sequences[argmax(values)]
                else
                    nothing
                end
                push!(S_best, best_action)
            end
            # Update actor using Fenchel-Young loss
            l_fyl = 0.0
            grads = gradient(Flux.params(model)) do
                for j in 1:batch_size
                    l_fyl += loss(model(batches[i][j].feat_t), S_best[j]; env=env)
                end
                return l_fyl
            end
            Flux.update!(opt_a, Flux.params(model), grads)

            # Train static critic
            grads = gradient(Flux.params(critic_rew)) do
                error =
                    [
                        critic_reward(batches[i][j].feat_t, batches[i][j].S_t, critic_rew) for j in 1:batch_size
                    ] .- getfield.(batches[i], :rev_t)
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

        # Update sigmas, learning rates, and temperature
        sigmaF = max(sigmaF - sigmaF_step, sigmaF_values[2])
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[2])
        lr = opt_a.os[2].eta
        opt_a.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c1.os[2].eta = max(lr - lr_step, lr_values[2])
        opt_c2.os[2].eta = max(lr - lr_step, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
    end

    # Final tests
    push!(train_rewards, val_test(env, best_model, customer_model, 100; first_seed=1)[1])
    push!(val_rewards, val_test(env, best_model, customer_model, 100; first_seed=1001)[1])
    @info "final train:",
    train_rewards[end], "final val:", val_rewards[end], "best_episode:",
    best_episode
    return best_model, train_rewards, val_rewards, losses
end

# Train SRL
SRL_model, train_hist, val_hist, losses = SRL_dynamic(
    env,
    SRL_model,
    critic_rew,
    critic_fut,
    customer_model;
    episodes=200,
    iterations=100,
    batch_size=4,
    sigmaF_values=[2.0, 1.0],
    sigmaB_values=[2.0, 1.0],
    lr_values=[1e-3, 5e-4],
    dynamic=true,
    temp_values=[1.0, 1.0],
);

# Test the trained model
SRL_final_train, SRL_final_train_rew = val_test(env, SRL_model, customer_model, 100; first_seed=1)
SRL_final_test, SRL_final_test_rew = val_test(env, SRL_model, customer_model, 100; first_seed=2001)

# Plot the train and validation rewards
dap_SRL_rew_line = plot(SRL_train; label="train history", title="DAP SRL", marker=:o)
plot!(dap_SRL_rew_line, SRL_val; label="val history", marker=:o)
savefig(dap_SRL_rew_line, joinpath(plotdir, "dap_SRL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "dap_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    val_rew=SRL_val,
    train_final=SRL_final_train_rew,
    test_final=SRL_final_test_rew,
)
