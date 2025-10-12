include("utils/utils.jl")

# Env initialization
env = GridWorld();

# SRL update function
function SRL_update!(
    env::GridWorld,
    model,
    critic_1,
    critic_2,
    batch,
    loss,
    opt_a;
    no_samples=40,
    sigmaB=0.5,
    temp=1.0,
)
    A_target = []
    for tr in batch
        # Perturb and sample candidate actions
        θ = model(tr.feat)
        η = perturber(θ, sigmaB, no_samples)
        actions = [path_dijkstra(θ; start=tr.start, goal=tr.goal)]
        c_features = features_critic(env, tr.feat, tr.param, actions[end])
        values = [middle(critic_1(c_features), critic_2(c_features))]
        for k in 1:no_samples
            push!(actions, path_dijkstra(η[k, :, :]; start=tr.start, goal=tr.goal))
            c_features = features_critic(env, tr.feat, tr.param, actions[end])
            push!(values, middle(critic_1(c_features), critic_2(c_features)))
        end
        # Compute target action
        values = values ./ temp
        lse = logsumexp(values)
        probs = exp.(values .- lse)
        best_action = sum(probs .* actions)
        any(isnan.(best_action)) ? best_action = actions[argmax(values)] : nothing
        push!(A_target, best_action)
    end
    # Update actor using Fenchel-Young loss
    l_fyl = 0.0
    grads = gradient(Flux.params(model)) do
        for j in eachindex(batch)
            l_fyl += loss(
                model(batch[j].feat),
                A_target[j];
                start=batch[j].start,
                goal=batch[j].goal,
            )
            return l_fyl
        end
    end
    return Flux.update!(opt_a, Flux.params(model), grads)
end

# SRL training function
function SRL_GSPP(
    env::GridWorld;
    episodes=200,
    iterations=100,
    batch_size=4,
    critic_eps=40,
    sigmaF_values=[0.05, 0.05],
    sigmaB_values=[0.05, 0.05],
    lr_values=[1e-3, 5e-4],
    temp_values=[0.1, 0.1],
    critic_mode="TD_0",
    seed=0
)
    # Initialize training
    loss = FenchelYoungLoss(PerturbedMultiplicative(path_dijkstra; ε=0.01, nb_samples=20))
    model = GSPP_model(; seed=seed)
    critic_1 = GSPP_critic(; seed=seed)
    critic_2 = GSPP_critic(; seed=seed+1)
    c1_target = deepcopy(critic_1)
    c2_target = deepcopy(critic_2)
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
    rb_capacity = 10000
    no_samples = 40

    for e in 1:episodes
        # Test model
        push!(train_rewards, val_test(env, model, 50; first_seed=1)[1],)
        push!(val_rewards, val_test(env, model, 50; first_seed=1001)[1],)
        @info e, "sigmaF:", sigmaF, "lr:", opt_a.os[2].eta, "temp:", temp, "train:", train_rewards[end], "val:", val_rewards[end]
        if reward_comparison(train_rewards, val_rewards)
            best_model = deepcopy(model)
            best_episode = e
        end

        # Collect experience
        experience = generate_episode(env, model, sigmaF, e%100+1)
        replay_buffer, rb_position, rb_size = rb_add(replay_buffer, experience, rb_capacity, rb_position, rb_size)
        batches = [rb_sample(replay_buffer, batch_size) for j in 1:iterations]
        shuffled_exp = Flux.DataLoader(experience, batchsize = batch_size, shuffle = true, rng=MersenneTwister(0))

        # Train actor
        for batch in batches
            if e > critic_eps
                SRL_update!(env, model, critic_1, critic_2, batch, loss, opt_a; no_samples=no_samples, sigmaB=sigmaB, temp=temp)
            end
        end

        # Train critics
        if critic_mode == "TD_0"
            for batch in batches
                critic_update!(env, model, critic_1, c1_target, batch, opt_c1; critic_mode=critic_mode)
                critic_update!(env, model, critic_2, c2_target, batch, opt_c2; critic_mode=critic_mode)
            end
        else
            for batch in shuffled_exp
                critic_update!(env, model, critic_1, c1_target, batch, opt_c1; critic_mode=critic_mode)
                critic_update!(env, model, critic_2, c2_target, batch, opt_c2; critic_mode=critic_mode)
            end
        end
        
        # Update target critics
        c1_target = deepcopy(critic_1)
        c2_target = deepcopy(critic_2)
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
    push!(train_rewards, val_test(env, best_model, 100; first_seed=1)[1],)
    push!(val_rewards, val_test(env, best_model, 100; first_seed=1001)[1],)
    @info "final train:", train_rewards[end], "final val:", val_rewards[end], "best_episode:", best_episode
    return best_model, train_rewards, val_rewards, losses
end

# Train SRL
SRL_model, SRL_train, SRL_val, SRL_losses = SRL_GSPP(
    env;
    episodes=200,
    iterations=100,
    batch_size=4,
    critic_eps=40,
    sigmaF_values=[0.05, 0.05],
    sigmaB_values=[0.05, 0.05],
    lr_values=[1e-3, 5e-4],
    temp_values=[0.1, 0.1],
    critic_mode="TD_0",
)

# Test the trained model
SRL_final_train_mean, SRL_final_train_rew = val_test(env, SRL_model, 100; first_seed=1)
SRL_final_test_mean, SRL_final_test_rew = val_test(env, SRL_model, 100; first_seed=2001)

# Plot the train and validation rewards
gspp_SRL_rew_line = plot(SRL_train; label="train history", title="GSPP SRL", marker=:o)
plot!(gspp_SRL_rew_line, SRL_val; label="val history", marker=:o)
savefig(gspp_SRL_rew_line, joinpath(plotdir, "gspp_SRL_rew_line.pdf"))

# Plot the SRL path
plot_paths(env, SRL_model)

# Save the model and rewards
jldsave(
    joinpath(logdir, "gspp_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    val_rew=SRL_val,
    train_final=SRL_final_train_rew,
    test_final=SRL_final_test_rew,
)
