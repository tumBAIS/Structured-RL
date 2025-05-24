include("utils/utils.jl")

# SIL setup
_, __, dataset = expert_solution(env, 100; first_seed=1, create_dataset=true);

# SIL training function
function SIL_GSPP(env::GridWorld, epochs, data; lr=1e-4, seed=0)
    # Initialize training
    loss = FenchelYoungLoss(PerturbedMultiplicative(path_dijkstra; ε=0.01, nb_samples=20))
    opt = ADAM(lr)
    model = GSPP_model(; seed=seed)

    train_rewards = Float64[]
    val_rewards = Float64[]
    losses = Float64[]
    best_model = deepcopy(model)
    best_epoch = 0

    for epoch in 1:epochs
        i = 4
        for d in data
            # Test model
            if i % 4 == 0 # To account for equal number of episodes
                push!(train_rewards, val_test(env, model, 50; first_seed=1)[1])
                push!(val_rewards, val_test(env, model, 50; first_seed=1001)[1])
                @info epoch, "train:", train_rewards[end], "val:", val_rewards[end]
                if reward_comparison(train_rewards, val_rewards; minim=Int(round(100))) &&
                    epoch > epochs / 2 # Ensure no "lucky shot"
                    best_model = deepcopy(model)
                    best_epoch = epoch
                end
            end
            i += 1
            # Train model
            for j in d
                loss_temp = 0.0
                grads = gradient(Flux.params(model)) do
                    return loss(model(j.feat), j.sol; start=j.start, goal=j.goal)
                end
                Flux.update!(opt, Flux.params(model), grads)
                push!(losses, loss_temp)
            end
        end
    end

    # Final tests
    push!(train_rewards, val_test(env, best_model, 100; first_seed=1)[1])
    push!(val_rewards, val_test(env, best_model, 100; first_seed=1001)[1])
    @info "final train:",
    train_rewards[end], "final val:", val_rewards[end], "best_epoch:",
    best_epoch
    return best_model, train_rewards, val_rewards, losses
end

# Train SIL
SIL_model, SIL_train, SIL_val, SIL_losses = SIL_GSPP(env, 8, dataset; lr=1e-4)

# Test the trained model
SIL_final_train_mean, SIL_final_train_rew = val_test(env, SIL_model, 100; first_seed=1)
SIL_final_test_mean, SIL_final_test_rew = val_test(env, SIL_model, 100; first_seed=2001)

# Plot the train and validation rewards
gspp_SIL_rew_line = plot(SIL_train; label="train history", title="GSPP SIL", marker=:o)
plot!(gspp_SIL_rew_line, SIL_val; label="val history", marker=:o)
savefig(gspp_SIL_rew_line, joinpath(plotdir, "gspp_SIL_rew_line.pdf"))

# Plot the SIL path
plot_paths(env, SIL_model)

# Save the model and rewards
jldsave(
    joinpath(logdir, "gspp_SIL_training_results.jld2");
    model=SIL_model,
    train_rew=SIL_train,
    val_rew=SIL_val,
    train_final=SIL_final_train_rew,
    test_final=SIL_final_test_rew,
)
