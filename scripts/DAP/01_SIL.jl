include("utils/utils.jl")

initial_model = load(joinpath(logdir, "DAP_initial_model.jld2"))["actor"]

# SIL setup
_, ___, training_data = expert_policy(
    env, customer_model, 100; first_seed=1, create_dataset=true
);
regularized_predictor = PerturbedAdditive(
    DAP_optimization; ε=1.0, nb_samples=20, is_parallel=true
)
loss = FenchelYoungLoss(regularized_predictor)
SIL_model = deepcopy(initial_model)

# SIL training function
function SIL_dynamic!(env::DAP, model, customer_model; epochs=8, lr_values=[1e-4, 1e-4])
    # Initialize training
    train_rewards, val_rewards = [], []
    losses = Float64[] # Can be used to store losses
    best_model = deepcopy(model)
    best_epoch = 0
    opt = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    lr_step = (lr_values[1] - lr_values[2]) / epochs
    for epoch in 1:epochs
        i = 4
        for e in training_data
            # Test model
            if i % 4 == 0 # To account for equal number of episodes
                push!(train_rewards, val_test(env, model, customer_model, 30; first_seed=1)[1])
                push!(val_rewards, val_test(env, model, customer_model, 30; first_seed=1001)[1])
                @info epoch, "train:", train_rewards[end], "val:", val_rewards[end]
                if reward_comparison(train_rewards, val_rewards; minim=Int(round(100))) && epoch > epochs/2 # Ensure no "lucky shot"
                    best_model = deepcopy(model)
                    best_epoch = epoch
                end
            end
            i += 1
            # Train model
            for (features, S_opt) in e
                grads = gradient(Flux.params(model)) do
                    return loss(model(features), S_opt; env)
                end
                Flux.update!(opt, Flux.params(model), grads)
            end
        end
        # Update learning rate
        lr = opt.os[2].eta
        opt.os[2].eta = max(lr - lr_step, lr_values[2])
    end
    # Final tests
    push!(train_rewards, val_test(env, best_model, customer_model, 30; first_seed=1)[1])
    push!(val_rewards, val_test(env, best_model, customer_model, 30; first_seed=1001)[1])
    @info "final train:",
    train_rewards[end], "final val:", val_rewards[end], "best_epoch:",
    best_epoch
    return best_model, train_rewards, val_rewards, losses
end

# Train SIL
SIL_model, SIL_train, SIL_val, SIL_losses = SIL_dynamic!(
    env, SIL_model, customer_model; epochs=8, lr_values=[1e-4, 1e-4]
)

# Test the trained model
SIL_final_train_mean, SIL_final_train_rew = val_test(env, SIL_model, customer_model, 100; first_seed=1)
SIL_final_test_mean, SIL_final_test_rew = val_test(env, SIL_model, customer_model, 100; first_seed=2001)

# Plot the train and validation rewards
dap_SIL_rew_line = plot(SIL_train; label="train history", title="DAP SIL", marker=:o)
plot!(dap_SIL_rew_line, SIL_val; label="val history", marker=:o)
savefig(dap_SIL_rew_line, joinpath(plotdir, "dap_SIL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "dap_SIL_training_results.jld2");
    model=SIL_model,
    train_rew=SIL_train,
    val_rew=SIL_val,
    train_final=SIL_final_train_rew,
    test_final=SIL_final_test_rew,
)
