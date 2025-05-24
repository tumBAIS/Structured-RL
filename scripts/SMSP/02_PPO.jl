include("utils/utils.jl")

# PPO training function
function PPO_SMSP(;
    grad_steps=2000,
    batch_size=20,
    clip=0.2,
    sigmaF_values=[0.01, 0.01],
    lr_values=[5e-4, 5e-4],
    seed=0,
)
    # Initialize training
    Random.seed!(seed)
    PPO_model = Chain(Dense(nb_features, 1; bias=false), X -> dropdims(X; dims=1))
    opt = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / grad_steps
    sigmaF_avg = (sigmaF_values[1] + sigmaF_values[2]) / 2
    lr_step = (lr_values[1] - lr_values[2]) / grad_steps
    train_rewards = Float64[]
    val_rewards = Float64[]
    best_model = deepcopy(PPO_model)
    best_episode = 0
    losses = [] # # Can be used to store losses
    for e in 1:grad_steps
        # Test model
        push!(
            train_rewards,
            mean([
                evaluate_solution_1_rj_sumCj(
                    data_train[j][3],
                    embedding_to_sequence(ranking(PPO_model(data_train[j][1]))),
                ) for j in eachindex(data_train)
            ]),
        )
        push!(
            val_rewards,
            mean([
                evaluate_solution_1_rj_sumCj(
                    data_val[j][3],
                    embedding_to_sequence(ranking(PPO_model(data_val[j][1]))),
                ) for j in eachindex(data_val)
            ]),
        )
        @info e,
        "sigmaF:", sigmaF, "lr", opt.os[2].eta, "train:", train_rewards[end], "val:",
        val_rewards[end]
        if val_rewards[end] == minimum(val_rewards)
            best_model = deepcopy(PPO_model)
            best_episode = e
        end
        batches = Flux.DataLoader(data_train; batchsize=batch_size, shuffle=true)
        # Train model
        for batch in batches
            thetas = []
            etas = []
            advantages = []
            # Calculate advantages
            for b in batch
                push!(thetas, PPO_model(b[1]))
                val_theta = evaluate_solution_1_rj_sumCj(
                    b[3], embedding_to_sequence(ranking(thetas[end]))
                )
                push!(etas, rand(rng, p(thetas[end], sigmaF)))
                val_eta = evaluate_solution_1_rj_sumCj(
                    b[3], embedding_to_sequence(ranking(etas[end]))
                )
                push!(advantages, val_theta - val_eta)
            end
            # Update model
            grads = gradient(Flux.params(PPO_model)) do
                old_probs = [pdf(p(thetas[b], sigmaF_avg), etas[b]) for b in eachindex(batch)]
                new_probs = [
                    pdf(p(PPO_model(batch[b][1]), sigmaF_avg), etas[b]) for
                    b in eachindex(batch)
                ]
                ratio_unclipped = [new_probs[b] / old_probs[b] for b in eachindex(batch)]
                ratio_clipped = clamp.(ratio_unclipped, 1 - clip, 1 + clip)
                return -mean(
                    min.(ratio_unclipped .* advantages, ratio_clipped .* advantages)
                )
            end
            Flux.update!(opt, Flux.params(PPO_model), grads)
        end
        # Update sigmaF and learning rate
        sigmaF = max(sigmaF - sigmaF_step, sigmaF_values[2])
        lr = opt.os[2].eta
        opt.os[2].eta = max(lr - lr_step, lr_values[2])
    end
    # Final tests
    push!(
        train_rewards,
        mean([
            evaluate_solution_1_rj_sumCj(
                data_train[j][3],
                embedding_to_sequence(ranking(best_model(data_train[j][1]))),
            ) for j in eachindex(data_train)
        ]),
    )
    push!(
        val_rewards,
        mean([
            evaluate_solution_1_rj_sumCj(
                data_val[j][3], embedding_to_sequence(ranking(best_model(data_val[j][1])))
            ) for j in eachindex(data_val)
        ]),
    )
    @info "final train:",
    train_rewards[end], "final val:", val_rewards[end], "best_episode:",
    best_episode
    return best_model, train_rewards, val_rewards, losses
end

# Train PPO
PPO_model, PPO_train, PPO_val, PPO_losses = PPO_SMSP(;
    grad_steps=2000,
    batch_size=20,
    clip=0.2,
    sigmaF_values=[0.01, 0.01],
    lr_values=[5e-4, 5e-4],
);

# Test the trained model
PPO_final_train_rew = [
    evaluate_solution_1_rj_sumCj(
        training_instances[j], embedding_to_sequence(ranking(PPO_model(training_states[j])))
    ) for j in eachindex(training_states)
];
PPO_final_test_rew = [
    evaluate_solution_1_rj_sumCj(
        testing_instances[j], embedding_to_sequence(ranking(PPO_model(testing_states[j])))
    ) for j in eachindex(testing_states)
];
PPO_final_train_mean = mean(PPO_final_train_rew)
PPO_final_test_mean = mean(PPO_final_test_rew)

# Plot the train and validation rewards
smsp_PPO_rew_line = plot(PPO_train; label="train history", title="SMSP PPO", marker=:o)
plot!(smsp_PPO_rew_line, PPO_val; label="val history", marker=:o)
savefig(smsp_PPO_rew_line, joinpath(plotdir, "smsp_PPO_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "smsp_PPO_training_results.jld2");
    model=PPO_model,
    train_rew=PPO_train,
    val_rew=PPO_val,
    train_final=PPO_final_train_rew,
    test_final=PPO_final_test_rew,
)
