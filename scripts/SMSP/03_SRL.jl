include("utils/utils.jl")

regularized_predictor = PerturbedAdditive(ranking; ε=1.0, nb_samples=20, is_parallel=true);
loss = FenchelYoungLoss(regularized_predictor);

# SRL training function
function SRL_SMSP(;
    grad_steps=2000,
    batch_size=20,
    no_samples=40,
    sigmaB_values=[2.0, 2.0],
    lr_values=[2e-3, 1e-3],
    temp_values=[1.0, 1.0],
    seed=0,
)
    # Initialize training
    Random.seed!(seed)
    SRL_model = Chain(Dense(nb_features, 1; bias=false), X -> dropdims(X; dims=1))
    opt = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))

    sigmaB = sigmaB_values[1]
    sigmaB_step = (sigmaB_values[1] - sigmaB_values[2]) / grad_steps
    lr_step = (lr_values[1] - lr_values[2]) / grad_steps
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / grad_steps

    train_rewards = Float64[]
    val_rewards = Float64[]
    losses = [] # Can be used to store actor or critic losses
    best_model = deepcopy(SRL_model)
    best_episode = 0

    for e in 1:grad_steps
        # Test model
        push!(
            train_rewards,
            mean([
                evaluate_solution_1_rj_sumCj(
                    data_train[j][3],
                    embedding_to_sequence(ranking(SRL_model(data_train[j][1]))),
                ) for j in eachindex(data_train)
            ]),
        )
        push!(
            val_rewards,
            mean([
                evaluate_solution_1_rj_sumCj(
                    data_val[j][3],
                    embedding_to_sequence(ranking(SRL_model(data_val[j][1]))),
                ) for j in eachindex(data_val)
            ]),
        )
        @info e,
        "sigmaB:",
        sigmaB,
        "lr:",
        opt.os[2].eta,
        "temp:",
        temp,
        "train:",
        train_rewards[end],
        "val:",
        val_rewards[end]
        if val_rewards[end] == minimum(val_rewards)
            best_model = deepcopy(SRL_model)
            best_episode = e
        end
        batches = Flux.DataLoader(data_train; batchsize=batch_size, shuffle=true)
        # Train model
        for batch in batches
            best_sequences = []
            # Perturb and sample candidate actions
            for b in batch
                θ = SRL_model(b[1])
                η = rand(rng, p(θ, sigmaB), no_samples - 1)
                sequences = Any[ranking(θ)]
                values = Any[evaluate_solution_1_rj_sumCj(
                    b[3], embedding_to_sequence(sequences[end])
                )]
                for i in 1:(no_samples - 1)
                    push!(sequences, ranking(η[:, i]))
                    push!(
                        values,
                        evaluate_solution_1_rj_sumCj(
                            b[3], embedding_to_sequence(sequences[end])
                        ),
                    )
                end
                # Compute target action
                values = values ./ (-temp)
                lse = logsumexp(values)
                probs = exp.(values .- lse)
                avg_action = sum(probs .* sequences)
                any(isnan.(avg_action)) ? avg_action = sequences[argmax(values)] : nothing
                push!(best_sequences, avg_action)
            end
            # Update actor using Fenchel-Young loss
            fyl_l = 0.0
            grads = gradient(Flux.params(SRL_model)) do
                for b in eachindex(batch)
                    fyl_l += loss(SRL_model(batch[b][1]), best_sequences[b])
                end
                return fyl_l
            end
            Flux.update!(opt, Flux.params(SRL_model), grads)
        end
        # Update sigmaB, learning rate, and temperature
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[2])
        lr = opt.os[2].eta
        opt.os[2].eta = max(lr - lr_step, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
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

# Train SRL
SRL_model, SRL_train, SRL_val, SRL_losses = SRL_SMSP(;
    grad_steps=2000,
    batch_size=20,
    no_samples=40,
    sigmaB_values=[2.0, 2.0],
    lr_values=[2e-3, 1e-3],
    temp_values=[1.0, 1.0],
);

# Test the trained model
SRL_final_train_rew = [
    evaluate_solution_1_rj_sumCj(
        training_instances[j], embedding_to_sequence(ranking(SRL_model(training_states[j])))
    ) for j in eachindex(training_states)
];
SRL_final_test_rew = [
    evaluate_solution_1_rj_sumCj(
        testing_instances[j], embedding_to_sequence(ranking(SRL_model(testing_states[j])))
    ) for j in eachindex(testing_states)
];
SRL_final_train_mean = mean(SRL_final_train_rew)
SRL_final_test_mean = mean(SRL_final_test_rew)

# Plot the train and validation rewards
smsp_SRL_rew_line = plot(SRL_train; label="train history", title="SMSP SRL", marker=:o)
plot!(smsp_SRL_rew_line, SRL_val; label="val history", marker=:o)
savefig(smsp_SRL_rew_line, joinpath(plotdir, "smsp_SRL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "smsp_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    val_rew=SRL_val,
    train_final=SRL_final_train_rew,
    test_final=SRL_final_test_rew,
)
