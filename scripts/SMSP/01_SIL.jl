include("utils/utils.jl")

# SIL training function
function SIL_SMSP(data_train; epochs=2000, seed=0)
    # Initialize training
    Random.seed!(seed)
    model = Chain(Dense(nb_features, 1; bias=false), X -> dropdims(X; dims=1))
    regularized_predictor = PerturbedAdditive(
        ranking; ε=1.0, nb_samples=20, is_parallel=true
    )
    loss = FenchelYoungLoss(regularized_predictor)
    losses = Float64[] # Can be used to store losses
    opt = ADAM()
    SIL_train_rewards = Float64[]
    SIL_val_rewards = Float64[]
    best_model = deepcopy(model)
    best_epoch = 0
    for epoch in 1:epochs
        # Test model
        push!(
            SIL_train_rewards,
            mean([
                evaluate_solution_1_rj_sumCj(
                    data_train[j][3],
                    embedding_to_sequence(ranking(model(data_train[j][1]))),
                ) for j in eachindex(data_train)
            ]),
        )
        push!(
            SIL_val_rewards,
            mean([
                evaluate_solution_1_rj_sumCj(
                    data_val[j][3], embedding_to_sequence(ranking(model(data_val[j][1])))
                ) for j in eachindex(data_val)
            ]),
        )
        @info epoch, "train:", SIL_train_rewards[end], "val:", SIL_val_rewards[end]
        if SIL_val_rewards[end] == minimum(SIL_val_rewards)
            best_model = deepcopy(model)
            best_epoch = epoch
        end
        # Train model
        fyl_l = 0.0
        loss_l = 0.0
        for (x, y, inst, val) in data_train
            grads = gradient(Flux.params(model)) do
                fyl_l += loss(model(x), y)
            end
            loss_l +=
                (
                    evaluate_solution_1_rj_sumCj(
                        inst, embedding_to_sequence(ranking(model(x)))
                    ) - val
                ) / val
            Flux.update!(opt, Flux.params(model), grads)
        end
    end
    # Final tests
    push!(
        SIL_train_rewards,
        mean([
            evaluate_solution_1_rj_sumCj(
                data_train[j][3],
                embedding_to_sequence(ranking(best_model(data_train[j][1]))),
            ) for j in eachindex(data_train)
        ]),
    )
    push!(
        SIL_val_rewards,
        mean([
            evaluate_solution_1_rj_sumCj(
                data_val[j][3], embedding_to_sequence(ranking(best_model(data_val[j][1])))
            ) for j in eachindex(data_val)
        ]),
    )
    @info "final train:",
    SIL_train_rewards[end], "final val:", SIL_val_rewards[end], "best_epoch:",
    best_epoch
    return best_model, SIL_train_rewards, SIL_val_rewards, losses
end

# Train SIL
SIL_model, SIL_train, SIL_val, SIL_losses = SIL_SMSP(
    data_train; epochs=2000, seed=0
)

# Test the trained model
SIL_final_train_rew = [
    evaluate_solution_1_rj_sumCj(
        training_instances[j], embedding_to_sequence(ranking(SIL_model(training_states[j])))
    ) for j in eachindex(training_states)
];
SIL_final_test_rew = [
    evaluate_solution_1_rj_sumCj(
        testing_instances[j], embedding_to_sequence(ranking(SIL_model(testing_states[j])))
    ) for j in eachindex(testing_states)
];
SIL_final_train_mean = mean(SIL_final_train_rew)
SIL_final_test_mean = mean(SIL_final_test_rew)

# Plot the train and validation rewards
smsp_SIL_rew_line = plot(SIL_train; label="train history", title="SMSP SIL", marker=:o)
plot!(smsp_SIL_rew_line, SIL_val; label="val history", marker=:o)
savefig(smsp_SIL_rew_line, joinpath(plotdir, "smsp_SIL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "smsp_SIL_training_results.jld2");
    model=SIL_model,
    train_rew=SIL_train,
    val_rew=SIL_val,
    train_final=SIL_final_train_rew,
    test_final=SIL_final_test_rew,
)
