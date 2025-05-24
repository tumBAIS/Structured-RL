include("utils/utils.jl");

train_dataset, val_dataset, test_dataset = load("logs/wspp_dataset.jld2")["data"];

Random.seed!(0);
SIL_model = new_warcraft_embedding();

# SIL training function
function SIL_training(model, data, val_data; nb_epochs=200, batch_size=60, lr_start=0.001)
    # Initialize training
    loss = FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=0.05, nb_samples=20))
    opt = ADAM(lr_start)
    data_train = Flux.DataLoader(data; batchsize=batch_size)

    train_costs = Float64[]
    val_costs = Float64[]
    losses = Float64[] # Can be used to store losses
    params = Flux.params(model)
    best_model = deepcopy(model)
    best_epoch = 0

    for epoch in 1:nb_epochs
        # Test model
        push!(
            train_costs,
            mean([
                cost(true_maximizer(model(b[1])); c_true=b[3].wg.vertex_weights) for
                b in data
            ]),
        )
        push!(
            val_costs,
            mean([
                cost(true_maximizer(model(b[1])); c_true=b[3].wg.vertex_weights) for
                b in val_data
            ]),
        )
        @info epoch, "train:", train_costs[end], "val:", val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_epoch = epoch
        end
        # Train model
        for batch in data_train
            batch_loss = 0
            gs = gradient(params) do
                batch_loss = sum([
                    loss(model(batch[j][1]), batch[j][2]; fw_kwargs=(max_iteration=50,)) for j in eachindex(batch)
                ])
            end
            Flux.update!(opt, params, gs)
        end
    end
    # Final tests
    push!(
        train_costs,
        mean([
            cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.vertex_weights) for
            b in data
        ]),
    )
    push!(
        val_costs,
        mean([
            cost(true_maximizer(best_model(b[1])); c_true=b[3].wg.vertex_weights) for
            b in val_data
        ]),
    )
    @info "final train:",
    train_costs[end], "final val:", val_costs[end], "best_epoch:",
    best_epoch
    return best_model, train_costs, val_costs, losses
end

# Train SIL
SIL_model, SIL_train, SIL_val, SIL_losses = SIL_training(
    SIL_model, train_dataset, val_dataset; nb_epochs=200, batch_size=60, lr_start=0.001
)

# Test the trained model
SIL_final_train_rew = [
    cost(true_maximizer(SIL_model(x)); c_true=kwargs.wg.vertex_weights) for
    (x, y, kwargs) in train_dataset
];
SIL_final_test_rew = [
    cost(true_maximizer(SIL_model(x)); c_true=kwargs.wg.vertex_weights) for
    (x, y, kwargs) in test_dataset
];
SIL_final_train_mean = mean(SIL_final_train_rew)
SIL_final_test_mean = mean(SIL_final_test_rew)

# Plot the train and validation rewards
wspp_SIL_rew_line = plot(SIL_train; label="train history", title="WSPP SIL", marker=:o)
plot!(wspp_SIL_rew_line, SIL_val; label="val history", marker=:o)
savefig(wspp_SIL_rew_line, joinpath(plotdir, "wspp_SIL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "wspp_SIL_training_results.jld2");
    model=SIL_model,
    train_rew=SIL_train,
    val_rew=SIL_val,
    train_final=SIL_final_train_rew,
    test_final=SIL_final_test_rew,
)
