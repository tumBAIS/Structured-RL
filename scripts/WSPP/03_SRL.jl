include("utils/utils.jl");

train_dataset, val_dataset, test_dataset = load("logs/wspp_dataset.jld2")["data"];

Random.seed!(0);
SRL_model = new_warcraft_embedding();

function SRL_training(
    model,
    data,
    val_data;
    nb_epochs=200,
    batch_size=60,
    no_samples=40,
    sigmaB_values=[0.1, 0.05],
    lr_values=[2e-3, 1e-3],
    temp_values=[0.1, 0.01],
)
    # Initialize training
    loss = FenchelYoungLoss(PerturbedMultiplicative(true_maximizer; ε=0.05, nb_samples=20))
    opt = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs

    train_costs = Float64[]
    val_costs = Float64[]
    losses = Float64[] # Can be used to store losses
    best_model = deepcopy(model)
    best_episode = 0

    prob(θ, stdev) = MvNormal(θ, stdev * I)
    sigmaB = sigmaB_values[1]
    sigmaB_step = (sigmaB_values[1] - sigmaB_values[2]) / nb_epochs
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / nb_epochs

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
        @info epoch,
        "sigmaB:",
        sigmaB,
        "lr:",
        opt.os[2].eta,
        "temp:",
        temp,
        "train:",
        train_costs[end],
        "val:",
        val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = epoch
        end

        batches = Flux.DataLoader(data; batchsize=batch_size, shuffle=true)

        # Train model
        for batch in batches
            best_solutions = []
            for b in batch
                # Perturb and sample candidate actions
                θ = model(b[1])
                η =
                    -abs.(
                        reshape(
                            rand(prob(reshape(θ, length(θ)), sigmaB), no_samples),
                            (size(θ)..., no_samples),
                        )
                    )
                solutions = []
                values = []
                push!(solutions, true_maximizer(θ))
                push!(values, cost(solutions[end]; c_true=b[3].wg.vertex_weights))
                for i in 1:no_samples
                    solution = true_maximizer(η[:, :, i])
                    push!(solutions, solution)
                    push!(values, cost(solution; c_true=b[3].wg.vertex_weights))
                end
                # Compute target action
                values = values ./ (-temp)
                lse = logsumexp(values)
                probs = exp.(values .- lse)
                best_action = sum(probs .* solutions)
                any(isnan.(best_action)) ? best_action = solutions[argmax(values)] : nothing
                push!(best_solutions, best_action)
            end

            # Update model using Fenchel-Young loss
            grads = gradient(Flux.params(model)) do
                actor_loss = sum([
                    loss(
                        model(batch[j][1]),
                        best_solutions[j];
                        fw_kwargs=(max_iteration=50,),
                    ) for j in 1:batch_size
                ])
            end
            Flux.update!(opt, Flux.params(model), grads)
        end
        # Update sigmaB, learning rate, and temperature
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[2])
        lr = opt.os[2].eta
        opt.os[2].eta = max(lr - lr_step, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
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
    train_costs[end], "final val:", val_costs[end], "best_episode:",
    best_episode
    return best_model, train_costs, val_costs, losses
end

# Train SRL
SRL_model, SRL_train, SRL_val, SRL_losses = SRL_training(
    SRL_model,
    train_dataset,
    val_dataset;
    nb_epochs=200,
    batch_size=60,
    no_samples=40,
    sigmaB_values=[0.1, 0.05],
    lr_values=[2e-3, 1e-3],
    temp_values=[0.1, 0.01],
)

# Test the trained model
SRL_final_train_rew = [
    cost(true_maximizer(SRL_model(x)); c_true=kwargs.wg.vertex_weights) for
    (x, y, kwargs) in train_dataset
];
SRL_final_test_rew = [
    cost(true_maximizer(SRL_model(x)); c_true=kwargs.wg.vertex_weights) for
    (x, y, kwargs) in test_dataset
];
SRL_final_train_mean = mean(SRL_final_train_rew)
SRL_final_test_mean = mean(SRL_final_test_rew)

# Plot the train and validation rewards
wspp_SRL_rew_line = plot(SRL_train; label="train history", title="WSPP SIL", marker=:o)
plot!(wspp_SRL_rew_line, SRL_val; label="val history", marker=:o)
savefig(wspp_SRL_rew_line, joinpath(plotdir, "wspp_SRL_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "wspp_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    val_rew=SRL_val,
    train_final=SRL_final_train_rew,
    test_final=SRL_final_test_rew,
)
