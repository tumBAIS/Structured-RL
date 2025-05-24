include("utils/utils.jl");

train_dataset, val_dataset, test_dataset = load("logs/wspp_dataset.jld2")["data"];

Random.seed!(0);
PPO_model = new_warcraft_embedding();

function PPO_training(
    model,
    data,
    val_data;
    nb_epochs=200,
    batch_size=20,
    clip=0.2,
    sigmaF_values=[0.1, 0.05],
    lr_values=[5e-4, 1e-4],
)
    # Initialize training
    opt = Optimiser(ClipValue(1e-3), Adam(lr_values[1]))
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs

    train_costs = Float64[]
    val_costs = Float64[]
    losses = Float64[] # Can be used to store losses
    best_model = deepcopy(model)
    best_episode = 0

    prob(θ, stdev) = MvNormal(θ, stdev * I)
    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / nb_epochs
    sigmaF_avg = ((sigmaF_values[1] + sigmaF_values[2]) / 2) * 1

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
        "sigmaF:", sigmaF, "lr", opt.os[2].eta, "train:", train_costs[end], "val:",
        val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = epoch
        end

        batches = Flux.DataLoader(data; batchsize=batch_size, shuffle=true)

        # Train model
        for batch in batches
            # Calculate advantages
            thetas = []
            etas = []
            advantages = []
            for b in batch
                push!(thetas, model(b[1]))
                push!(
                    etas,
                    -abs.(
                        reshape(
                            rand(prob(reshape(thetas[end], length(thetas[end])), sigmaF)),
                            size(thetas[end]),
                        )
                    ),
                )
                push!(
                    advantages,
                    cost(true_maximizer(thetas[end]); c_true=b[3].wg.vertex_weights) -
                    cost(true_maximizer(etas[end]); c_true=b[3].wg.vertex_weights),
                )
            end

            # Update model
            grads = gradient(Flux.params(model)) do
                old_probs = [
                    pdf(
                        prob(reshape(thetas[b], length(thetas[b])), sigmaF_avg),
                        reshape(etas[b], length(etas[b])),
                    ) for b in 1:batch_size
                ]
                new_probs = [
                    pdf(
                        prob(reshape(model(batch[b][1]), length(thetas[b])), sigmaF_avg),
                        reshape(etas[b], length(etas[b])),
                    ) for b in 1:batch_size
                ]
                ratio_unclipped = [new_probs[b] / old_probs[b] for b in 1:batch_size]
                ratio_clipped = clamp.(ratio_unclipped, 1 - clip, 1 + clip)
                actor_loss =
                    -mean(min.(ratio_unclipped .* advantages, ratio_clipped .* advantages))
            end
            Flux.update!(opt, Flux.params(model), grads)
        end
        # Update sigmaF and learning rate
        sigmaF = max(sigmaF - sigmaF_step, sigmaF_values[2])
        lr = opt.os[2].eta
        opt.os[2].eta = max(lr - lr_step, lr_values[2])
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

# Train PPO
PPO_model, PPO_train, PPO_val, PPO_losses = PPO_training(
    PPO_model,
    train_dataset,
    val_dataset;
    nb_epochs=200,
    batch_size=20,
    clip=0.2,
    sigmaF_values=[0.1, 0.05],
    lr_values=[5e-4, 1e-4],
)

# Test the trained model
PPO_final_train_rew = [
    cost(true_maximizer(PPO_model(x)); c_true=kwargs.wg.vertex_weights) for
    (x, y, kwargs) in train_dataset
];
PPO_final_test_rew = [
    cost(true_maximizer(PPO_model(x)); c_true=kwargs.wg.vertex_weights) for
    (x, y, kwargs) in test_dataset
];
PPO_final_train_mean = mean(PPO_final_train_rew)
PPO_final_test_mean = mean(PPO_final_test_rew)

# Plot the train and validation rewards
wspp_PPO_rew_line = plot(PPO_train; label="train history", title="WSPP PPO", marker=:o)
plot!(wspp_PPO_rew_line, PPO_val; label="val history", marker=:o)
savefig(wspp_PPO_rew_line, joinpath(plotdir, "wspp_PPO_rew_line.pdf"))

# Save the model and rewards
jldsave(
    joinpath(logdir, "wspp_PPO_training_results.jld2");
    model=PPO_model,
    train_rew=PPO_train,
    val_rew=PPO_val,
    train_final=PPO_final_train_rew,
    test_final=PPO_final_test_rew,
)
