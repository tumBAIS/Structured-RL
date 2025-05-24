using Flux
using Flux.Optimise
using Gurobi: Gurobi
using Distributions
using LinearAlgebra
using InferOpt

include("utils/utils.jl")

# PPO training function
function PPO_training(
    model,
    train_set,
    val_set;
    nb_epochs=200,
    batch_size=4,
    clip=0.2,
    sigmaF_values=[0.5, 0.1],
    lr_values=[1e-2, 1e-2],
)
    # Initialize training
    opt = Flux.Optimise.Adam(lr_values[1])
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs
    train_costs = Float64[]
    val_costs = Float64[]
    losses = Float64[] # Can be used to store losses
    best_model = deepcopy(model)
    best_episode = 0
    opt_state = Flux.setup(opt, model)

    prob(θ, eps) = MvNormal(θ, eps * I)
    sigmaF = sigmaF_values[1]
    sigmaF_step = (sigmaF_values[1] - sigmaF_values[2]) / nb_epochs

    for e in 1:nb_epochs
        # Test model
        push!(
            train_costs,
            mean([
                evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance)
                for i in train_set
            ]),
        )
        push!(
            val_costs,
            mean([
                evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance)
                for i in val_set
            ]),
        )
        @info e, "sigmaF:", sigmaF,
        "lr", lr_values[1],
        "train:", train_costs[end],
        "val:", val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = e
        end

        batches = Flux.DataLoader(train_set; batchsize=batch_size, shuffle=true)

        # Train model
        for batch in batches
            # Calculate advantages
            thetas = [model(b.x) for b in batch]
            etas = [rand(prob(thetas[j], sigmaF)) for j in eachindex(batch)]
            advantages = [
                evaluate_solution(
                    maximizer(thetas[j]; instance=batch[j].instance), batch[j].instance
                ) - evaluate_solution(
                    maximizer(etas[j]; instance=batch[j].instance), batch[j].instance
                ) for j in eachindex(batch)
            ]

            # Update model
            val, grads = Flux.withgradient(model) do m
                old_probs = [pdf(prob(thetas[j], sigmaF), etas[j]) for j in eachindex(batch)]
                new_probs = [
                    pdf(prob(m(batch[j].x), sigmaF), etas[j]) for j in eachindex(batch)
                ]
                ratio_unclipped = [new_probs[j] / old_probs[j] for j in eachindex(batch)]
                ratio_clipped = clamp.(ratio_unclipped, 1 - clip, 1 + clip)
                return -mean(
                    min.(ratio_unclipped .* advantages, ratio_clipped .* advantages)
                )
            end
            Flux.update!(opt_state, model, grads[1])
        end
        # Update sigmaF and learning rate
        sigmaF = max(sigmaF - sigmaF_step, sigmaF_values[2])
        lr = opt.eta
        opt.eta = max(lr - lr_step, lr_values[2])
    end

    # Final tests
    push!(
        train_costs,
        mean([
            evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance)
            for i in train_set
        ]),
    )
    push!(
        val_costs,
        mean([
            evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance)
            for i in val_set
        ]),
    )
    @info "final train:", train_costs[end],
    "final val:", val_costs[end],
    "best_episode:", best_episode
    return best_model, train_costs, val_costs, losses
end

# Train PPO
PPO_model, PPO_train, PPO_val, PPO_losses = PPO_training(
    deepcopy(model),
    train_set,
    val_set;
    nb_epochs=200,
    batch_size=4,
    clip=0.2,
    sigmaF_values=[0.5, 0.1],
    lr_values=[1e-2, 1e-2],
)

# Test the trained model
PPO_final_train_rew = [
    evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for
    i in train_set
]
PPO_final_test_rew = [
    evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for
    i in test_set
]
PPO_final_train_mean = mean(PPO_final_train_rew)
PPO_final_test_mean = mean(PPO_final_test_rew)

# Plot the train and validation rewards
svsp_PPO_rew_line = plot(PPO_train; label="train history", title="SVSP PPO", marker=:o)
plot!(svsp_PPO_rew_line, PPO_val; label="val history", marker=:o)
savefig(svsp_PPO_rew_line, joinpath(plotdir, "svsp_PPO_rew_line.pdf"))

# Save the model and results
JLD2.jldsave(
    joinpath(logdir, "svsp_PPO_training_results.jld2");
    model=PPO_model,
    train_rew=PPO_train,
    val_rew=PPO_val,
    train_final=PPO_final_train_rew,
    test_final=PPO_final_test_rew,
)
