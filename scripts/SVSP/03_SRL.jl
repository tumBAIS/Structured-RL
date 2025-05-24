using Flux
using Flux.Optimise
using Gurobi: Gurobi
using Distributions
using LinearAlgebra
using InferOpt

include("utils/utils.jl")

# SRL training function
function SRL_training(
    model,
    train_set,
    val_set;
    nb_epochs=200,
    batch_size=4,
    no_samples=20,
    sigmaB_values=[0.1, 0.01],
    lr_values=[1e-2, 5e-3],
    temp_values=[10000.0, 100.0],
)
    # Initialize training
    loss = FenchelYoungLoss(PerturbedAdditive(maximizer; ε=1.0, nb_samples=20))
    opt = Flux.Optimise.Adam(lr_values[1])
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs
    train_costs = Float64[]
    val_costs = Float64[]
    losses = Float64[] # Can be used to store losses
    best_model = deepcopy(model)
    best_episode = 0
    opt_state = Flux.setup(opt, model)

    prob(θ, eps) = MvNormal(θ, eps * I)
    sigmaB = sigmaB_values[1]
    sigmaB_step = (sigmaB_values[1] - sigmaB_values[2]) / nb_epochs
    temp = temp_values[1]
    temp_step = (temp_values[1] - temp_values[2]) / nb_epochs

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
        @info e, "sigmaB:", sigmaB,
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
            best_solutions = []
            for b in batch
                # Perturb and sample candidate actions
                θ = model(b.x)
                η = rand(prob(θ, sigmaB), no_samples)
                solutions = [maximizer(θ; instance=b.instance)]
                values = [evaluate_solution(solutions[end], b.instance)]
                for i in 1:no_samples
                    push!(solutions, maximizer(η[:, i]; instance=b.instance))
                    push!(values, evaluate_solution(solutions[end], b.instance))
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
            val, grads = Flux.withgradient(model) do m
                sum([loss(m(b.x), b.y_true; instance=b.instance) for b in batch])
            end
            Flux.update!(opt_state, model, grads[1])
        end
        # Update sigmaB, learning rate, and temperature
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[2])
        lr = opt.eta
        opt.eta = max(lr - lr_step, lr_values[2])
        temp = max(temp - temp_step, temp_values[2])
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

# Train SRL
SRL_model, SRL_train, SRL_val, SRL_losses = SRL_training(
    deepcopy(model),
    train_set,
    val_set;
    nb_epochs=200,
    batch_size=4,
    no_samples=20,
    sigmaB_values=[0.1, 0.01],
    lr_values=[1e-2, 5e-3],
    temp_values=[10000.0, 100.0],
)

# Test the trained model
SRL_final_train_rew = [
    evaluate_solution(maximizer(SRL_model(i.x); instance=i.instance), i.instance) for
    i in train_set
]
SRL_final_test_rew = [
    evaluate_solution(maximizer(SRL_model(i.x); instance=i.instance), i.instance) for
    i in test_set
]
SRL_final_train_mean = mean(SRL_final_train_rew)
SRL_final_test_mean = mean(SRL_final_test_rew)

# Plot the train and validation rewards
svsp_SRL_rew_line = plot(SRL_train; label="train history", title="SVSP SRL", marker=:o)
plot!(svsp_SRL_rew_line, SRL_val; label="val history", marker=:o)
savefig(svsp_SRL_rew_line, joinpath(plotdir, "svsp_SRL_rew_line.pdf"))

# Save the model and results
JLD2.jldsave(
    joinpath(logdir, "svsp_SRL_training_results.jld2");
    model=SRL_model,
    train_rew=SRL_train,
    val_rew=SRL_val,
    train_final=SRL_final_train_rew,
    test_final=SRL_final_test_rew,
)
