using Flux: Flux, Adam, Descent
using Gurobi: Gurobi
using InferOpt
using ProgressMeter: @showprogress
using Statistics: mean
using Plots

include("utils/utils.jl")

## Model setup

dataset = load(dataset_path)["dataset"]
train_set, val_set, test_set = splitobs(dataset; at=(50, 50));

# Define losses
perturbed = PerturbedAdditive(maximizer; ε=1.0, nb_samples=20)
fyl_loss = FenchelYoungLoss(perturbed)
SIL_loss(sample, m) = fyl_loss(m(sample.x), sample.y_true; instance=sample.instance)

# SIL training function
function SIL_training!(
    model, maximizer, train_set, val_set, loss; nb_epochs=200, optimizer=Adam()
)
    # Initialize training
    train_costs = []
    val_costs = []
    loss_history = [] # Can be used to store losses
    best_model = deepcopy(model)
    best_epoch = 0

    opt_state = Flux.setup(optimizer, model)
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
        @info e, "train:", train_costs[end], "val:", val_costs[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_epoch = e
        end
        # Train model
        loss_sum = 0.0
        for sample in train_set
            val, grads = Flux.withgradient(model) do m
                loss(sample, m)
            end
            Flux.update!(opt_state, model, grads[1])
            loss_sum += val
        end
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
    @info "final train:",
    train_costs[end], "final val:", val_costs[end], "best_epoch:",
    best_epoch
    return best_model, train_costs, val_costs, loss_history
end

# Train SIL
SIL_model, SIL_train, SIL_val, SIL_losses = SIL_training!(
    deepcopy(model), maximizer, train_set, val_set, SIL_loss; nb_epochs=200
)

# Test the trained model
SIL_final_train_rew = [
    evaluate_solution(maximizer(SIL_model(i.x); instance=i.instance), i.instance) for
    i in train_set
]
SIL_final_test_rew = [
    evaluate_solution(maximizer(SIL_model(i.x); instance=i.instance), i.instance) for
    i in test_set
]
SIL_final_train_mean = mean(SIL_final_train_rew)
SIL_final_test_mean = mean(SIL_final_test_rew)

# Plot the train and validation rewards
svsp_SIL_rew_line = plot(SIL_train; label="train history", title="SVSP SIL", marker=:o)
plot!(svsp_SIL_rew_line, SIL_val; label="val history", marker=:o)
savefig(svsp_SIL_rew_line, joinpath(plotdir, "svsp_SIL_rew_line.pdf"))

# Save the model and results
JLD2.jldsave(
    joinpath(logdir, "svsp_SIL_training_results.jld2");
    model=SIL_model,
    train_rew=SIL_train,
    val_rew=SIL_val,
    train_final=SIL_final_train_rew,
    test_final=SIL_final_test_rew,
)
