using InferOpt: GeneralizedMaximizer, PerturbedAdditive, FenchelYoungLoss

include("utils/utils.jl")
include("utils/policy.jl")

# Load dataset
model_builder = grb_model # highs_model if you do not have gurobi

X, Y = load_VSP_dataset(train_instances; model_builder);
X, Y = X[1:(8 * nb_train_instances)], Y[1:(8 * nb_train_instances)];

Random.seed!(0)
SIL_model = Chain(Dense(14 => 1; bias=false), vec)

# Define functions for Fenchel-Young loss
function optimization(θ; instance)
    routes = prize_collecting_vsp(θ; instance=instance, model_builder)
    return VSPSolution(routes; max_index=nb_locations(instance.instance)).edge_matrix
end

function g(y; instance, kwargs...)
    return vec(sum(y[:, instance.is_postponable]; dims=1))
end

function h(y, duration)
    value = 0.0
    N = size(duration, 1)
    for i in 1:N
        for j in 1:N
            value -= y[i, j] * duration[i, j]
        end
    end
    return value
end

function h(y; instance, kwargs...)
    return h(y, instance.instance.duration)
end

# SIL training function
function SIL_training!(SIL_model; nb_epochs=400)
    # Initialize training
    gm = GeneralizedMaximizer(optimization, g, h)
    perturbed_layer = PerturbedAdditive(gm; ε=1e-2, nb_samples=20, is_parallel=true)
    fyl = FenchelYoungLoss(perturbed_layer)
    opt = Adam()
    train_reward_history = []
    val_reward_history = []
    losses = Float32[] # Can be used to store actor or critic losses
    best_model = deepcopy(SIL_model)
    best_performance = -Inf
    best_epoch = 0
    for epoch in 1:nb_epochs
        # Test model
        policy = CombinatorialACPolicy(;
            actor_model=SIL_model,
            critic_model=nothing,
            p=nothing,
            CO_layer=prize_collecting_vsp,
            seed=0,
        )
        push!(
            train_reward_history,
            evaluate_policy(
                policy,
                train_envs;
                nb_episodes=1,
                rng=MersenneTwister(0),
                perturb=false,
                model_builder,
            ),
        )
        push!(
            val_reward_history,
            evaluate_policy(
                policy,
                val_envs;
                nb_episodes=1,
                rng=MersenneTwister(0),
                perturb=false,
                model_builder,
            ),
        )
        if val_reward_history[end] >= best_performance
            best_performance = val_reward_history[end]
            best_model = deepcopy(SIL_model)
            best_epoch = epoch
        end
        @info epoch, "train:", train_reward_history[end], "val:", val_reward_history[end]
        # Train model
        l = 0.0
        for ((x, instance), y_true) in zip(X, Y)
            grads = gradient(Flux.params(SIL_model)) do
                l += fyl(SIL_model(x), y_true; instance)
            end
            Flux.update!(opt, Flux.params(SIL_model), grads)
        end
    end
    # Final tests
    policy = CombinatorialACPolicy(;
        actor_model=best_model,
        critic_model=nothing,
        p=nothing,
        CO_layer=prize_collecting_vsp,
        seed=0,
    )
    push!(
        train_reward_history,
        evaluate_policy(
            policy,
            train_envs;
            nb_episodes=1,
            rng=MersenneTwister(0),
            perturb=false,
            model_builder,
        ),
    )
    push!(
        val_reward_history,
        evaluate_policy(
            policy,
            val_envs;
            nb_episodes=1,
            rng=MersenneTwister(0),
            perturb=false,
            model_builder,
        ),
    )
    @info "final train:",
    train_reward_history[end], "final val:", val_reward_history[end], "best_epoch:",
    best_epoch
    return best_model, train_reward_history, val_reward_history, losses
end

# Train SIL
SIL_model, SIL_train, SIL_val, SIL_losses = SIL_training!(SIL_model; nb_epochs=400)

# Test the trained model
SIL_policy = KleopatraVSPPolicy(SIL_model)
SIL_final_train_mean, SIL_final_train_rew =
    .-evaluate_policy(SIL_policy, train_envs; nb_episodes=10, model_builder)
SIL_final_test_mean, SIL_final_test_rew =
    .-evaluate_policy(SIL_policy, test_envs; nb_episodes=10, model_builder)

# Plot training and validation rewards
dvsp_SIL_rew_line = plot(SIL_train; label="train history", title="DVSP SIL", marker=:o)
plot!(dvsp_SIL_rew_line, SIL_val; label="val history", marker=:o)
savefig(dvsp_SIL_rew_line, "plots/dvsp_SIL_rew_line.pdf")

# Save the model and rewards
jldsave(
    joinpath(logdir, "dvsp_SIL_training_results.jld2");
    model=SIL_model,
    train_rew=SIL_train,
    val_rew=SIL_val,
    train_final=SIL_final_train_rew,
    test_final=SIL_final_test_rew,
)
