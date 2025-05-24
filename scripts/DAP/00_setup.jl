using Flux
using JLD2
using Random

include("utils/utils.jl")

Random.seed!(14)

env = DAP(20, 2, 4, 200)

# Initial models
initial_model = Chain(Dense(10 => 5; bias=true), Dense(5 => 1; bias=true), vec)

val_test(env, initial_model, customer_model, 10; first_seed=1)

jldsave(joinpath(logdir, "DAP_initial_model.jld2"); actor=initial_model)

critic_rew = Chain(Dense(6 => 3; bias=true), vec, Dense(12 => 1; bias=true), vec)
jldsave(joinpath(logdir, "DAP_critic_rew.jld2"); critic=critic_rew)

critic_fut = Chain(
    Dense(11 => 5; bias=true),
    vec,
    Dense(100 => 10; bias=true),
    Dense(10 => 1; bias=true),
    vec,
)
jldsave(joinpath(logdir, "DAP_critic_fut.jld2"); critic=critic_fut)

# Baseline models
greedy_train, greedy_train_rew = val_test(env, model_greedy, customer_model, 100; first_seed=1)
expert_rev, expert_train_rew = expert_policy(env, customer_model, 100; first_seed=1)

greedy_test, greedy_test_rew = val_test(env, model_greedy, customer_model, 100; first_seed=2001)
expert_test, expert_test_rew = expert_policy(env, customer_model, 100; first_seed=2001)

jldsave(
    joinpath(logdir, "dap_baselines.jld2");
    greedy_train=greedy_train_rew,
    expert_train=expert_train_rew,
    greedy_test=greedy_test_rew,
    expert_test=expert_test_rew,
)
