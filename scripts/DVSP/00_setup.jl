include("utils/utils.jl")

"""
Seeds used in the paper:

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

## Baseline policies

# Greedy policy
greedy_policy = GreedyVSPPolicy()
greedy_train_mean, greedy_train_rews = .- evaluate_policy(
    greedy_policy,
    train_envs;
    nb_episodes=1,
    # return_scores=true,
    # rng=MersenneTwister(0),
    model_builder,
)
greedy_test_mean, greedy_test_rews = .- evaluate_policy(
    greedy_policy,
    test_envs;
    nb_episodes=1,
    # return_scores=true,
    # rng=MersenneTwister(0),
    model_builder,
)

# Expert policy
expert_train_mean, expert_train_rews = expert_evaluation(train_envs; model_builder)
expert_test_mean, expert_test_rews = expert_evaluation(test_envs; model_builder)

jldsave(
    joinpath(logdir, "dvsp_baselines.jld2");
    greedy_train=greedy_train_rews,
    expert_train=expert_train_rews,
    greedy_test=greedy_test_rews,
    expert_test=expert_test_rews,
)
