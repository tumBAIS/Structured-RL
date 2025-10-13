include("utils/utils.jl")

"""
Seeds used in the paper:

[0, 7, 8, 33, 34, 39, 58, 62, 63, 98]

Initial rewards should be sufficiently low to avoid "chance" convergence and to better see algorithmic differences.
"""

# Env initialization
env = GridWorld();

# Baseline models
expert_train, expert_train_rew = expert_solution(
    env, 100; first_seed=1, create_dataset=false
)
expert_test, expert_test_rew = expert_solution(
    env, 100; first_seed=2001, create_dataset=false
    )

greedy_train, greedy_train_rew = greedy_heuristic(env, 100; first_seed=1)
greedy_test, greedy_test_rew = greedy_heuristic(env, 100; first_seed=2001)

jldsave(
    joinpath(logdir, "gspp_baselines.jld2");
    greedy_train=greedy_train_rew,
    expert_train=expert_train_rew,
    greedy_test=greedy_test_rew,
    expert_test=expert_test_rew,
)
