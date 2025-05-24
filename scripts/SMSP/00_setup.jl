include("utils/utils.jl")

Random.seed!(0)

# Expert policy
expert_train_rews = getfield.(train_test, 4)
expert_test_rews = getfield.(data_test, 4)
expert_train = mean(expert_train_rews)
expert_test = mean(expert_test_rews)

# Greedy policy
greedy_train_rews = [
    evaluate_solution_1_rj_sumCj(instance, greedy_heuristic(instance)) for
    instance in training_instances
]
greedy_test_rews = [
    evaluate_solution_1_rj_sumCj(instance, greedy_heuristic(instance)) for
    instance in testing_instances
]
greedy_train = mean(greedy_train_rews)
greedy_test = mean(greedy_test_rews)

jldsave(
    joinpath(logdir, "smsp_baselines.jld2");
    greedy_train=greedy_train_rews,
    expert_train=expert_train_rews,
    greedy_test=greedy_test_rews,
    expert_test=expert_test_rews,
)