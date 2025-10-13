include("utils/utils.jl")

"""
Seeds used in the paper:

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

## Dataset creation

model_builder = grb_model # mip solver

b = StochasticVehicleSchedulingBenchmark(; nb_tasks=25, nb_scenarios=10)

dataset = generate_dataset(
    b, 150; seed=0, algorithm=compact_mip, model_builder, silent=false
); # model_builder can be omitted if Gurobi is not available

# Feature normalization
train_set, _, __ = splitobs(dataset; at=(50, 50));
dt = StatsBase.fit(StatsBase.ZScoreTransform, train_set; center=false, scale=true);
StatsBase.transform!(dt, dataset)

jldsave(dataset_path; dataset, dt)

train_set, val_set, test_set = splitobs(dataset; at=(50, 50));

# COaML-pipeline
model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)

## Baseline solutions

# Expert solution
expert_train_rew = [evaluate_solution(i.y_true, i.instance) for i in train_set]
expert_test_rew = [evaluate_solution(i.y_true, i.instance) for i in test_set]
expert_train = mean(expert_train_rew)
expert_test = mean(expert_test_rew)

# Greedy solution
greedy_train_rew = [
    evaluate_solution(deterministic_mip(i.instance; model_builder=grb_model), i.instance)
    for i in train_set
]
greedy_test_rew = [
    evaluate_solution(deterministic_mip(i.instance; model_builder=grb_model), i.instance)
    for i in test_set
]
greedy_train = mean(greedy_train_rew)
greedy_test = mean(greedy_test_rew)

JLD2.jldsave(
    joinpath(logdir, "svsp_baselines.jld2");
    greedy_train=greedy_train_rew,
    expert_train=expert_train_rew,
    greedy_test=greedy_test_rew,
    expert_test=expert_test_rew,
)
