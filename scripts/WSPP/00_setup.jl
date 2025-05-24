include("utils/utils.jl");

# Import dataset
Random.seed!(63);
data_path = joinpath(datadep"warcraft", "data")
options = (nb_epochs=100, batch_size=80, lr_start=0.001);

dataset = create_dataset(data_path, 200);
train_dataset, val_dataset, test_dataset = train_test_split(
    dataset, 0.6; val_percentage=0.2, use_val=true
);

jldsave("logs/wspp_dataset.jld2"; data=(; train_dataset, val_dataset, test_dataset))

# Expert solution
expert_train_rew = [
    cost(y; c_true=kwargs.wg.vertex_weights) for (x, y, kwargs) in train_dataset
]
expert_test_rew = [
    cost(y; c_true=kwargs.wg.vertex_weights) for (x, y, kwargs) in test_dataset
]
expert_train = mean(expert_train_rew)
expert_test = mean(expert_test_rew)

# Greedy solution
greedy_path = I(12)
greedy_train_rew = [
    cost(greedy_path; c_true=kwargs.wg.vertex_weights) for (x, y, kwargs) in train_dataset
]
greedy_train = mean(greedy_train_rew)
greedy_test_rew = [
    cost(greedy_path; c_true=kwargs.wg.vertex_weights) for (x, y, kwargs) in test_dataset
]
greedy_test = mean(greedy_test_rew)

jldsave(
    joinpath(logdir, "wspp_baselines.jld2");
    greedy_train=greedy_train_rew,
    expert_train=expert_train_rew,
    greedy_test=greedy_test_rew,
    expert_test=expert_test_rew,
)
