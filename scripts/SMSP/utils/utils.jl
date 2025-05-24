using SingleMachineScheduling
using Flux
using Flux.Optimise
using InferOpt
using Gurobi
using Random
using LinearAlgebra: I
using Statistics
using Distributions
using JLD2
using Plots

include("../../utils.jl")

## Instance creation

# Define the instance parameters
encoder = encoder_1_rj_sumCj;
nb_features = nb_features_encoder(encoder);

nb_jobs = 50:10:100;
range_values = 0.2:0.2:1.4;
seeds_train = 1:10;
seeds_val = 21:25;
seeds_test = 41:45;
seeds_train_for_tests = 1:5;

# Initialize the instance solver
env = Gurobi.Env()
gurobi_solver = () -> Gurobi.Optimizer(env)

function gurobi_1_rj_sumCj(inst::Instance1_rj_sumCj)
    return milp_solve_1_rj_sumCj(inst; MILP_solver=gurobi_solver)
end
SingleMachineScheduling.solver_name(sol::typeof(gurobi_1_rj_sumCj)) = "gurobi";

# Create the instances
data_train = [
    build_solve_and_encode_instance(;
        seed=s, nb_jobs=n, range=r, solver=gurobi_1_rj_sumCj, load_and_save=true
    ) for s in seeds_train for n in nb_jobs for r in range_values
]; # 420 instances
data_val = [
    build_solve_and_encode_instance(;
        seed=s, nb_jobs=n, range=r, solver=gurobi_1_rj_sumCj, load_and_save=true
    ) for s in seeds_val for n in nb_jobs for r in range_values
]; # 210 instances
data_test = [
    build_solve_and_encode_instance(;
        seed=s, nb_jobs=n, range=r, solver=gurobi_1_rj_sumCj, load_and_save=true
    ) for s in seeds_test for n in nb_jobs for r in range_values
]; # 210 instances

train_test = [
    build_solve_and_encode_instance(;
        seed=s, nb_jobs=n, range=r, solver=gurobi_1_rj_sumCj, load_and_save=true
    ) for s in seeds_train_for_tests for n in nb_jobs for r in range_values
];
# Used for final tests
training_states = getfield.(train_test, 1);
training_instances = getfield.(train_test, 3);
testing_states = getfield.(data_test, 1);
testing_instances = getfield.(data_test, 3);

## Greedy heuristic

function greedy_heuristic(instance)
    (; nb_jobs, processing_times, release_times) = instance
    return sort(
        1:nb_jobs;
        lt=(i, j) ->
            (release_times[i] < release_times[j]) || (
                (release_times[i] == release_times[j]) &&
                processing_times[i] < processing_times[j]
            ),
    )
end

p(θ, stdev) = MvNormal(θ, stdev * I);
rng = MersenneTwister(0);