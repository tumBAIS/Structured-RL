# Structured Reinforcement Learning for Combinatorial Decision-Making

This folder contains the code for the paper

> Heiko Hoppe, Léo Baty, Louis Bouvier, Axel Parmentier, Maximilian Schiffer (2025). Structured Reinforcement Learning for Combinatorial Decision-Making. arXiv preprint on arXiv: tba.

The code implements COaML-pipelines trained using Structured Reinforcement Learning (SRL), Structured Imitation Learning (SIL), and Proximal Policy Optimization (PPO) for six industrial problem settings using Julia 1.11.5.

## Folder Structure

The folder scripts contains all source code for the paper. It contains a sub-folder for each of the environments:
1. DAP: Dynamic Assortment Problem
2. DVSP: Dynamic Vehicle Scheduling Problem
3. GSPP: Gridworld Shortest Paths Problem
4. SMSP: Single Machine Scheduling Problem
5. SVSP: Stochastic Vehicle Scheduling Problem
6. WSPP: Warcraft Shortest Paths Problem

The folder of each environment contains an implementation of SIL, PPO, and SRL, as well as a greedy and an expert benchmark for the specific environment. Each environment-folder is sturctured as follows:
1. utils: Folder containing environment funcions, should not be run directly
2. 00_setup.jl: Dataset setup and baseline (expert and greedy) solutions
3. 01_SIL.jl: Structured Imitation Learning training function and executable code
4. 02_PPO.jl: Proximal Policy Optimization training function and executable code
5. 03_SRL.jl: Structured Reinforcement Learning training function and executable code
6. 04_plots.jl: Code to create a cumulative lineplot of training performance and a boxplot of testing performance

## Environment setup

To set up a working environment for the code, please follow these steps:
1. Install the Julia programming language, version 1.11.5 (see https://julialang.org/install/)
2. Open this software in your favorite IDE and activate a Julia REPL
3. Instantiate the Julia environment of this folder:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
4. Make sure to have an active internet connection and ca. 150MB of free disc space for downloading and storing instance and log files when running the code for the first time

## Running code

To train and test the algorithms for an environment, please follow these steps:
1. Find the corresponding environment folder
2. Run 00_setup.jl:
```bash
julia --project=. folder/00_setup.jl
```
3. Run the algorithm scripts 01_SIL.jl, 02_PPO.jl, and 03_SRL.jl (same as 2.)
4. Run 04_plots.jl (same as 2.)

To reproduce the results from the paper, please run the algorithms using ten random seeds and average the rewards across these seeds. The seeds used in the paper are stated in the respective setup script.
