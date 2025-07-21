include("utils/utils.jl")

using StatsPlots

# Load data
function load_data(dir)
    log_sl = load(joinpath(logdir, dir * "_SIL_training_results.jld2"))
    log_ppo = load(joinpath(logdir, dir * "_PPO_training_results.jld2"))
    log_il = load(joinpath(logdir, dir * "_SRL_training_results.jld2"))
    log_bl = load(joinpath(logdir, dir * "_baselines.jld2"))

    train_data = [
        log_sl["val_rew"][1:(end - 1)], log_ppo["val_rew"][1:(end - 1)], log_il["val_rew"][1:(end - 1)]
    ]

    final_data = [log_sl["train_final"], log_ppo["train_final"], log_il["train_final"],
        log_bl["expert_train"], log_bl["greedy_train"],
        log_sl["test_final"], log_ppo["test_final"], log_il["test_final"],
        log_bl["expert_test"], log_bl["greedy_test"]]

    return train_data, final_data
end

# Basic lineplot
function training_plot(
    data;
    include_legend=true,
    cumulative=true,
    title,
    ylabel_text,
    yticks=nothing,
    size=(800, 400),
)
    # Apply cumulative max if requested
    processed_data = data
    if cumulative
        processed_data = [accumulate(max, d) for d in data]
    end

    plt = plot(
        processed_data[1];
        label="SIL",
        color=:blue,
        linewidth=4,
        xlabel="training episode",
        title=title,
        labelfontsize=16,
        tickfontsize=14,
        size=size,
    )

    # Set y-label
    plot!(plt; ylabel=ylabel_text, labelfontsize=16)

    if include_legend
        plot!(plt; legend=:bottomright, legendfontsize=16)
    else
        plot!(plt; legend=:none)
    end

    plot!(plt, processed_data[2]; label="PPO", color=:green, linewidth=4)
    plot!(plt, processed_data[3]; label="SRL", color=:red, linewidth=4)

    if !isnothing(yticks)
        yticks!(plt, yticks)
    end

    return plt
end

# Basic boxplot
function create_boxplot(data, colors, labels; title, yticks=nothing, ylims=(-100, 100))
    positions = collect(1:length(data))
    plt = plot(ylims=ylims)
    # plt = plot(yscale=:log10; tex_output_standalone=standalone)

    for i in 1:length(data)
        boxplot!(
            fill(positions[i], length(data[i])),
            data[i];
            fillcolor=colors[i],
            alpha=0.6,
            linewidth=2,
            linecolor=:black,
            legend=false,
            label="",
            title=title,
            outliers=false,
        )
        scatter!(
            [positions[i]],
            [mean(data[i])];
            markershape=:circle,
            markersize=4,
            markercolor=:black,
            label="",
        )
    end

    # Apply custom yticks if provided
    if !isnothing(yticks)
        yticks!(plt, yticks...)
    end

    xticks!(positions, labels; tickfontsize=14, xrotation=45)
    return plt
end

function boxplot_greedy(
    data,
    factor; # Factor to accound for positive or negative rewards
    size=(800, 400),
    log_ticks=[-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3],
    ylimits = (-100, 100),
    ytext="Env: delta greedy (%)",
)
    ε = eps()
    train_benchmark = data[5]
    test_benchmark = data[10]
    data_plot = data[[1, 2, 3, 4, 6, 7, 8, 9]]

    # Calculate percentage differences from benchmark
    for i in 1:4
        data_plot[i] = factor .* (data_plot[i] .- train_benchmark) .* 100 ./ train_benchmark
    end
    for i in 5:8
        data_plot[i] = factor .* (data_plot[i] .- test_benchmark) .* 100 ./ test_benchmark
    end

    # Apply log transform to data
    data_plot = [
        sign.(data_series) .* log10.(ε .+ abs.(data_series)) for data_series in data_plot
    ]

    # Filter log ticks based on transformed data range
    filtered_log_ticks = log_ticks

    # Calculate original percentage values for these tick positions (for labels)
    # Inverse of sign(x) * log10(1 + |x|) is sign(y) * (10^|y| - 1)
    orig_ticks = [sign(t) * (10^abs(t) - ε) for t in filtered_log_ticks]

    # Format as percentages with appropriate precision
    tick_labels = ["$(round(Int, t))" for t in orig_ticks]

    labels = ["SIL", "PPO", "SRL", "expert"]
    colors = [:blue, :green, :red, :orange]

    # Create two boxplots with custom ticks
    plt_train = create_boxplot(
        data_plot[1:4],
        colors,
        labels;
        title="train",
        yticks=(filtered_log_ticks, tick_labels),
        ylims=ylimits,
    )

    plt_test = create_boxplot(
        data_plot[5:8],
        colors,
        labels;
        title="test",
        yticks=(filtered_log_ticks, tick_labels),
        ylims=ylimits,
    )

    # Combine them side by side
    plt = plot(plt_train, plt_test; layout=(1, 2), size)
    ylabel!(plt[1], ytext; labelfontsize=16)
    return plt
end

# Data loading

smsp_training, smsp_final = load_data("smsp") .* (-0.0001); # Factor to account for reward scale

# Training plot
smsp_training_plot = training_plot(
    smsp_training;
    include_legend=true,
    cumulative=true,
    title="SMSP",
    ylabel_text="val. rew. (10^4)",
    yticks=[-30, -25, -20, -15],
);
display(smsp_training_plot)
savefig(smsp_training_plot, joinpath(plotdir, "smsp_training_plot.pdf"))

# Results plot
smsp_results_plot = boxplot_greedy(
    smsp_final,
    -1;
    log_ticks=[-2, -1, 0, 1, 2, 3],
    ylimits=(-2.5, 3),
    ytext="SMSP: delta greedy (%)",
);
display(smsp_results_plot)
savefig(smsp_results_plot, joinpath(plotdir, "smsp_results_plot.pdf"))
