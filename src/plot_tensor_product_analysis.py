# Updated imports from benchmark_tensor_products
from benchmark_tensor_products import (
    run_benchmarks,
    BenchmarkRunConfig,
)
from tensor_products import (
    project_tensor_product,
    project_tensor_product_inter_channel,
    project_tensor_product_intra_channel,
    sparse_project_tensor_product,
)
from sparse_tensor_products import (
    project_tensor_product_e3nn,
    project_tensor_product_torch_sparse,
    project_tensor_product_torch_sparse_v2,
    project_tensor_product_torch_sparse_v3,
    project_tensor_product_torch_sparse_v4,
    project_tensor_product_torch_sparse_v1_einsum,
    project_tensor_product_torch_sparse_v4_einsum,
)

# Define the list of functions to benchmark, similar to benchmark_tensor_products.py
# Note: project_tensor_product_torch_sparse is the one we want to optimize
# The others are baselines for comparison
ALL_FUNCTIONS_TO_BENCHMARK = [
    # Our optimization target from sparse_tensor_products.py:
    # project_tensor_product_torch_sparse,
    project_tensor_product_torch_sparse_v2,  # NOTE: test v2
    # project_tensor_product_torch_sparse_v3,  # NOTE: test v3, with SU(2) half integers
    # project_tensor_product_torch_sparse_v4,  # NOTE: test v4: new SO(3) sparse method; still debugging
    # project_tensor_product_torch_sparse_v1_einsum,
    # project_tensor_product_torch_sparse_v4_einsum,
    # Baseline methods for comparison:
    project_tensor_product,  # Wrapper that calls _inter/_intra internally
    # project_tensor_product_inter_channel,  # Don't need to benchmark
    # project_tensor_product_intra_channel,  # Don't need to benchmark
    # sparse_project_tensor_product,  # Original sparse method (Python loop based)
]


def plot_aggregated_metrics(
    data: pd.DataFrame,
    metric_col_mean: str,  # e.g., 'peak_memory' (which is now mean after aggregation)
    metric_col_std: str,  # e.g., 'peak_memory_std_across_seeds'
    group_by_col: str,  # e.g., 'ell_out' or 'num_tokens'
    save_path: str,
    title: str,
    x_label: str,
    y_label: str,
    figsize: Tuple[int, int] = (4, 3),
    log_x: bool = False,
    log_y: bool = True,
    plot_title_suffix: str = "",  # Add suffix to title (e.g., " (Direct TP)" or " (Convolution)")
):
    """Helper function to plot aggregated metrics (mean with std error bars from seeds)."""
    final_title = title + plot_title_suffix  # Add suffix to title
    print(f"\nGenerating plot: {final_title}...")
    start_time = time.time()

    if data.empty:
        print(f"WARNING: DataFrame is empty for plot '{final_title}'. No data to plot.")
        return

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Set up the plot with a light gray grid that matches the image
    ax = plt.gca()
    ax.grid(True, which="both", color="lightgray", linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)  # Put grid below data

    implementations = data["function"].unique()
    projection_types = data["projection_type"].unique()

    # Use custom color scheme similar to the image
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    markers = ["o", "s", "^", "D", "P"]

    # Map function names to shorter display names
    display_names = {
        "project_tensor_product": "TP (Dense)",
        "project_tensor_product_torch_sparse_v2": "TP (Sparse)",
        "project_tensor_product_e3nn": "TP (e3nn)",
        # 'project_tensor_product_torch_sparse': 'Long-Conv'
    }

    plot_idx = 0
    lines = []
    labels = []

    for impl_name in implementations:
        for proj_type in projection_types:
            current_data_segment = data[
                (data["function"] == impl_name) & (data["projection_type"] == proj_type)
            ]
            if proj_type == "intra":
                current_data_segment = current_data_segment[
                    current_data_segment["m1"] == current_data_segment["m2"]
                ]

            if current_data_segment.empty:
                continue

            grouped_for_plot = current_data_segment.sort_values(by=group_by_col)
            x_coords = grouped_for_plot[group_by_col].unique()
            means = grouped_for_plot.groupby(group_by_col)[metric_col_mean].first()
            stds = grouped_for_plot.groupby(group_by_col)[metric_col_std].first()

            means = means.reindex(x_coords)
            stds = stds.reindex(x_coords).fillna(0)

            if means.empty:
                print(
                    f"  No valid mean data for {impl_name} ({proj_type}) for metric {metric_col_mean} grouped by {group_by_col}"
                )
                continue

            # Use simplified function names for display
            display_name = display_names.get(
                impl_name,
                impl_name.replace("project_tensor_product_", "")
                .replace("_", "-")
                .title(),
            )

            # Don't show projection type in label to keep it clean
            # color = colors.get(display_name, f'C{plot_idx}')
            # marker = markers.get(display_name, 'o')

            line = plt.errorbar(
                x_coords,
                means,
                yerr=stds.to_numpy() if not stds.empty else np.array([]),
                label=display_name,
                marker="o",
                # color=color,
                capsize=3,
                linestyle="-",
                elinewidth=1,
                capthick=1,
                markersize=7,
                linewidth=2,
            )
            plot_idx += 1
            lines.append(line)
            labels.append(display_name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(final_title)  # Use final_title with suffix
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    if group_by_col == "ell_out":
        min_l_val = data[group_by_col].min()
        max_l_val = data[group_by_col].max()
        if pd.notna(min_l_val) and pd.notna(max_l_val) and max_l_val >= min_l_val:
            plt.xticks(np.arange(int(min_l_val), int(max_l_val) + 1, 1.0))

    # Tighter positioning for the legend
    plt.legend(loc="lower right", framealpha=0.5)

    # Fix linter error: rect should be a tuple not a list
    plt.tight_layout(rect=(0, 0, 1, 1))

    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    end_time = time.time()
    print(
        f"Plot saved to: {os.path.abspath(save_path)}. Time: {end_time - start_time:.2f}s"
    )


def plot_sparsity_metrics(
    data: pd.DataFrame,
    save_path: str,
    figsize: Tuple[int, int] = (10, 5),
    plot_title_suffix: str = "",  # Add suffix to title
):
    """Plot sparsity metrics (nnz and density) for Clebsch-Gordon coefficients vs ell_out (J)."""
    # Apply suffix to titles
    final_title_nnz = f"CG Sparsity: Non-Zero Elements vs J{plot_title_suffix}"
    final_title_density = f"CG Sparsity: Density vs J{plot_title_suffix}"
    print(f"\nGenerating sparsity analysis plot...")
    start_time = time.time()

    # Filter for sparse methods only
    sparse_methods = [
        "sparse_project_tensor_product",
        "project_tensor_product_torch_sparse",
    ]
    sparse_data = data[data["function"].isin(sparse_methods)]

    if sparse_data.empty:
        print(
            f"WARNING: No data for sparse methods. Cannot generate sparsity analysis."
        )
        return

    # Create figure with two subplots (nnz and density)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Style setup
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    # Set up grid styling to match the main plots
    for ax in [ax1, ax2]:
        ax.grid(True, which="both", color="lightgray", linestyle="-", alpha=0.7)
        ax.set_axisbelow(True)  # Put grid below data

    # Define colors for methods
    colors = {
        "Sparse Original": "#1f77b4",  # Blue
        "Torch Sparse": "#ff7f0e",  # Orange
    }

    # Map function names to shorter display names
    display_names = {
        "sparse_project_tensor_product": "Sparse Original",
        "project_tensor_product_torch_sparse": "Torch Sparse",
    }

    # Plot 1: NNZ vs ell_out
    for idx, method in enumerate(sparse_methods):
        method_data = sparse_data[sparse_data["function"] == method]
        if method_data.empty:
            continue

        # Group by ell1, ell2, ell_out to get unique combinations
        groups = method_data.groupby(["ell1", "ell2"])

        for group_idx, ((l1, l2), group_data) in enumerate(groups):
            display_name = f"{display_names.get(method, method)} (l1={l1},l2={l2})"
            method_display = display_names.get(method, method)

            # Sort by ell_out
            plot_data = group_data.sort_values("ell_out")

            # Plot NNZ vs ell_out
            color = colors.get(method_display, f"C{idx}")
            ax1.plot(
                plot_data["ell_out"],
                plot_data["cg_nnz_avg"],
                label=display_name,
                marker="o",
                color=color,
                linestyle="-",
                markersize=7,
                linewidth=2,
            )

    ax1.set_xlabel("Output Harmonic Order (J)")
    ax1.set_ylabel("Non-Zero CG Coefficients")
    ax1.set_title(final_title_nnz)  # Use final title with suffix
    ax1.legend(loc="best", framealpha=0.5)

    # Plot 2: Density vs ell_out
    for idx, method in enumerate(sparse_methods):
        method_data = sparse_data[sparse_data["function"] == method]
        if method_data.empty:
            continue

        # Group by ell1, ell2, ell_out
        groups = method_data.groupby(["ell1", "ell2"])

        for group_idx, ((l1, l2), group_data) in enumerate(groups):
            display_name = f"{display_names.get(method, method)} (l1={l1},l2={l2})"
            method_display = display_names.get(method, method)

            # Sort by ell_out
            plot_data = group_data.sort_values("ell_out")

            # Plot density vs ell_out
            color = colors.get(method_display, f"C{idx}")
            ax2.plot(
                plot_data["ell_out"],
                plot_data["cg_density_avg"],
                label=display_name,
                marker="o",
                color=color,
                linestyle="-",
                markersize=7,
                linewidth=2,
            )

    ax2.set_xlabel("Output Harmonic Order (J)")
    ax2.set_ylabel("Density of CG Coefficients")
    ax2.set_title(final_title_density)  # Use final title with suffix
    ax2.legend(loc="best", framealpha=0.5)

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    end_time = time.time()
    print(
        f"Sparsity analysis plot saved to: {os.path.abspath(save_path)}. Time: {end_time - start_time:.2f}s"
    )


def run_experiment(
    configs: List[BenchmarkRunConfig], exp_name: str, plots_dir: str
) -> pd.DataFrame:
    """Run a specific experiment with its own configs and save results separately."""
    print(
        f"\n{'='*80}\nRunning Experiment: {exp_name} ({len(configs)} configurations)\n{'='*80}"
    )

    exp_start_time = time.time()
    results_raw_data = run_benchmarks(configs=configs)

    if not results_raw_data:
        print(f"No results obtained from {exp_name}. Skipping.")
        return pd.DataFrame()

    # Save raw results
    df_raw = pd.DataFrame(results_raw_data)
    raw_results_path = os.path.join(plots_dir, f"{exp_name}_raw_results.csv")
    df_raw.to_csv(raw_results_path, index=False)
    print(f"Raw benchmark results saved to: {raw_results_path}")

    # Aggregate results by taking the mean and std over seeds
    print(f"\nAggregating results over seeds for {exp_name}...")
    grouping_cols = [
        "function",
        "ell1",
        "ell2",
        "ell_out",
        "m1",
        "m2",
        "projection_type",
        "batch_size",
        "num_tokens",
        "device",
        "operation_mode",
    ]  # Include operation_mode in grouping columns

    # Fixed for linter - Use named aggregation
    df_aggregated = (
        df_raw.groupby(grouping_cols)
        .agg(
            mean_time_avg_seeds=("mean_time", "mean"),
            mean_time_std_seeds=("mean_time", "std"),
            peak_memory_avg_seeds=("peak_memory", "mean"),
            peak_memory_std_seeds=("peak_memory", "std"),
            # For sparse methods, these will be the same for all seeds of a config
            cg_nnz_avg=("cg_nnz", "first"),  # Just take first value from seeds
            cg_density_avg=("cg_density", "first"),
        )
        .reset_index()
    )

    # Save aggregated results
    agg_results_path = os.path.join(plots_dir, f"{exp_name}_aggregated_results.csv")
    df_aggregated.to_csv(agg_results_path, index=False)
    print(f"Aggregated benchmark results saved to: {agg_results_path}")

    exp_end_time = time.time()
    print(
        f"Experiment {exp_name} completed in {exp_end_time - exp_start_time:.2f} seconds."
    )

    return df_aggregated


def main():
    print("Starting tensor product benchmarking for plotting analysis...")
    total_start_time = time.time()

    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    if device_to_use == "cpu":
        print("WARNING: CUDA is not available. Running on CPU may be very slow.")
    else:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    plots_dir = "plots_tensor_products"
    os.makedirs(plots_dir, exist_ok=True)

    seeds = [123, 456, 789]
    projection_types = ["intra"]  # Only "intra" for now, "inter" not used now
    # operation_modes = ["direct_tp", "convolution"]  # Test both direct TP and convolution with FFT
    operation_modes = ["direct_tp"]  # Test only direct TP for now

    # ======= EXPERIMENT 1: Varying J (ell_out) =======
    # exp1_ell1 = 6  # Input angular momentum 1
    # exp1_ell2 = 6  # Input angular momentum 2
    # exp1_min_ell_out = abs(exp1_ell1 - exp1_ell2)  # Should be 0
    # exp1_max_ell_out = exp1_ell1 + exp1_ell2       # Should be 12
    # exp1_batch_size = 8
    # exp1_num_tokens = 128
    # exp1_m1, exp1_m2 = 2, 2  # Fixed multiplicities for this experiment

    # # NOTE: experiment set 2
    # exp1_ell1 = 8  # Input angular momentum 1
    # exp1_ell2 = 8  # Input angular momentum 2
    # exp1_min_ell_out = abs(exp1_ell1 - exp1_ell2)  # Should be 0
    # exp1_max_ell_out = exp1_ell1 + exp1_ell2       # Should be 16
    # exp1_batch_size = 8
    # exp1_num_tokens = 2048
    # exp1_m1, exp1_m2 = 1, 1  # Fixed multiplicities for this experiment

    # NOTE: experiment set 3
    exp1_ell1 = 8  # Input angular momentum 1
    exp1_ell2 = 8  # Input angular momentum 2
    exp1_min_ell_out = abs(exp1_ell1 - exp1_ell2)  # Should be 0
    exp1_max_ell_out = exp1_ell1 + exp1_ell2  # Should be 16
    exp1_batch_size = 8
    exp1_num_tokens = 128
    exp1_m1, exp1_m2 = 3, 3  # Fixed multiplicities for this experiment

    print(
        f"Generating Experiment 1 configs (Vary J=ell_out): ell1={exp1_ell1}, ell2={exp1_ell2}, B={exp1_batch_size}, N={exp1_num_tokens}, m1={exp1_m1}, m2={exp1_m2}"
    )

    exp1_configs = []
    for func_to_run in ALL_FUNCTIONS_TO_BENCHMARK:
        for ell_out_val in range(exp1_min_ell_out, exp1_max_ell_out + 1):
            for proj_type in projection_types:
                for op_mode in operation_modes:  # Test both operation modes
                    # Skip invalid projection types for this m1, m2 configuration or function
                    if proj_type == "intra" and exp1_m1 != exp1_m2:
                        continue
                    if (
                        func_to_run.__name__ == "project_tensor_product_intra_channel"
                        and exp1_m1 != exp1_m2
                    ):
                        continue
                    if (
                        func_to_run.__name__ == "project_tensor_product_inter_channel"
                        and proj_type == "intra"
                    ):
                        continue
                    if (
                        func_to_run.__name__ == "project_tensor_product_e3nn"
                        and proj_type == "intra"
                        and exp1_m1 != exp1_m2
                    ):
                        continue

                    for seed_val in seeds:
                        config = BenchmarkRunConfig(
                            ell1=exp1_ell1,
                            ell2=exp1_ell2,
                            ell_out=ell_out_val,
                            m1=exp1_m1,
                            m2=exp1_m2,
                            batch_size=exp1_batch_size,
                            num_tokens=exp1_num_tokens,
                            projection_type=proj_type,
                            function_to_run=func_to_run,
                            device=device_to_use,
                            seed=seed_val,
                            operation_mode=op_mode,  # Set operation mode explicitly
                        )
                        exp1_configs.append(config)

    # Run Experiment 1 separately
    df_exp1 = run_experiment(exp1_configs, "exp1_vary_J", plots_dir)

    # Plot Experiment 1 Results for each operation_mode separately
    if not df_exp1.empty:
        for op_mode in df_exp1["operation_mode"].unique():
            # Filter data for this operation mode
            df_mode_subset = df_exp1[df_exp1["operation_mode"] == op_mode]
            # Create title suffix and file suffix for this mode
            mode_suffix = (
                " (Tensor Product)"
                if op_mode == "direct_tp"
                else f" ({op_mode.replace('_', ' ').title()})"
            )
            file_suffix = f"_{op_mode}"

            # Runtime vs J plot for this operation mode
            plot_aggregated_metrics(
                df_mode_subset,
                metric_col_mean="mean_time_avg_seeds",
                metric_col_std="mean_time_std_seeds",
                group_by_col="ell_out",
                save_path=os.path.join(
                    plots_dir, f"exp1_time_vs_J_avg{file_suffix}.pdf"
                ),
                title=f"Runtime vs J",
                x_label="Output Harmonic Order (J)",
                y_label="Runtime (s)",
                log_y=True,
                plot_title_suffix=mode_suffix,
            )

            # Memory vs J plot for this operation mode
            plot_aggregated_metrics(
                df_mode_subset,
                metric_col_mean="peak_memory_avg_seeds",
                metric_col_std="peak_memory_std_seeds",
                group_by_col="ell_out",
                save_path=os.path.join(
                    plots_dir, f"exp1_memory_vs_J_avg{file_suffix}.pdf"
                ),
                title=f"Memory vs J",
                x_label="Output Harmonic Order (J)",
                y_label="Peak Memory (MB)",
                log_y=True,
                plot_title_suffix=mode_suffix,
            )

        # Sparsity plots are properties of the CG coefficients, not dependent on operation mode
        # So we generate them once, using data from direct_tp mode
        plot_sparsity_metrics(
            df_exp1[df_exp1["operation_mode"] == "direct_tp"],
            save_path=os.path.join(plots_dir, "exp1_sparsity_analysis.pdf"),
        )

    # Clear CUDA cache between experiments
    if device_to_use == "cuda":
        torch.cuda.empty_cache()

    # ======= EXPERIMENT 2: Varying N (num_tokens) =======
    exp2_ell1 = 5  # Input angular momentum 1
    exp2_ell2 = 5  # Input angular momentum 2
    exp2_ell_out = 10  # Output angular momentum
    exp2_n_values = [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        2**15,
        2**16,
        2**17,
        2**18,
    ]  # Sequence lengths to test
    exp2_batch_size = 8
    exp2_m1, exp2_m2 = 4, 4  # Multiplicities for this experiment

    print(
        f"Generating Experiment 2 configs (Vary N): ell1={exp2_ell1}, ell2={exp2_ell2}, ell_out={exp2_ell_out}, B={exp2_batch_size}, m1={exp2_m1}, m2={exp2_m2}"
    )

    exp2_configs = []
    if not (abs(exp2_ell1 - exp2_ell2) <= exp2_ell_out <= exp2_ell1 + exp2_ell2):
        print(
            f"Warning: Experiment 2 L config (l1={exp2_ell1},l2={exp2_ell2},l_out={exp2_ell_out}) is invalid. Skipping N-sweep."
        )
    else:
        for func_to_run in ALL_FUNCTIONS_TO_BENCHMARK:
            for n_val in exp2_n_values:
                for proj_type in projection_types:
                    for op_mode in operation_modes:  # Test both operation modes
                        # Skip invalid projection types
                        if proj_type == "intra" and exp2_m1 != exp2_m2:
                            continue
                        if (
                            func_to_run.__name__
                            == "project_tensor_product_intra_channel"
                            and exp2_m1 != exp2_m2
                        ):
                            continue
                        if (
                            func_to_run.__name__
                            == "project_tensor_product_inter_channel"
                            and proj_type == "intra"
                        ):
                            continue
                        if (
                            func_to_run.__name__ == "project_tensor_product_e3nn"
                            and proj_type == "intra"
                            and exp2_m1 != exp2_m2
                        ):
                            continue

                        for seed_val in seeds:
                            config = BenchmarkRunConfig(
                                ell1=exp2_ell1,
                                ell2=exp2_ell2,
                                ell_out=exp2_ell_out,
                                m1=exp2_m1,
                                m2=exp2_m2,
                                batch_size=exp2_batch_size,
                                num_tokens=n_val,
                                projection_type=proj_type,
                                function_to_run=func_to_run,
                                device=device_to_use,
                                seed=seed_val,
                                operation_mode=op_mode,  # Set operation mode explicitly
                            )
                            exp2_configs.append(config)

    # Run Experiment 2 separately
    df_exp2 = run_experiment(exp2_configs, "exp2_vary_N", plots_dir)

    # Plot Experiment 2 Results for each operation_mode separately
    if not df_exp2.empty:
        for op_mode in df_exp2["operation_mode"].unique():
            # Filter data for this operation mode
            df_mode_subset = df_exp2[df_exp2["operation_mode"] == op_mode]
            # Create title suffix and file suffix for this mode
            mode_suffix = (
                " (Tensor Product)"
                if op_mode == "direct_tp"
                else f" ({op_mode.replace('_', ' ').title()})"
            )
            file_suffix = f"_{op_mode}"

            # Runtime vs N plot for this operation mode
            plot_aggregated_metrics(
                df_mode_subset,
                metric_col_mean="mean_time_avg_seeds",
                metric_col_std="mean_time_std_seeds",
                group_by_col="num_tokens",
                save_path=os.path.join(
                    plots_dir, f"exp2_time_vs_N_avg{file_suffix}.pdf"
                ),
                title=f"Runtime vs Sequence Length",
                x_label="Sequence Length",
                y_label="Runtime (s)",
                log_x=True,
                log_y=True,
                plot_title_suffix=mode_suffix,
            )

            # Memory vs N plot for this operation mode
            plot_aggregated_metrics(
                df_mode_subset,
                metric_col_mean="peak_memory_avg_seeds",
                metric_col_std="peak_memory_std_seeds",
                group_by_col="num_tokens",
                save_path=os.path.join(
                    plots_dir, f"exp2_memory_vs_N_avg{file_suffix}.pdf"
                ),
                title=f"Memory vs Sequence Length",
                x_label="Sequence Length",
                y_label="Peak Memory (MB)",
                log_x=True,
                log_y=True,
                plot_title_suffix=mode_suffix,
            )

    # Clear CUDA cache between experiments
    if device_to_use == "cuda":
        torch.cuda.empty_cache()

    # ======= EXPERIMENT 3: Varying Multiplicity (m1=m2) =======
    exp3_ell1 = 4
    exp3_ell2 = 4
    exp3_ell_out = 8
    exp3_batch_size = 8
    # exp3_num_tokens = 128
    exp3_num_tokens = 4096  # NOTE: test 4096
    exp3_m_values = [1, 2, 4, 8, 16, 32, 64]  # Multiplicity values to test

    print(
        f"Generating Experiment 3 configs (Vary Multiplicity): ell1={exp3_ell1}, ell2={exp3_ell2}, ell_out={exp3_ell_out}, B={exp3_batch_size}, N={exp3_num_tokens}"
    )

    exp3_configs = []
    for func_to_run in ALL_FUNCTIONS_TO_BENCHMARK:
        for m_val in exp3_m_values:
            for proj_type in projection_types:
                for op_mode in operation_modes:  # Test both operation modes
                    # Skip invalid projection types
                    if (
                        func_to_run.__name__ == "project_tensor_product_inter_channel"
                        and proj_type == "intra"
                    ):
                        continue
                    if (
                        func_to_run.__name__ == "project_tensor_product_e3nn"
                        and proj_type == "intra"
                        and m_val != m_val
                    ):
                        continue  # This is redundant but keeps the check pattern consistent

                    for seed_val in seeds:
                        config = BenchmarkRunConfig(
                            ell1=exp3_ell1,
                            ell2=exp3_ell2,
                            ell_out=exp3_ell_out,
                            m1=m_val,
                            m2=m_val,  # Same multiplicity for both inputs
                            batch_size=exp3_batch_size,
                            num_tokens=exp3_num_tokens,
                            projection_type=proj_type,
                            function_to_run=func_to_run,
                            device=device_to_use,
                            seed=seed_val,
                            operation_mode=op_mode,  # Set operation mode explicitly
                        )
                        exp3_configs.append(config)

    # Run Experiment 3 separately
    df_exp3 = run_experiment(exp3_configs, "exp3_vary_mult", plots_dir)

    # Plot Experiment 3 Results for each operation_mode separately
    if not df_exp3.empty:
        for op_mode in df_exp3["operation_mode"].unique():
            # Filter data for this operation mode
            df_mode_subset = df_exp3[df_exp3["operation_mode"] == op_mode]
            # Create title suffix and file suffix for this mode
            mode_suffix = (
                " (Tensor Product)"
                if op_mode == "direct_tp"
                else f" ({op_mode.replace('_', ' ').title()})"
            )
            file_suffix = f"_{op_mode}"

            # Runtime vs Multiplicity plot for this operation mode
            plot_aggregated_metrics(
                df_mode_subset,
                metric_col_mean="mean_time_avg_seeds",
                metric_col_std="mean_time_std_seeds",
                group_by_col="m1",  # m1=m2 in this experiment
                save_path=os.path.join(
                    plots_dir, f"exp3_time_vs_mult_avg{file_suffix}.pdf"
                ),
                title=f"Runtime vs Multiplicity",
                x_label="Multiplicity",
                y_label="Runtime (s)",
                log_x=True,
                log_y=True,
                plot_title_suffix=mode_suffix,
            )

            # Memory vs Multiplicity plot for this operation mode
            plot_aggregated_metrics(
                df_mode_subset,
                metric_col_mean="peak_memory_avg_seeds",
                metric_col_std="peak_memory_std_seeds",
                group_by_col="m1",  # m1=m2 in this experiment
                save_path=os.path.join(
                    plots_dir, f"exp3_memory_vs_mult_avg{file_suffix}.pdf"
                ),
                title=f"Memory vs Multiplicity",
                x_label="Multiplicity",
                y_label="Peak Memory (MB)",
                log_x=True,
                log_y=True,
                plot_title_suffix=mode_suffix,
            )

    total_end_time = time.time()
    print(
        f"\nTotal script execution time: {total_end_time - total_start_time:.2f} seconds"
    )
    if device_to_use == "cuda":
        torch.cuda.empty_cache()
        print(
            f"Final CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        )


if __name__ == "__main__":
    main()
