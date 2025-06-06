import torch
import time
import wandb
from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass, field
from src.tensor_products import (
    project_tensor_product,
    project_tensor_product_inter_channel,
    project_tensor_product_intra_channel,
    sparse_project_tensor_product,
    get_clebsch_gordon,
)
from src.sparse_tensor_products import (
    project_tensor_product_torch_sparse,
    get_sparse_coo_clebsch_gordon,
    project_tensor_product_e3nn,
    project_tensor_product_torch_sparse_v3,
)
from equiFFT import generic_equiFFT


"""
TO DO: 
-> benchmark L vs memory for all fivemethods
-> extract plot L vs memory on log-log scale y = mx + b,
-> Zhao: take your spare implementation and extract the expoent for L
--> add other two methods
"""


@dataclass
class BenchmarkRunConfig:
    # Parameters for tensor creation and identification
    ell1: int  # Angular momentum of the first input tensor (u)
    ell2: int  # Angular momentum of the second input tensor (v)
    ell_out: int  # Angular momentum of the output tensor (y)
    m1: int  # Multiplicity of the first input tensor (u)
    m2: int  # Multiplicity of the second input tensor (v)
    batch_size: int  # Batch size for the input tensors
    num_tokens: int  # Number of tokens/sequence length for the input tensors
    projection_type: str  # Type of projection, e.g., "inter" or "intra"
    function_to_run: Callable  # The actual tensor product function to benchmark
    device: str = "cuda"  # Device to run benchmarks on ("cuda" or "cpu")
    seed: Optional[int] = None  # Optional seed for reproducibility of a single run
    operation_mode: str = (
        "direct_tp"  # "direct_tp" (just TP) or "convolution" (FFT + TP + IFFT)
    )

    # To store results, Optional as they are filled after benchmark
    mean_time: Optional[float] = None  # Mean execution time over num_runs
    std_time: Optional[float] = None  # Standard deviation of execution time
    min_time: Optional[float] = None  # Minimum execution time observed
    max_time: Optional[float] = None  # Maximum execution time observed
    peak_memory: Optional[float] = (
        None  # Peak GPU memory usage in MB during the benchmark
    )
    cg_nnz: Optional[int] = (
        None  # Number of non-zero elements in Clebsch-Gordan coefficients (for sparse methods)
    )
    cg_density: Optional[float] = (
        None  # Density (nnz/total) of CG coefficients (for sparse methods)
    )

    def get_id_dict(self) -> Dict[str, Any]:
        return {
            "function": self.function_to_run.__name__,
            "ell1": self.ell1,
            "ell2": self.ell2,
            "ell_out": self.ell_out,
            "m1": self.m1,
            "m2": self.m2,
            "projection_type": self.projection_type,
            "batch_size": self.batch_size,
            "num_tokens": self.num_tokens,
            "device": self.device,
            "seed": self.seed,  # Include seed in the identifier dict
            "operation_mode": self.operation_mode,  # Include operation_mode in dict
            "cg_nnz": self.cg_nnz,
            "cg_density": self.cg_density,
        }


def benchmark_tensor_product(
    func: Callable,
    u: torch.Tensor,
    v: torch.Tensor,
    ell_out: int,
    config: BenchmarkRunConfig,  # Pass the whole config for context
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark a tensor product function, or a convolution using it."""
    # ell1 and ell2 are now part of config, but also derivable from u,v
    # For consistency, we can use them from config or ensure they match.
    # Here, using u,v shape as it's direct from the tensor.
    current_ell1 = (u.shape[2] - 1) // 2
    current_ell2 = (v.shape[2] - 1) // 2

    # Set seed for reproducibility if provided in config for this specific call
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    # Determine actual projection_type for functions that don't take it explicitly
    actual_projection_type = config.projection_type
    if func.__name__ == "project_tensor_product_inter_channel":
        actual_projection_type = "inter"
    elif func.__name__ == "project_tensor_product_intra_channel":
        actual_projection_type = "intra"

    # Get CG NNZ count for sparse methods before benchmarking
    cg_nnz = None
    cg_density = None

    if func.__name__ in [
        "sparse_project_tensor_product",
        "project_tensor_product_torch_sparse",
    ]:
        if func.__name__ == "project_tensor_product_torch_sparse":
            # For sparse_tensor_products.py implementation, get nnz directly from sparse tensor
            sparse_cg = get_sparse_coo_clebsch_gordon(
                J=ell_out,
                l_in=current_ell1,
                l_out=current_ell2,
                device=u.device,
                dtype=u.dtype,
            )
            cg_nnz = sparse_cg.indices().shape[1]  # Number of non-zero elements
            total_elements = (
                sparse_cg.shape[0] * sparse_cg.shape[1] * sparse_cg.shape[2]
            )
            cg_density = float(cg_nnz) / total_elements if total_elements > 0 else 0.0
        else:
            # For tensor_products.py sparse implementation, compute from dense CG
            dense_cg = get_clebsch_gordon(
                ell_out, current_ell1, current_ell2, device=u.device
            )
            cg_nnz = torch.nonzero(dense_cg).shape[0]
            total_elements = dense_cg.numel()
            cg_density = float(cg_nnz) / total_elements if total_elements > 0 else 0.0

    # --- Helper to call the core operation (TP or TP within convolution) ---
    def _execute_operation(op_u, op_v, op_ell_out, tp_func, op_mode, proj_type):
        """Execute tensor product directly or wrapped in a convolution."""
        if op_mode == "convolution":
            # Use the generic_equiFFT function to handle convolution
            return generic_equiFFT(
                u=op_u,
                v=op_v,
                ell_out=op_ell_out,
                tensor_product_fn=tp_func,
                projection_type=proj_type,
            )
        else:
            # Direct tensor product mode - call TP function directly
            tp_input_ell1 = (op_u.shape[2] - 1) // 2
            tp_input_ell2 = (op_v.shape[2] - 1) // 2
            func_name = tp_func.__name__

            if func_name == "project_tensor_product":
                return tp_func(op_u, op_v, op_ell_out, type=proj_type)
            elif func_name in [
                "project_tensor_product_inter_channel",
                "project_tensor_product_intra_channel",
            ]:
                return tp_func(op_u, op_v, op_ell_out)
            elif func_name == "sparse_project_tensor_product":
                return tp_func(op_u, op_v, tp_input_ell1, tp_input_ell2, op_ell_out)
            elif func_name in [
                "project_tensor_product_torch_sparse",
                "project_tensor_product_e3nn",
                "project_tensor_product_torch_sparse_v2",
                "project_tensor_product_torch_sparse_v3",
                "project_tensor_product_torch_sparse_v4",
                "project_tensor_product_torch_sparse_v1_einsum",
                "project_tensor_product_torch_sparse_v4_einsum",
            ]:
                return tp_func(op_u, op_v, op_ell_out, projection_type=proj_type)
            else:
                raise ValueError(f"Unsupported function {func_name} for direct_tp mode")

    # Warmup
    for _ in range(num_warmup):
        _ = _execute_operation(
            u, v, ell_out, func, config.operation_mode, actual_projection_type
        )

    # Synchronize and reset memory stats before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        output = _execute_operation(
            u, v, ell_out, func, config.operation_mode, actual_projection_type
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        # Minimal check to ensure function ran
        if output is None:
            print(f"Warning: {func.__name__} returned None")

    peak_mem = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )

    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "peak_memory": float(peak_mem),
        "cg_nnz": cg_nnz,
        "cg_density": cg_density,
    }


def run_benchmarks(configs: List[BenchmarkRunConfig]) -> List[Dict[str, Any]]:
    """Run benchmarks for a list of tensor product configurations."""
    results = []

    for config_idx, config in enumerate(configs):
        print(
            f"Running benchmark {config_idx+1}/{len(configs)}: {config.function_to_run.__name__} "
            f"l1={config.ell1}, l2={config.ell2}, l_out={config.ell_out}, "
            f"m1={config.m1}, m2={config.m2}, N={config.num_tokens}, B={config.batch_size}, "
            f"type={config.projection_type}, mode={config.operation_mode}, seed={config.seed}"
        )

        # Set seed for reproducible tensor generation for this specific config run
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        # Skip invalid intra-channel configurations early
        if config.projection_type == "intra" and config.m1 != config.m2:
            print(f"  Skipping intra-channel for m1({config.m1}) != m2({config.m2})")
            continue
        if (
            config.function_to_run.__name__ == "project_tensor_product_intra_channel"
            and config.m1 != config.m2
        ):
            print(
                f"  Skipping project_tensor_product_intra_channel for m1({config.m1}) != m2({config.m2})"
            )
            continue
        if (
            config.function_to_run.__name__ == "sparse_project_tensor_product"
            and config.projection_type == "intra"
            and config.m1 != config.m2
        ):
            print(
                f"  Skipping sparse_project_tensor_product for intra-channel with m1({config.m1}) != m2({config.m2})"
            )
            continue
        # Skip e3nn intra projection if m1 != m2 (it raises an error)
        if (
            config.function_to_run.__name__ == "project_tensor_product_e3nn"
            and config.projection_type == "intra"
            and config.m1 != config.m2
        ):
            print(
                f"  Skipping project_tensor_product_e3nn for intra-channel with m1({config.m1}) != m2({config.m2})"
            )
            continue

        # Create input tensors
        u = torch.randn(
            config.batch_size,
            config.num_tokens,
            2 * config.ell1 + 1,
            config.m1,
            device=config.device,
            dtype=torch.float32,
        )
        v = torch.randn(
            config.batch_size,
            config.num_tokens,
            2 * config.ell2 + 1,
            config.m2,
            device=config.device,
            dtype=torch.float32,
        )

        # Benchmark the function from config
        metrics = benchmark_tensor_product(
            config.function_to_run, u, v, config.ell_out, config  # Pass config
        )

        # Combine config identifiers and metrics
        run_data = config.get_id_dict()
        run_data.update(metrics)

        results.append(run_data)

        # Optional: log to wandb here if desired per run, or aggregate and log later
        # wandb.log(run_data)

    return results


if __name__ == "__main__":
    # Define a list of functions to benchmark
    ALL_FUNCTIONS_TO_BENCHMARK = [
        project_tensor_product,
        project_tensor_product_inter_channel,
        project_tensor_product_intra_channel,
        sparse_project_tensor_product,
        project_tensor_product_torch_sparse,
        project_tensor_product_e3nn,
        project_tensor_product_torch_sparse_v3,
    ]

    # Manually create a list of BenchmarkRunConfig instances
    benchmark_configurations: List[BenchmarkRunConfig] = []

    # Example: A small set of configurations for testing
    max_degree_test = 1  # Small for quick test
    test_batch_size = 4
    test_num_tokens = 64
    test_multiplicities = [1, 2]  # m1 and m2 will be iterated from this
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    projection_types = ["intra"]  # Only "intra" for now, "inter" not used now
    operation_modes_to_test = [
        "direct_tp",
        "convolution",
    ]  # Test both direct TP and convolution

    for func_to_run in ALL_FUNCTIONS_TO_BENCHMARK:
        for ell1 in range(max_degree_test + 1):
            for ell2 in range(max_degree_test + 1):
                for ell_out in range(
                    abs(ell1 - ell2), min(ell1 + ell2, max_degree_test) + 1
                ):  # Corrected Clebsch-Gordan range
                    for m1 in test_multiplicities:
                        for m2 in test_multiplicities:
                            for proj_type in projection_types:
                                for (
                                    op_mode
                                ) in (
                                    operation_modes_to_test
                                ):  # Iterate over operation modes
                                    # Basic filtering:
                                    if proj_type == "intra" and m1 != m2:
                                        continue
                                    if (
                                        func_to_run.__name__
                                        == "project_tensor_product_intra_channel"
                                        and m1 != m2
                                    ):
                                        continue
                                    if (
                                        func_to_run.__name__
                                        == "project_tensor_product_inter_channel"
                                        and proj_type == "intra"
                                    ):
                                        # This function is always inter, so skip if user explicitly asks for intra for it.
                                        continue

                                    # Skip e3nn for intra projection if m1 != m2
                                    if (
                                        func_to_run.__name__
                                        == "project_tensor_product_e3nn"
                                        and proj_type == "intra"
                                        and m1 != m2
                                    ):
                                        print(
                                            f"  Skipping project_tensor_product_e3nn for intra-channel with m1({m1}) != m2({m2})"
                                        )
                                        continue

                                    config = BenchmarkRunConfig(
                                        ell1=ell1,
                                        ell2=ell2,
                                        ell_out=ell_out,
                                        m1=m1,
                                        m2=m2,
                                        batch_size=test_batch_size,
                                        num_tokens=test_num_tokens,
                                        projection_type=proj_type,
                                        function_to_run=func_to_run,
                                        device=device_to_use,
                                        seed=0,  # Example seed for testing this script directly
                                        operation_mode=op_mode,  # Set operation mode
                                    )
                                    benchmark_configurations.append(config)

    if not benchmark_configurations:
        print("No benchmark configurations generated. Check loops and conditions.")
    else:
        print(f"Generated {len(benchmark_configurations)} benchmark configurations.")

        # Initialize wandb (optional, can be done in the calling script like plot_tensor_product_analysis.py)
        # wandb.init(project="SE3-Hyena-TensorProduct-Bench", name="manual_config_run")

        results_data = run_benchmarks(configs=benchmark_configurations)

        print("\n--- Benchmark Results ---")
        for res in results_data:
            print(res)
            # wandb.log(res) # Log aggregated results if not logged per run

        # wandb.finish() # if initialized here

        # Example: Save to a file (e.g., CSV)
        # import pandas as pd
        # df = pd.DataFrame(results_data)
        # df.to_csv("benchmark_results.csv", index=False)
        # print("\nResults saved to benchmark_results.csv")
