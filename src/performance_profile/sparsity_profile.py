import torch
import gc
import wandb

import os
import sys

# Add the root directory to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)


from src.tensor_products import sparse_project_tensor_product, project_tensor_product
import numpy as np
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

types = ["Dense", "Sparse"]


def profile_operation(operation, *args, burn_in=1, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # Burn-in to trigger JIT/kernel caching
    for _ in range(burn_in):
        _ = operation(*args, **kwargs)
        torch.cuda.synchronize()

    # Clear previous memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Ensure GPU is ready
    torch.cuda.synchronize()
    start_time = time.time()

    # Run the operation
    result = operation(*args, **kwargs)

    # Synchronize and measure time
    torch.cuda.synchronize()
    end_time = time.time()

    # Get peak GPU memory allocated during the operation (in MB)
    mem_used = torch.cuda.max_memory_allocated() / (1024**2)  # Convert bytes to MB

    return end_time - start_time, mem_used, result


def test_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ### multiplicities
    m_vals = [1, 2, 3, 4, 5]
    J_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ell_out_vals = [0, 1, 2, 3, 4, 5]
    ell_in_vals = [0, 2, 3, 4, 5]

    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_out_vals:
                if np.abs(ell_in_2 - ell_in_1) <= J and J <= ell_in_1 + ell_in_2:
                    for m1 in m_vals:
                        for m2 in m_vals:
                            ell_out = J

                            batch_size = 1
                            num_tokens = 200

                            # Reuse tensors between operations to avoid reallocating
                            q = torch.randn(
                                batch_size, num_tokens, 2 * ell_in_1 + 1, m1
                            ).to(device)
                            k = torch.randn(
                                batch_size, num_tokens, 2 * ell_in_2 + 1, m2
                            ).to(device)

                            # Profiling project_tensor_product and logging to wandb
                            print(
                                f"Profiling project_tensor_product for J={J}, ell_in_1={ell_in_1}, ell_in_2={ell_in_2}, m1={m1}, m2={m2}"
                            )
                            time_taken, mem_used, _ = profile_operation(
                                project_tensor_product,
                                q,
                                k,
                                ell_in_1,
                                ell_in_2,
                                ell_out,
                            )

                            # Log to wandb
                            wandb.log(
                                {
                                    "Time_project_tensor_product": time_taken,
                                    "Memory_project_tensor_product": mem_used,
                                    "J": J,
                                    "ell_in_1": ell_in_1,
                                    "ell_in_2": ell_in_2,
                                    "m1": m1,
                                    "m2": m2,
                                    "operation": "project_tensor_product",
                                }
                            )

                            # Clear memory after profiling the first operation
                            torch.cuda.empty_cache()
                            gc.collect()

                            # Reinitialize tensors for the next operation
                            q.fill_(0)  # Reset q
                            k.fill_(0)  # Reset k

                            # Profiling sparse_project_tensor_product and logging to wandb
                            print(
                                f"Profiling sparse_project_tensor_product for J={J}, ell_in_1={ell_in_1}, ell_in_2={ell_in_2}, m1={m1}, m2={m2}"
                            )
                            time_taken, mem_used, _ = profile_operation(
                                sparse_project_tensor_product,
                                q,
                                k,
                                ell_in_1,
                                ell_in_2,
                                ell_out,
                            )

                            # Log to wandb
                            wandb.log(
                                {
                                    "Time_sparse_project_tensor_product": time_taken,
                                    "Memory_sparse_project_tensor_product": mem_used,
                                    "J": J,
                                    "ell_in_1": ell_in_1,
                                    "ell_in_2": ell_in_2,
                                    "m1": m1,
                                    "m2": m2,
                                    "operation": "sparse_project_tensor_product",
                                }
                            )

                            # Free up memory if needed after the second operation
                            torch.cuda.empty_cache()
                            gc.collect()


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="SE3-Hyena", name="sparse_vs_dense_projection_timing")

    # Call your test function
    test_projection()

    # Close the wandb run after the test
    wandb.finish()
