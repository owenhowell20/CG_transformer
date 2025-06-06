import torch
import gc
import wandb
import numpy as np
import matplotlib.pyplot as plt
from src.tensor_products import sparse_project_tensor_product, project_tensor_product

# Init wandb
wandb.init(project="SE3-Hynea", name="sparse_project_tensor_product_timing")

# Parameter sweeps
Nvals = [8, 16, 32, 64, 128, 256, 512]
j1_vals = [0, 1, 2, 3, 4, 5, 6]
j2_vals = [0, 1, 2, 3, 4, 5, 6]
J_vals = [0, 1, 2, 3, 4, 5, 6]
batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store results
results = []

for N in Nvals:
    for j1 in j1_vals:
        for j2 in j2_vals:
            for J in J_vals:
                if abs(j1 - j2) <= J <= j1 + j2:
                    q = torch.randn(batch_size, N, 2 * j1 + 1, device=device)
                    k = torch.randn(batch_size, N, 2 * j2 + 1, device=device)

                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    _ = project_tensor_product(q, k, j1, j2, J)
                    end.record()

                    torch.cuda.synchronize()
                    elapsed_time_ms = start.elapsed_time(end)

                    results.append((N, j1, j2, J, elapsed_time_ms / 1000))

                    # Log raw data
                    wandb.log(
                        {
                            "runtime (s)": elapsed_time_ms / 1000,
                            "N": N,
                            "j1": j1,
                            "j2": j2,
                            "J": J,
                        }
                    )

                    gc.collect()

# Convert to numpy array
results_np = np.array(
    results, dtype=[("N", int), ("j1", int), ("j2", int), ("J", int), ("time", float)]
)

# Create plots
for fixed_j in [0, 1, 2, 3, 4, 5, 6]:
    mask = (
        (results_np["j1"] == fixed_j)
        & (results_np["j2"] == fixed_j)
        & (results_np["J"] == fixed_j)
    )
    subset = results_np[mask]
    if len(subset) > 0:
        plt.figure()
        plt.plot(subset["N"], subset["time"], marker="o")
        plt.xlabel("N")
        plt.ylabel("Runtime (s)")
        plt.title(f"Runtime vs N (j1=j2=J={fixed_j})")
        plt.grid(True)

        # Save to file and log
        filename = f"runtime_vs_N_j{fixed_j}.png"
        plt.savefig(filename)
        wandb.log({f"plot_j{fixed_j}": wandb.Image(filename)})
        plt.close()

wandb.finish()
