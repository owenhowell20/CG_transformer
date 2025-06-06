import torch
import gc
import wandb
import os
import sys
import numpy as np
import time

# Add the root directory to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from src.equiFFT import equiFFT
from src.projections import HybridProjection, HybridLayerNorm, HybridRelu
from src.models import SE3HyenaOperator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Constants
B, D = 1, 64  # Batch size, input dim

token_space = [8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

sequence_lengths_dict = {
    "Hyena": token_space,
    "FFT": token_space,
}
attention_types = ["FFT", "Hyena"]

# Track all metrics
sequence_logs = []
time_logs = []
memory_logs = []

for attn_type in attention_types:
    # Track all metrics
    temp_sequence_log = []
    temp_time_log = []
    temp_memory_log = []

    sequence_lengths = attention_types[attn_type]

    # WandB init
    wandb.init(
        project="SE3-Hyena", name="FFT_scaling_vs_sequence_length_" + str(attn_type)
    )

    for N in sequence_lengths:
        x = torch.randn(B, N, 3, device=device)
        f = torch.randn(B, N, D, device=device)

        print(f"Benchmarking {attn_type} at N={N}...")

        torch.cuda.empty_cache()
        gc.collect()

        if attn_type == "FFT":
            model = SE3HyenaOperator(
                input_inv_dimension=32,
                input_vector_multiplicity=1,
                output_inv_dimension=32,
                output_vector_multiplicity=1,
                hidden_vector_multiplicity=3,
                hidden_inv_dimension=32,
                group=None,
                device=device,
                vector_attention_type="Standard",
                scalar_attention_type="Standard",
            ).to(device)

        else:
            model = SE3HyenaOperator(
                input_inv_dimension=32,
                input_vector_multiplicity=1,
                output_inv_dimension=32,
                output_vector_multiplicity=1,
                hidden_vector_multiplicity=3,
                hidden_inv_dimension=32,
                group=None,
                device=device,
                vector_attention_type="Hyena",
                scalar_attention_type="Hyena",
            ).to(device)

        # Warmup
        for _ in range(3):
            _ = model(x, f)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(3):
                _ = model(x, f)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        temp_sequence_log.append(N)
        temp_time_log.append(elapsed_time_ms / 3)
        temp_memory_log.append(peak_memory_gb)

        # Log with metadata
        wandb.log(
            {
                "attention_type": attn_type,
                "sequence_length": N,
                "time_ms_total_3_runs": elapsed_time_ms,
                "time_ms_per_run": elapsed_time_ms / 3,
                "peak_memory_MB": peak_memory_gb,
            },
            step=N,
        )

    wandb.finish()
    sequence_logs.append(temp_sequence_log)
    time_logs.append(temp_time_log)
    memory_logs.append(temp_memory_log)


wandb.init(project="SE3-Hyena", name="FFT Attention Scaling Plots")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

# After all attention type loops are done:
for title, ylogs, ylabel in [
    ("Time vs Sequence Length", time_logs, "Time per Run (ms)"),
    ("Memory vs Sequence Length", memory_logs, "Peak Memory (MB)"),
]:
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, attn_type in enumerate(attention_types):
        ax.plot(sequence_logs[i], ylogs[i], marker="o", label=attn_type)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Save to buffer and upload
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)

    wandb.log({title: wandb.Image(image)})

    plt.close(fig)
    buf.close()
