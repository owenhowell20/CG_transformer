import torch
import gc
import wandb
from src.models import Hyena_Operator, SE3HyenaOperator
from escnn.group import SO3
from escnn.gspaces import no_base_space

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

attention_types = ["SE3-Hyena"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Constants
B, D = 4, 64  # Batch size, input dim
harmonic_lengths_dict = {"SE3-Hyena": [1, 2, 3, 4, 5]}


# Track all metrics
sequence_logs = []
time_logs = []
memory_logs = []

for attn_type in attention_types:
    # Track all metrics
    temp_sequence_log = []
    temp_time_log = []
    temp_memory_log = []

    harmonic_lengths = harmonic_lengths_dict[attn_type]

    # WandB init
    wandb.init(project="SE3-Hyena", name="scaling_vs_max_harmonic_" + str(attn_type))

    for max_harmonic in harmonic_lengths:
        N = 215  ### hardcode number of points
        x = torch.randn(B, N, 3, device=device)
        f = torch.randn(B, N, D, device=device)

        print(f"Benchmarking {attn_type} at max harmonic={max_harmonic}...")

        torch.cuda.empty_cache()
        gc.collect()

        if attn_type == "SE3-Hyena":
            so3 = no_base_space(SO3())

            model = SE3HyenaOperator(
                input_dimension=D,
                output_dimension=32,
                group=so3,
                device=device,
                kernel_size=7,
                scalar_attention_type="FFT",
                scalar_attention_kwargs=None,
            )

            model = model.to(device)
        else:
            so3 = no_base_space(SO3())

            model = SE3HyenaOperator(
                input_dimension=D,
                output_dimension=32,
                group=so3,
                device=device,
                kernel_size=7,
                scalar_attention_type="FFT",
                scalar_attention_kwargs=None,
            )

            model = model.to(device)

        # Warmup. Make sure lazy init not effect results
        for _ in range(5):
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
                "max_harmonic_number": max_harmonic,
                "time_ms_total_3_runs": elapsed_time_ms,
                "time_ms_per_run": elapsed_time_ms / 3,
                "peak_memory_gb": peak_memory_gb,
            },
            step=max_harmonic,
        )

    wandb.finish()
    sequence_logs.append(temp_sequence_log)
    time_logs.append(temp_time_log)
    memory_logs.append(temp_memory_log)


wandb.init(project="SE3-Hyena", name="Harmonic Scaling Plots")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

# After all attention type loops are done:
for title, ylogs, ylabel in [
    ("Time vs Max Harmonic", time_logs, "Time per Run (ms)"),
    ("Memory vs Max Harmonic", memory_logs, "Peak Memory (GB)"),
]:
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, attn_type in enumerate(attention_types):
        ax.plot(sequence_logs[i], ylogs[i], marker="o", label=attn_type)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Max Harmonic")
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
