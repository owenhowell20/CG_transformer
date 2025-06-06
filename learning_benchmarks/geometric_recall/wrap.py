import torch
import gc
import wandb

import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)


from flags import get_flags
from train import main, wrap_main
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_types = ["SE3HyperHyena", "SE3Hyena", "Standard"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Constants
lengths = [256, 512, 1024, 2048, 4096]
sequence_lengths_dict = {
    "Standard": lengths,
    "SE3Hyena": lengths,
    "SE3HyperHyena": lengths,
}


# Track all metrics
sequence_logs = []
time_logs = []
memory_logs = []

for N in lengths:
    # Track all metrics
    temp_sequence_log = []
    temp_time_log = []
    temp_memory_log = []

    for attn_type in model_types:
        FLAGS, UNPARSED_ARGV = get_flags()
        FLAGS.num_epochs = 50
        FLAGS.model = attn_type
        print("model:", attn_type)
        FLAGS.sequence_length = N

        wrap_main(FLAGS, UNPARSED_ARGV)
