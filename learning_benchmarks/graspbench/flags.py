import argparse
import torch
import numpy as np


def get_flags():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument(
        "--model", type=str, default="SE3Hyena", help="String name of model"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,  # resolution in [ "pts_4096", "pts_2048",  "pts_1024", "pts_512", ]
        help="Number of timesteps",
    )

    # Model parameters
    parser.add_argument(
        "--task", type=str, default="Normals", help="Task to train on: Normals or Grasp"
    )

    parser.add_argument(
        "--use_normals",
        type=bool,
        default=True,
        help="Option to use normals on the grasp task",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./fine_data",
        help="Directory name to save models",
    )

    parser.add_argument(
        "--positional_encoding_dimension",
        type=int,
        default=256,
        help="Positional encoding dim",
    )
    parser.add_argument(
        "--input_dimension_1",
        type=int,
        default=128,
        help="Input dimension",
    )
    parser.add_argument(
        "--input_dimension_2",
        type=int,
        default=64,
        help="Input dimension",
    )
    parser.add_argument(
        "--input_dimension_3",
        type=int,
        default=32,
        help="Input dimension",
    )

    # Meta-parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--loss_type", type=str, default="cosine", help="Loss function")

    # Logging
    parser.add_argument("--name", type=str, default="graspbench", help="Run name")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="Number of steps between logging key stats",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=500,
        help="Number of steps between printing key stats",
    )
    parser.add_argument(
        "--save_dir", type=str, default="models", help="Directory name to save models"
    )
    parser.add_argument(
        "--restore", type=str, default=None, help="Path to model to restore"
    )
    parser.add_argument("--verbose", type=int, default=0)

    # Miscellaneous
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Exit after 10 steps for profiling"
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=1992)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Append model name to run name
    FLAGS.name = f"{FLAGS.name}_{FLAGS.model}"

    # Set random seeds
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set device
    if torch.cuda.is_available():
        FLAGS.device = torch.device("cuda:0")
    else:
        FLAGS.device = torch.device("cpu")

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    return FLAGS, UNPARSED_ARGV
