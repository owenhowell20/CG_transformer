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
        "--positional_encoding_dimension",
        type=int,
        default=256,
        help="Postional encoding dim",
    )

    parser.add_argument(
        "--input_dimension_1",
        type=int,
        default=256,
        help="Input dimension",
    )
    parser.add_argument(
        "--input_dimension_2",
        type=int,
        default=128,
        help="Input dimension",
    )
    parser.add_argument(
        "--input_dimension_3",
        type=int,
        default=64,
        help="Input dimension",
    )
    parser.add_argument(
        "--output_dimension",
        type=int,
        default=19,
        help="Input dimension",
    )

    # Meta-parameters
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )  ### must be greater than 1!
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--grad_clip", type=bool, default=True, help="Use Gradient Clipping"
    )

    parser.add_argument("--test_size", type=int, default=1, help="Size of test set")
    parser.add_argument(
        "--val_size", type=int, default=1, help="Size of validation set"
    )

    # Data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/QM9",
        help="Address to saved QM9 dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="homo",
        help="QM9 task: ['homo, 'mu', 'alpha', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']",
    )

    # Logging
    parser.add_argument("--name", type=str, default="qm9", help="Run name")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=25,
        help="Number of steps between logging key stats",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=250,
        help="Number of steps between printing key stats",
    )
    parser.add_argument(
        "--save_dir", type=str, default="models", help="Directory name to save models"
    )
    parser.add_argument(
        "--restore", type=str, default=None, help="Path to model to restore"
    )
    parser.add_argument("--verbose", type=int, default=0)

    # Miscellanea
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )

    # Random seed for both Numpy and Pytorch
    parser.add_argument("--seed", type=int, default=1992)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    if torch.cuda.is_available():
        FLAGS.device = torch.device("cuda:0")
    else:
        FLAGS.device = torch.device("cpu")

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    return FLAGS, UNPARSED_ARGV
