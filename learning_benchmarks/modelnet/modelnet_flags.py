import argparse
import torch
import numpy as np


def get_flags():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="SE3Hyena",
        choices=["SE3Hyena", "DGCNN", "PointNet2"],
        help="Model to use (SE3Hyena, DGCNN, or PointNet2)",
    )

    parser.add_argument(
        "--modelnet_version",
        type=str,
        default="40",
        choices=["10", "40"],
        help="ModelNet version (10 or 40)",
    )

    parser.add_argument(
        "--positional_encoding_dimension",
        type=int,
        default=32,
        help="Positional encoding dimension for SE3Hyena",
    )

    parser.add_argument(
        "--input_dimension_1", type=int, default=128, help="First hidden dimension"
    )

    parser.add_argument(
        "--input_dimension_2", type=int, default=64, help="Second hidden dimension"
    )

    parser.add_argument(
        "--input_dimension_3", type=int, default=32, help="Third hidden dimension"
    )

    parser.add_argument(
        "--num_points",
        type=int,
        default=1024,
        help="Number of points to sample from each object",
    )

    parser.add_argument(
        "--k", type=int, default=20, help="Number of nearest neighbors for DGCNN"
    )

    # Meta-parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")

    parser.add_argument(
        "--grad_clip", type=bool, default=True, help="Use Gradient Clipping"
    )

    # Data
    parser.add_argument(
        "--dataset_path", type=str, default="data", help="Path to dataset directory"
    )

    parser.add_argument(
        "--use_normals", action="store_true", help="Use normal features if available"
    )

    # Logging
    parser.add_argument("--name", type=str, default="modelnet", help="Run name")

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Number of steps between logging key stats",
    )

    parser.add_argument(
        "--print_interval",
        type=int,
        default=50,
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
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use_random_rotation",
        action="store_true",
        help="Apply random rotations as data augmentation",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use",
    )

    # PointNet++ specific parameters
    parser.add_argument(
        "--set_abstraction_ratio_1",
        type=float,
        default=0.5,
        help="First set abstraction layer sampling ratio",
    )

    parser.add_argument(
        "--set_abstraction_ratio_2",
        type=float,
        default=0.25,
        help="Second set abstraction layer sampling ratio",
    )

    parser.add_argument(
        "--set_abstraction_radius_1",
        type=float,
        default=0.2,
        help="First set abstraction layer radius",
    )

    parser.add_argument(
        "--set_abstraction_radius_2",
        type=float,
        default=0.4,
        help="Second set abstraction layer radius",
    )

    parser.add_argument(
        "--feature_dim_1",
        type=int,
        default=128,
        help="First feature dimension for PointNet++",
    )

    parser.add_argument(
        "--feature_dim_2",
        type=int,
        default=256,
        help="Second feature dimension for PointNet++",
    )

    parser.add_argument(
        "--feature_dim_3",
        type=int,
        default=1024,
        help="Third feature dimension for PointNet++",
    )

    # New dropout parameter
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate for the model"
    )

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
