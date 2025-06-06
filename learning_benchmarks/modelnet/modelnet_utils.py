import torch
import numpy as np
from src.utils import to_np


def random_rotation_explict():
    """Generate a random 3D rotation matrix"""
    # Generate a random rotation matrix using PyTorch
    theta = torch.rand(1) * 2 * np.pi  # Random angle between 0 and 2π
    axis = torch.randn(3)  # Random axis
    axis = axis / axis.norm()  # Normalize axis

    # Create the rotation matrix using the axis-angle representation (Rodrigues' rotation formula)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    one_minus_cos = 1 - cos_theta

    x, y, z = axis
    rotation_matrix = torch.tensor(
        [
            [
                cos_theta + x**2 * one_minus_cos,
                x * y * one_minus_cos - z * sin_theta,
                x * z * one_minus_cos + y * sin_theta,
            ],
            [
                y * x * one_minus_cos + z * sin_theta,
                cos_theta + y**2 * one_minus_cos,
                y * z * one_minus_cos - x * sin_theta,
            ],
            [
                z * x * one_minus_cos - y * sin_theta,
                z * y * one_minus_cos + x * sin_theta,
                cos_theta + z**2 * one_minus_cos,
            ],
        ]
    )

    return rotation_matrix


def random_rotation():
    """Generate a random 3D rotation matrix using QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1  # Flip sign to ensure det(Q) = 1
    return Q


def rotate_point_cloud(points):
    """
    Randomly rotate the point cloud to augment the data

    Args:
        points: (B, N, 3) tensor of point clouds

    Returns:
        Rotated point cloud of shape (B, N, 3)
    """
    batch_size, num_points, _ = points.shape
    rotated_points = torch.zeros_like(points)

    for i in range(batch_size):
        rotation = random_rotation().to(points.device)
        rotated_points[i] = torch.matmul(points[i], rotation.t())

    return rotated_points


def scale_point_cloud(points, scale_low=0.8, scale_high=1.25):
    """
    Scale the point cloud randomly

    Args:
        points: (B, N, 3) tensor of point clouds
        scale_low: Lower bound of scale
        scale_high: Upper bound of scale

    Returns:
        Scaled point cloud of shape (B, N, 3)
    """
    batch_size = points.shape[0]
    scale = (
        torch.rand(batch_size, 1, 1).to(points.device) * (scale_high - scale_low)
        + scale_low
    )
    return points * scale
