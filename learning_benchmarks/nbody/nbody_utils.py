import torch
import numpy as np


def to_np(x):
    """Convert torch tensor to numpy array"""
    return x.cpu().detach().numpy()


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


def translate_point_cloud(points):
    """
    Randomly translate the point cloud to augment the data.

    Args:
        points: (B, N, 3) tensor of point clouds

    Returns:
        Translated point cloud of shape (B, N, 3)
    """
    batch_size, num_points, _ = points.shape
    translation = torch.empty(batch_size, 1, 3, device=points.device).uniform_(
        -0.2, 0.2
    )  # or another range
    translated_points = points + translation
    return translated_points
