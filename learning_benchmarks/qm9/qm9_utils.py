import torch
import numpy as np


# Assuming G.pos is a tensor of shape (batch_size, N, 3)
def random_rotation():
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
