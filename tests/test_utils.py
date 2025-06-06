import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def sampleSO3():
    """
    Samples a random rotation matrix from the special orthogonal group SO(3).
    Returns:
        A (3, 3) numpy array representing a random rotation matrix.
    """

    random_rotation = R.random()
    return torch.tensor(random_rotation.as_matrix(), dtype=torch.float64)


if __name__ == "__main__":
    # Example usage
    rotation_matrix = sampleSO3()
    print("Random Rotation Matrix from SO(3):")
    print(rotation_matrix)
