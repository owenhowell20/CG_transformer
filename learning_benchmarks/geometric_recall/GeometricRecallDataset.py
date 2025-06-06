import torch
from torch.utils.data import Dataset

import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from scipy.spatial.transform import Rotation as R


def sampleSO3():
    """
    Samples a random rotation matrix from the special orthogonal group SO(3).
    Returns:
        A (3, 3) torch tensor representing a random rotation matrix.
    """
    # Generate a random rotation in SO(3) using scipy
    random_rotation = R.random()  # Generates a random rotation in SO(3)

    # Convert to torch tensor (float64 for precision, can change to float32 if needed)
    rotation_matrix = torch.tensor(random_rotation.as_matrix(), dtype=torch.float64)

    # Return as a float32 tensor if you need precision reduced (for faster computation)
    return rotation_matrix.to(torch.float32)


class VectorEquivariantRecall(Dataset):
    def __init__(
        self,
        num_sequences=2000,
        sequence_length=28,
        vocab_size=4,
        random=True,
        device="cpu",
    ):
        self.sequence_length = sequence_length
        self.random = random
        self.num_sequences = num_sequences
        self.vocab_size = vocab_size
        self.device = device

        # Create a fixed vocabulary of key-value pairs
        self.keys = torch.randn(self.vocab_size, 3)  # Random vectors for keys
        self.values = self.keys + torch.randn(
            self.vocab_size, 3
        )  # Values are slightly different for diversity

        # Normalize keys and values
        self.keys = self.keys / self.keys.norm(dim=1, keepdim=True)
        self.values = self.values / self.values.norm(dim=1, keepdim=True)

        # Generate SO(3) rotation matrices for each key-value pair
        self.rotations = torch.stack(
            [sampleSO3() for _ in range(self.vocab_size)]
        )  # Shape: (vocab_size, 3, 3)

        # Now, let's generate sequences of vectors from the fixed vocabulary
        self.vectors = torch.randint(
            0, self.vocab_size, (self.num_sequences, self.sequence_length)
        )  # Indices from vocab

        # Ensure valid indices for self.rotations and self.keys
        assert torch.all(
            self.vectors < self.vocab_size
        ), "Index out of bounds in self.vectors"

        # Apply rotations to the vectors in each sequence
        self.rotated_vectors = torch.stack(
            [
                torch.stack(
                    [torch.matmul(self.rotations[v], self.keys[v]) for v in seq]
                )
                for seq in self.vectors
            ]
        )  # Shape: (num_sequences, sequence_length, 3)

        # Random indices for the questions (query) and corresponding answers
        random_indices = torch.randint(0, self.sequence_length, (self.num_sequences,))

        self.questions = self.keys[
            self.vectors[torch.arange(self.num_sequences), random_indices]
        ]
        self.answers = self.rotated_vectors[
            torch.arange(self.num_sequences), random_indices
        ]

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        seq_indices = self.vectors[idx]
        keys = self.keys[seq_indices]  # Shape: (sequence_length, 3)
        values = self.values[seq_indices]  # Shape: (sequence_length, 3)

        rotated_keys = torch.stack(
            [
                torch.matmul(self.rotations[k], keys[i])
                for i, k in enumerate(seq_indices)
            ]
        )
        rotated_values = torch.stack(
            [
                torch.matmul(self.rotations[k], values[i])
                for i, k in enumerate(seq_indices)
            ]
        )

        question = self.questions[idx]
        answer = self.answers[idx]

        return {
            "key": keys,
            "value": values,
            "query": rotated_keys,
            "question": question,
            "answer": answer,
        }


class RotationEquivariantRecall(Dataset):
    def __init__(self, sequence_length=2048, random=True):
        self.size = sequence_length
        self.random = random

        # Sample two sets of SO(3) rotations
        self.rotations = torch.stack([sampleSO3() for _ in range(self.size)])  # R
        self.rots = torch.stack([sampleSO3() for _ in range(self.size)])  # Q

        # Compute QR (matrix multiplication of each pair)
        self.rot_rotations = torch.matmul(self.rots, self.rotations)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        R = self.rotations[idx]
        Q = self.rots[idx]
        QR = self.rot_rotations[idx]

        return {"key": R, "transform": Q, "value": QR}


if __name__ == "__main__":
    vector_dataset = VectorEquivariantRecall(sequence_length=2048, random=True)

    # Fetch an example
    example = vector_dataset[0]
    print(example)

    rotation_dataset = RotationEquivariantRecall(sequence_length=2048, random=True)

    # Fetch an example
    example = vector_dataset[0]
    print(example)
