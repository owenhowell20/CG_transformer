import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GraspingDataset(Dataset):
    def __init__(self, root_dir, resolution="pts_4096", transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory containing resolution subfolders.
            resolution (str): One of ['pts_4096', 'pts_2048', 'pts_1024', 'pts_512'].
            transform (callable, optional): Optional transform to apply to each sample.
        """
        assert resolution in [
            "pts_4096",
            "pts_2048",
            "pts_1024",
            "pts_512",
        ], "Invalid resolution"
        self.root_dir = os.path.join(root_dir, resolution)
        self.file_paths = sorted(
            [
                os.path.join(self.root_dir, fname)
                for fname in os.listdir(self.root_dir)
                if fname.endswith(".npz")
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)

        pos = torch.tensor(data["pos"], dtype=torch.float32)  # [N, 3]
        normals = torch.tensor(data["normals"], dtype=torch.float32)  # [N, 3]
        idx = torch.tensor(data["idx"], dtype=torch.long).item()  # int
        quat = torch.tensor(data["quat"], dtype=torch.float32)  # [1, 4]
        rotation = torch.tensor(data["rotation"], dtype=torch.float32)  # [3, 3]
        depth = torch.tensor(
            data["depth_projection"], dtype=torch.float32
        ).item()  # float

        sample = {
            "pos": pos,
            "normals": normals,
            "idx": idx,
            "quat": quat,
            "rotation": rotation,
            "depth": depth,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    data = GraspingDataset(
        root_dir="./fine_data", resolution="pts_4096", transform=None
    )

    print("Number of datapoints:", len(data))

    sample = data[0]  # Load the first sample

    print("pos shape:", sample["pos"].shape)  # Expected: [N, 3]
    print("normals shape:", sample["normals"].shape)  # Expected: [N, 3]
    print("idx:", sample["idx"])  # Expected: int
    print("quat shape:", sample["quat"].shape)  # Expected: [1, 4]
    print("rotation shape:", sample["rotation"].shape)  # Expected: [3, 3]
    print("depth:", sample["depth"])  # Expected: float
