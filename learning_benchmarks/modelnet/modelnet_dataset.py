import os
import sys
import numpy as np
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from torch_geometric.loader import DataLoader


def get_modelnet_dataset(
    root, name="40", num_points=1024, use_normals=False, batch_size=32, num_workers=4
):
    """
    Get ModelNet dataset and dataloaders

    Args:
        root: Path to dataset
        name: "10" for ModelNet10, "40" for ModelNet40
        num_points: Number of points to sample from each mesh
        use_normals: Whether to use normal features
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders

    Returns:
        train_loader: Training data loader
        test_loader: Testing data loader
        num_classes: Number of classes
    """
    # Transformations
    transforms = Compose([SamplePoints(num_points), NormalizeScale()])

    # Create datasets
    train_dataset = ModelNet(root=root, name=name, train=True, transform=transforms)

    test_dataset = ModelNet(root=root, name=name, train=False, transform=transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Get number of classes
    num_classes = 10 if name == "10" else 40

    return train_loader, test_loader, num_classes


# Example usage
if __name__ == "__main__":
    train_loader, test_loader, num_classes = get_modelnet_dataset("data/ModelNet")

    print(f"Number of classes: {num_classes}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")

    # Inspect a batch
    for batch in train_loader:
        print("Batch shape:", batch)
        print("Positions shape:", batch.pos.shape)
        print("Batch indices:", batch.batch.shape)
        print("Labels:", batch.y.shape)
        break
