import os
import sys
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from scipy.constants import physical_constants
from torch_geometric.datasets import QM9

dataset = QM9(root="data/QM9")
from torch_geometric.data import Batch


def collate(samples):
    return Batch.from_data_list(samples)


if __name__ == "__main__":
    dataset = QM9(root="data/QM9")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)

    for batch in dataloader:
        print(batch)
        print(batch.x.shape, batch.edge_index.shape)
