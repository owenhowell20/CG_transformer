import pytest
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

import pytest
import torch
import escnn
import escnn.nn
from escnn.group import SO3
from escnn.gspaces import no_base_space


@pytest.fixture()
def mock_so3_group():
    so3 = no_base_space(SO3())
    return so3


@pytest.fixture
def mock_qm9_batch():
    # Mock two molecules (graphs) with 4 and 3 atoms respectively
    data1 = Data(
        x=torch.randn(4, 11),  # 4 atoms, 11 atom features each
        pos=torch.randn(4, 3),  # 3D coordinates
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),  # example edges
        edge_attr=torch.randn(3, 4),  # 3 edges, 4 edge features
        y=torch.randn(1, 19),  # 19 regression targets
    )

    data2 = Data(
        x=torch.randn(3, 11),  ### 3 atoms, 11 atom features each
        pos=torch.randn(3, 3),  ### 3D coordonates
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 4),
        y=torch.randn(1, 19),
    )

    batch = Batch.from_data_list([data1, data2])
    return batch


@pytest.fixture
def mock_qm9_dataloader():
    # Create two mock molecules
    data1 = Data(
        x=torch.randn(4, 11),
        pos=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, 4),
        y=torch.randn(1, 19),
    )

    data2 = Data(
        x=torch.randn(3, 11),
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 4),
        y=torch.randn(1, 19),
    )

    dataset = [data1, data2]

    # Use PyG's collate function for graph batches
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Batch.from_data_list)
    return dataloader


def test_qm9_batch_shape(mock_qm9_batch):
    batch = mock_qm9_batch
    assert batch.x.shape[1] == 11  # Atom features
    assert batch.pos.shape[1] == 3  # Coordinates
    assert batch.y.shape == (2, 19)  # 2 molecules, 19 targets each
    assert batch.batch.shape[0] == batch.x.shape[0]  # Mapping from node to graph
