from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.loader import DataLoader

import pytest
import torch
import escnn
import escnn.nn
from escnn.group import SO3
from escnn.gspaces import no_base_space
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modelnet_se3hyena import ModelNetSE3Hyena
from modelnet_dgcnn import ModelNetDGCNN, ModelNetDGCNNLoss


@pytest.fixture()
def so3_group():
    so3 = no_base_space(SO3())
    return so3


@pytest.fixture
def mock_graph_batch():
    num_classes = 10

    # Mock two graphs with 4 and 3 atoms respectively
    data1 = Data(
        x=torch.randn(4, 11),  # 4 nodes, 11 node features each
        pos=torch.randn(4, 3),  # 3D coordinates
        edge_index=torch.tensor(
            [[0, 1, 2], [1, 2, 3]]
        ),  # 3 edges (nodes 0-1, 1-2, 2-3)
        edge_attr=torch.randn(3, 4),  # 3 edges, 4 edge features
        y=torch.randn(1, num_classes),  # 19 regression targets
    )

    data2 = Data(
        x=torch.randn(5, 11),  # 5 atoms, 11 atom features each
        pos=torch.randn(5, 3),  # 3D coordinates
        edge_index=torch.tensor([[0, 1], [1, 2]]),  # 2 edges (nodes 0-1, 1-2)
        edge_attr=torch.randn(2, 4),  # 2 edges, 4 edge features
        y=torch.randn(1, num_classes),  # 19 regression targets
    )

    batch = Batch.from_data_list([data1, data2])
    return batch


def test_se3hyena_forward(mock_graph_batch):
    batch_size = 2
    num_classes = 10

    model = ModelNetSE3Hyena(num_classes=10)

    G = mock_graph_batch
    out = model(G)

    assert out.shape[0] == batch_size
    assert out.shape[1] == num_classes

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_se3_equivariant(mock_graph_batch, so3_group):
    se3hyena_model_10 = ModelNetSE3Hyena(num_classes=10)
    G = mock_graph_batch
    out = se3hyena_model_10(G)

    ## Unbatch into individual graphs
    graphs = G.to_data_list()

    # Create random rotations (for example)
    num_graphs = len(graphs)
    rotations = []
    for _ in range(num_graphs):
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(Q) < 0:
            Q[:, 0] *= -1  # Flip a column to make det = +1
        rotations.append(Q)
    rotations = torch.stack(rotations)  # (num_graphs, 3, 3)

    # Apply rotation to each graph
    for i, graph in enumerate(graphs):
        graph.pos = (rotations[i] @ graph.pos.T).T

    G_rotated = Batch.from_data_list(graphs)
    out_g = se3hyena_model_10(G_rotated)

    assert out.shape == out_g.shape, "shape mismatch"
    assert torch.allclose(out, out_g, atol=1e-3)


#
# # def test_dgcnn(mock_graph_batch):
# #     batch_size = 2
# #     num_classes = 10
# #
# #     # For ModelNet10
# #     dgcnn_model_10 = ModelNetDGCNN(num_classes=10)
# #
# #     G = mock_graph_batch
# #
# #     out = dgcnn_model_10(G)
# #
# #     assert out.shape[0] == batch_size
# #     assert out.shape[1] == num_classes
# #
# #     assert isinstance(out, torch.Tensor), "Output is not a tensor"
# #     assert not torch.isnan(out).any(), "Output contains NaN values"
