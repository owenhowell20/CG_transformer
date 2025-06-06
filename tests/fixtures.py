import pytest
import torch
import escnn
import escnn.nn
from escnn.group import SO3
from escnn.gspaces import no_base_space
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from dgl.data import DGLDataset
from dgl import graph
import pytest
import torch
import dgl
from dgl import DGLGraph
import pytest
import torch
import numpy as np
import os
import pickle
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from learning_benchmarks.nbody.nbody_dataloader import RIDataset


def make_symmetric_edges(edge_list):
    symmetric_edges = edge_list + [(dst, src) for src, dst in edge_list]
    edge_index = torch.tensor(symmetric_edges, dtype=torch.long).t().contiguous()
    return edge_index


class MockData:
    """Mock data generator for testing"""

    def __init__(self, batch_size=4, num_tokens=8, feature_dim=512):
        self.batch_size = batch_size
        self.seq_length = num_tokens
        self.dim = feature_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.x = torch.randn(
            batch_size, num_tokens, 3, device=self.device
        )  # Random graph coordonates (b,3,N)
        self.f = torch.randn(
            batch_size, num_tokens, feature_dim, device=self.device
        )  # Random features (b,d,N)

    def get_data(self):
        """Returns mock x and f tensors."""
        return self.x, self.f


@pytest.fixture
def mock_graph_batch():
    # Mock two molecules (graphs) with 4 and 3 atoms respectively
    data1 = Data(
        x=torch.randn(4, 11),  # 4 atoms, 11 atom features each
        pos=torch.randn(4, 3),  # 3D coordinates
        edge_index=torch.tensor(
            [[0, 1, 2], [1, 2, 3]]
        ),  # 3 edges (nodes 0-1, 1-2, 2-3)
        edge_attr=torch.randn(3, 4),  # 3 edges, 4 edge features
        y=torch.randn(1, 19),  # 19 regression targets
    )

    data2 = Data(
        x=torch.randn(3, 11),  # 3 atoms, 11 atom features each
        pos=torch.randn(3, 3),  # 3D coordinates
        edge_index=torch.tensor([[0, 1], [1, 2]]),  # 2 edges (nodes 0-1, 1-2)
        edge_attr=torch.randn(2, 4),  # 2 edges, 4 edge features
        y=torch.randn(1, 19),  # 19 regression targets
    )

    batch = Batch.from_data_list([data1, data2])
    return batch


@pytest.fixture
def mock_big_graph_batch():
    # Define edges for graph 1 (7 nodes)
    edges1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (1, 5)]
    edge_index1 = make_symmetric_edges(edges1)

    data1 = Data(
        x=torch.randn(7, 11),  # 7 atoms, 11 features each
        pos=torch.randn(7, 3),  # 3D coordinates
        edge_index=edge_index1,
        edge_attr=torch.randn(edge_index1.size(1), 4),  # 2x number of original edges
        y=torch.randn(1, 19),
    )

    # Define edges for graph 2 (10 nodes)
    edges2 = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 0),
        (2, 7),
    ]
    edge_index2 = make_symmetric_edges(edges2)

    data2 = Data(
        x=torch.randn(10, 11),  # 10 atoms, 11 features each
        pos=torch.randn(10, 3),
        edge_index=edge_index2,
        edge_attr=torch.randn(edge_index2.size(1), 4),
        y=torch.randn(1, 19),
    )

    batch = Batch.from_data_list([data1, data2])
    return batch


def permute_batch(batch):
    num_nodes = batch.x.size(0)
    device = batch.x.device

    per_graph_perms = []

    out = Batch()

    new_x = []
    new_pos = []
    new_batch = []
    new_edge_index = []
    new_edge_attr = []
    new_y = []

    offset = 0  # To shift edge indices graph by graph

    for graph_idx in batch.batch.unique(sorted=True):
        node_mask = batch.batch == graph_idx
        nodes = node_mask.nonzero(as_tuple=False).view(-1)

        n_nodes = nodes.size(0)
        perm = torch.randperm(n_nodes, device=device)

        per_graph_perms.append(perm)  # <<<<<<<<<<< RECORD PER-GRAPH PERM

        # Permute nodes of this graph
        new_x.append(batch.x[nodes][perm])
        new_pos.append(batch.pos[nodes][perm])
        new_batch.append(
            torch.full((n_nodes,), graph_idx, dtype=torch.long, device=device)
        )

        # Create inverse permutation
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n_nodes, device=device)

        # Adjust edge_index
        edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
        edges = batch.edge_index[:, edge_mask]

        local_row = (edges[0] - nodes.min()).clone()
        local_col = (edges[1] - nodes.min()).clone()

        new_row = inv_perm[local_row] + offset
        new_col = inv_perm[local_col] + offset
        new_edge_index.append(torch.stack([new_row, new_col], dim=0))

        if batch.edge_attr is not None:
            new_edge_attr.append(batch.edge_attr[edge_mask])

        offset += n_nodes  # Increase offset for next graph

        if hasattr(batch, "y"):
            new_y.append(batch.y[graph_idx].unsqueeze(0))

    # Concatenate results
    out.x = torch.cat(new_x, dim=0)
    out.pos = torch.cat(new_pos, dim=0)
    out.batch = torch.cat(new_batch, dim=0)
    out.edge_index = torch.cat(new_edge_index, dim=1)

    if batch.edge_attr is not None:
        out.edge_attr = torch.cat(new_edge_attr, dim=0)

    if hasattr(batch, "y"):
        out.y = torch.cat(new_y, dim=0)

    return out, per_graph_perms


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


@pytest.fixture
def mock_data():
    data = MockData()
    return data.get_data()


@pytest.fixture()
def so3_group():
    so3 = no_base_space(SO3())
    return so3


def test_mock_data(mock_data):
    ### x ~ (b,3,N) , f ~ (b,d,N)
    x, f = mock_data  # Extract x and f from the fixture
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert x.device == device, "Device mismatch"
    assert f.device == device, "Device mismatch"
    assert (
        x.shape[0] == f.shape[0]
    ), "Coordonates and features must have same batch dimension"
    assert (
        x.shape[1] == f.shape[1]
    ), "Coordonates and features must have same token dimension"


@pytest.fixture
def mock_dgl_graph_with_edges():
    # Create a mock DGL graph with nodes and edges

    num_nodes = 10  ### 10 nodes,
    num_edges = 40  ### 40 edges
    G = dgl.graph(
        (
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ),
        num_nodes=num_nodes,
    )

    # Add node features
    G.ndata["x"] = torch.randn(num_nodes, 3)  # Example node feature 'x': vector
    G.ndata["c"] = torch.randn(num_nodes, 1)  # Example node feature 'c': invariant
    G.ndata["v"] = torch.randn(num_nodes, 3)  # Another example node feature 'v': vector
    G.ndata["d"] = torch.randn(num_nodes, 3)  # Example node feature 'd'

    # Add edge features
    G.edata["w"] = torch.randn(num_edges, 1)  # Example edge feature 'w'
    G.edata["d"] = torch.randn(num_edges, 3)  # Example edge feature 'd'

    # Calculate and add the relative distance ('r') as an edge feature
    src, dst = G.edges()
    G.edata["r"] = (
        (G.ndata["x"][dst] - G.ndata["x"][src]).norm(dim=1).unsqueeze(-1)
    )  # Compute relative distance

    return G


@pytest.fixture
def mock_ridataset():
    """Create a RIDataset using the existing dataset file."""
    FLAGS = SimpleNamespace(
        ri_data_type="charged",
        data_str="5_new",
        ri_data="learning_benchmarks/nbody/nbody_data_generation",  # Fixed path
        ri_burn_in=0,
        ri_start_at="zero",
        ri_delta_t=1,
        graph_type="fully_connected",
    )

    dataset = RIDataset(FLAGS, split="train")
    yield dataset
