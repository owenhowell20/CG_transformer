import pytest
import torch
import escnn
import escnn.nn
from escnn.group import SO3
from escnn.gspaces import no_base_space
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader


class MockData:
    """Mock data generator for testing"""

    def __init__(self, batch_size=4, num_tokens=8, feature_dim=509):
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


class TensorMockData:
    """Mock data generator for testing"""

    def __init__(self, batch_size=8, num_tokens=8, feature_dim=512):
        self.batch_size = batch_size
        self.seq_length = num_tokens
        self.dim = feature_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define the group SO(3), no base space
        self.so3 = no_base_space(SO3())

        # Type-0 (invariant) and Type-1 (vector) representations
        trivial_repr = self.so3.irrep(0)
        type1_representation = self.so3.irrep(1)

        # Random features
        x = torch.randn(
            batch_size, 3, num_tokens, device=self.device
        )  # Vector features
        f = torch.randn(
            batch_size, feature_dim, num_tokens, device=self.device
        )  # Invariant features

        x = x.permute(0, 2, 1)  # (B, N, 3)
        x = x.reshape(batch_size * num_tokens, 3)  # (BN,3)

        f = f.permute(0, 2, 1)  # (B, N, feature_dim)
        f = f.reshape(batch_size * num_tokens, feature_dim)  # (BN,feature_dim)

        # Field types
        vector_type = escnn.nn.FieldType(self.so3, [type1_representation])
        invariant_type = escnn.nn.FieldType(self.so3, feature_dim * [trivial_repr])

        # Wrap tensors
        vector_features = escnn.nn.GeometricTensor(x, vector_type)
        invariant_features = escnn.nn.GeometricTensor(f, invariant_type)

        self.x = vector_features
        self.f = invariant_features

    def get_data(self):
        """Returns mock x and f tensors."""
        return self.x, self.f

    def random_transform(self):
        g = self.so3.random_element()

        # Apply the transformation to the vector features (x)
        transformed_vector_features = self.x.transform(g)

        # Apply the transformation to the invariant features (f)
        transformed_invariant_features = self.f.transform(g)

        return transformed_vector_features, transformed_invariant_features, g


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


@pytest.fixture
def mock_tensor_data():
    data = TensorMockData()
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


def test_mock_tensor_data(mock_tensor_data):
    x, f = mock_tensor_data  # Extract x and f from the fixture

    assert True
    assert x.tensor.shape[1] == 3, "Coordonates always have dimension 3"

    assert (
        x.tensor.shape[0] == f.tensor.shape[0]
    ), "Batch and Token dimensions must be the same"
