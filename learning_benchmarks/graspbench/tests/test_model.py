import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import wandb
import escnn

from escnn.group import SO3
from escnn.gspaces import no_base_space

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from models import (
    SE3HyenaNormal,
    StandardNormal,
    NormalDGCNN,
    SE3HyperHyenaNormal,
    StandardGrasp,
)

from train import train_epoch, run_test_epoch
from flags import get_flags
from dataset import GraspingDataset


def test_dgcnn_model():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    b = 4
    N = 10

    pos = torch.randn(b, N, 3, device=device)
    normals = torch.randn(b, N, 3, device=device)

    model = NormalDGCNN(k=5, emb_dims=20, dropout=0.5).to(device)

    out = model(pos)
    assert out.shape == normals.shape, "Output dimension mistmatch"

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_baseline_model():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    b = 4
    N = 10

    pos = torch.randn(b, N, 3, device=device)
    normals = torch.randn(b, N, 3, device=device)

    model = StandardNormal(
        sequence_length=N,  # Max number of points
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
        positional_encoding_type="pos_only",  ### pos_only, none
    ).to(device)

    out = model(pos)
    assert out.shape == normals.shape, "Output dimension mistmatch"

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_bespoke_model():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    b = 4
    N = 10

    pos = torch.randn(b, N, 3, device=device)
    normals = torch.randn(b, N, 3, device=device)

    model = SE3HyperHyenaNormal(
        sequence_length=N,  # Max number of points
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
    ).to(device)

    out = model(pos)
    assert out.shape == normals.shape, "Output dimension mistmatch"

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_bespoke_model_equivariant():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    batch_size = 4
    num_tokens = 10

    pos = torch.randn(batch_size, num_tokens, 3, device=device)
    normals = torch.randn(batch_size, num_tokens, 3, device=device)

    model = SE3HyperHyenaNormal(
        sequence_length=num_tokens,  # Max number of points
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
        positional_encoding_type="pos_only",  ### pos_only, none
        kernel_size=3,
        scalar_attention_type="Standard",
    ).to(device)

    ### apply model
    out = model(pos)
    so3_group = no_base_space(SO3())

    # Type-1 (vector) representations
    type1_representation = so3_group.irrep(1)

    pos = pos.reshape(batch_size * num_tokens, 3)  # (BN,3)
    normals = normals.reshape(batch_size * num_tokens, 3)  # (BN,3 )
    out = out.reshape(batch_size * num_tokens, 3)

    # Field types
    vector_type = escnn.nn.FieldType(so3_group, [type1_representation])

    ### Wrap tensors
    pos = escnn.nn.GeometricTensor(pos, vector_type)
    normals = escnn.nn.GeometricTensor(normals, vector_type)
    out = escnn.nn.GeometricTensor(out, vector_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation to the vector features (x)
    pos_g = pos.transform(g)
    normals_g = normals.transform(g)
    out_g = out.transform(g)

    ### apply model to transform
    pos_g = pos_g.tensor.reshape(batch_size, num_tokens, 3)
    normals_g = normals_g.tensor.reshape(batch_size, num_tokens, 3)
    out_g = out_g.tensor.reshape(batch_size, num_tokens, 3)
    g_out = model(pos_g)

    assert g_out.shape == out_g.shape, "Shape mismatch"
    assert torch.allclose(out_g, g_out, atol=1e-5), "Projection model not equivariant"


#


def test_model():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    b = 4
    N = 10

    pos = torch.randn(b, N, 3, device=device)
    normals = torch.randn(b, N, 3, device=device)

    model = SE3HyenaNormal(
        sequence_length=N,  # Max number of points
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
        positional_encoding_type="pos_only",  ### pos_only, none
    ).to(device)

    out = model(pos)
    assert out.shape == normals.shape, "Output dimension mistmatch"

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_full_model_equivariant():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    batch_size = 4
    num_tokens = 10

    pos = torch.randn(batch_size, num_tokens, 3, device=device)
    normals = torch.randn(batch_size, num_tokens, 3, device=device)

    model = SE3HyenaNormal(
        sequence_length=num_tokens,  # Max number of points
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
        positional_encoding_type="pos_only",  ### pos_only, none
    ).to(device)

    ### apply model
    out = model(pos)
    so3_group = no_base_space(SO3())

    # Type-1 (vector) representations
    type1_representation = so3_group.irrep(1)

    pos = pos.reshape(batch_size * num_tokens, 3)  # (BN,3)
    normals = normals.reshape(batch_size * num_tokens, 3)  # (BN,3 )
    out = out.reshape(batch_size * num_tokens, 3)

    # Field types
    vector_type = escnn.nn.FieldType(so3_group, [type1_representation])

    ### Wrap tensors
    pos = escnn.nn.GeometricTensor(pos, vector_type)
    normals = escnn.nn.GeometricTensor(normals, vector_type)
    out = escnn.nn.GeometricTensor(out, vector_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation to the vector features (x)
    pos_g = pos.transform(g)
    normals_g = normals.transform(g)
    out_g = out.transform(g)

    ### apply model to transform
    pos_g = pos_g.tensor.reshape(batch_size, num_tokens, 3)
    normals_g = normals_g.tensor.reshape(batch_size, num_tokens, 3)
    out_g = out_g.tensor.reshape(batch_size, num_tokens, 3)
    g_out = model(pos_g)

    assert g_out.shape == out_g.shape, "Shape mismatch"
    assert torch.allclose(out_g, g_out, atol=1e-5), "Projection model not equivariant"
