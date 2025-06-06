import pytest
import torch
from fixtures import mock_data, so3_group
import escnn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.models import SE3HyenaOperator


def test_full_model_equivariant(mock_data, so3_group):
    batch_size, num_tokens = 2, 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dimension = 512
    input_vector_multiplicity = 3

    x = torch.randn(
        batch_size, num_tokens, 3 * input_vector_multiplicity, device=device
    )
    f = torch.randn(batch_size, num_tokens, input_dimension, device=device)

    output_dimension = 256
    output_vector_multiplicity = 3

    attention_types = ["FFT"]
    for vector_attention_type in attention_types:
        model = SE3HyenaOperator(
            input_inv_dimension=input_dimension,
            input_vector_multiplicity=input_vector_multiplicity,
            hidden_inv_dimension=2 * input_dimension,
            hidden_vector_multiplicity=5,
            output_inv_dimension=output_dimension,
            output_vector_multiplicity=output_vector_multiplicity,
            vector_attention_type=vector_attention_type,
        ).to(device)

        ### apply model
        x_proj, f_proj = model(x, f)  ### output (b,N,3),(b,N,d)

        # Type-0 (invariant) and Type-1 (vector) representations
        trivial_repr = so3_group.irrep(0)
        type1_representation = so3_group.irrep(1)

        x = x.reshape(batch_size * num_tokens, 3 * input_vector_multiplicity)  # (BN,3)
        f = f.reshape(batch_size * num_tokens, input_dimension)  # (BN,feature_dim)

        x_proj = x_proj.reshape(batch_size * num_tokens, 3 * output_vector_multiplicity)
        f_proj = f_proj.reshape(batch_size * num_tokens, output_dimension)

        # Field types
        vector_type = escnn.nn.FieldType(
            so3_group, input_vector_multiplicity * [type1_representation]
        )
        invariant_type = escnn.nn.FieldType(so3_group, input_dimension * [trivial_repr])
        output_invariant_type = escnn.nn.FieldType(
            so3_group, output_dimension * [trivial_repr]
        )
        output_vector_type = escnn.nn.FieldType(
            so3_group, output_vector_multiplicity * [type1_representation]
        )

        # Wrap tensors
        x = escnn.nn.GeometricTensor(x, vector_type)
        f = escnn.nn.GeometricTensor(f, invariant_type)

        x_proj = escnn.nn.GeometricTensor(x_proj, output_vector_type)
        f_proj = escnn.nn.GeometricTensor(f_proj, output_invariant_type)

        # apply G transformation
        g = so3_group.fibergroup.sample()

        # Apply the transformation to the vector features (x)
        x_g = x.transform(g)
        x_proj_g = x_proj.transform(g)

        # Apply the transformation to the invariant features (f)
        f_g = f.transform(g)
        f_proj_g = f_proj.transform(g)

        ### apply model to transform
        x_g = x_g.tensor.reshape(batch_size, num_tokens, 3 * input_vector_multiplicity)
        f_g = f_g.tensor.reshape(batch_size, num_tokens, input_dimension)
        x_g_proj, f_g_proj = model(x_g, f_g)

        ### reshape model outputs
        x_proj_g = x_proj_g.tensor.reshape(
            batch_size, num_tokens, 3 * output_vector_multiplicity
        )
        f_proj_g = f_proj_g.tensor.reshape(batch_size, num_tokens, output_dimension)

        assert x_g_proj.shape == x_proj_g.shape, "Shape mismatch"
        assert f_g_proj.shape == f_proj_g.shape, "Shape Mismatch"

        assert torch.allclose(
            f_proj_g, f_g_proj, atol=1e-5
        ), "Projection model not equivariant"
        assert torch.allclose(
            x_proj_g, x_g_proj, atol=1e-5
        ), "Projection model not equivariant"


def test_full_model(mock_data, so3_group):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_inv_dimension = 512
    input_vector_multiplicity = 3

    hidden_inv_dimension = 64
    hidden_vector_multiplicity = 2

    output_vector_multiplicity = 3
    output_dim = 128

    batch_size = 4
    num_tokens = 10

    attention_types = ["FFT"]
    for vector_attention_type in attention_types:
        x = torch.randn(
            batch_size, num_tokens, 3 * input_vector_multiplicity, device=device
        )
        f = torch.randn(batch_size, num_tokens, input_inv_dimension, device=device)

        assert x.device == f.device
        assert x.shape[0] == f.shape[0], "batch dimensions should be the same"
        assert x.shape[1] == f.shape[1], "token dimensions should be the same"
        assert (
            x.shape[2] == 3 * input_vector_multiplicity
        ), "Coordonates must be dimension three"
        assert f.shape[2] == input_inv_dimension, "Output dimension mistmatch"

        model = SE3HyenaOperator(
            input_inv_dimension=input_inv_dimension,
            input_vector_multiplicity=input_vector_multiplicity,
            hidden_inv_dimension=hidden_inv_dimension,
            hidden_vector_multiplicity=hidden_vector_multiplicity,
            output_inv_dimension=output_dim,
            output_vector_multiplicity=output_vector_multiplicity,
            vector_attention_type=vector_attention_type,
        ).to(device)

        x, f = model(x, f)

        assert x.shape[0] == f.shape[0], "batch dimensions should be the same"
        assert x.shape[1] == f.shape[1], "token dimensions should be the same"
        assert (
            x.shape[2] == 3 * output_vector_multiplicity
        ), "Coordonates must be dimension three"
        assert f.shape[2] == output_dim, "Output dimension mistmatch"

        assert isinstance(x, torch.Tensor), "Output is not a tensor"
        assert not torch.isnan(x).any(), "Output contains NaN values"

        assert isinstance(f, torch.Tensor), "Output is not a tensor"
        assert not torch.isnan(f).any(), "Output contains NaN values"
