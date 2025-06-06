import pytest
import torch
from fixtures import mock_data, so3_group
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.projections import (
    TensorProductLayer,
    LinearProjection,
    NormActivationLayer,
    NormActivationLayer,
)
import escnn


def test_tensor_product_layer():
    B = 2
    N = 10

    input_inv_dimension = 50
    input_vector_multiplicty = 5

    output_inv_dimension = 50
    output_vector_multiplicty = 5

    layer = TensorProductLayer(
        input_inv_dimension=input_inv_dimension,
        input_vector_multiplicity=input_vector_multiplicty,
        output_inv_dimension=output_inv_dimension,
        output_vector_multiplicity=output_vector_multiplicty,
    )

    scalars = torch.randn(B, N, input_inv_dimension)
    vectors = torch.randn(B, N, 3 * input_vector_multiplicty)

    vector_out, scalar_out = layer(vectors, scalars)

    assert scalar_out.shape == (
        B,
        N,
        output_inv_dimension,
    ), "Scalar output shape mismatch"
    assert vector_out.shape == (
        B,
        N,
        3 * output_vector_multiplicty,
    ), "Vector output shape mismatch"

    ### Check values
    assert torch.isfinite(scalar_out).all(), "Scalar output contains NaNs or Infs"
    assert torch.isfinite(vector_out).all(), "Vector output contains NaNs or Infs"


def test_linear_layer():
    B = 2
    N = 10

    input_inv_dimension = 50
    input_vector_multiplicty = 5

    output_inv_dimension = 50
    output_vector_multiplicty = 5

    layer = LinearProjection(
        input_inv_dimension=input_inv_dimension,
        input_vector_multiplicity=input_vector_multiplicty,
        output_inv_dimension=output_inv_dimension,
        output_vector_multiplicity=output_vector_multiplicty,
    )

    scalars = torch.randn(B, N, input_inv_dimension)
    vectors = torch.randn(B, N, 3 * input_vector_multiplicty)

    vector_out, scalar_out = layer(vectors, scalars)

    ### Check shapes
    assert scalar_out.shape == (
        B,
        N,
        output_inv_dimension,
    ), "Scalar output shape mismatch"
    assert vector_out.shape == (
        B,
        N,
        3 * output_vector_multiplicty,
    ), "Vector output shape mismatch"

    ### Check values
    assert torch.isfinite(scalar_out).all(), "Scalar output contains NaNs or Infs"
    assert torch.isfinite(vector_out).all(), "Vector output contains NaNs or Infs"


def test_layer_norm():
    B = 2
    N = 10

    input_inv_dimension = 50
    input_vector_multiplicty = 5

    output_inv_dimension = input_inv_dimension
    output_vector_multiplicty = input_vector_multiplicty

    layer = NormActivationLayer(
        input_inv_dimension=input_inv_dimension,
        input_vector_multiplicity=input_vector_multiplicty,
    )

    scalars = torch.randn(B, N, input_inv_dimension)
    vectors = torch.randn(B, N, 3 * input_vector_multiplicty)

    vector_out, scalar_out = layer(vectors, scalars)

    ### Check shapes
    assert scalar_out.shape == (
        B,
        N,
        output_inv_dimension,
    ), "Scalar output shape mismatch"
    assert vector_out.shape == (
        B,
        N,
        3 * output_vector_multiplicty,
    ), "Vector output shape mismatch"

    ### Check values
    assert torch.isfinite(scalar_out).all(), "Scalar output contains NaNs or Infs"
    assert torch.isfinite(vector_out).all(), "Vector output contains NaNs or Infs"


def test_equivariance_linear_projection(mock_data, so3_group):
    batch_size = 2
    num_tokens = 10

    input_inv_dimension = 50
    input_vector_multiplicty = 5

    output_inv_dimension = 50
    output_vector_multiplicty = 5

    layer = LinearProjection(
        input_inv_dimension=input_inv_dimension,
        input_vector_multiplicity=input_vector_multiplicty,
        output_inv_dimension=output_inv_dimension,
        output_vector_multiplicity=output_vector_multiplicty,
    )

    f = torch.randn(batch_size, num_tokens, input_inv_dimension)
    x = torch.randn(batch_size, num_tokens, 3 * input_vector_multiplicty)

    ### apply projection
    x_proj, f_proj = layer(x, f)

    # Type-0 (invariant) and Type-1 (vector) representations
    trivial_repr = so3_group.irrep(0)
    type1_representation = so3_group.irrep(1)

    x = x.reshape(batch_size * num_tokens, x.shape[-1])
    f = f.reshape(batch_size * num_tokens, f.shape[-1])

    # Field types
    vector_type = escnn.nn.FieldType(
        so3_group, input_vector_multiplicty * [type1_representation]
    )
    invariant_type = escnn.nn.FieldType(so3_group, input_inv_dimension * [trivial_repr])

    output_vector_type = escnn.nn.FieldType(
        so3_group, output_vector_multiplicty * [type1_representation]
    )
    output_invariant_type = escnn.nn.FieldType(
        so3_group, output_inv_dimension * [trivial_repr]
    )

    # Wrap tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    f = escnn.nn.GeometricTensor(f, invariant_type)

    x_proj = x_proj.reshape(batch_size * num_tokens, x_proj.shape[-1])
    f_proj = f_proj.reshape(batch_size * num_tokens, f_proj.shape[-1])
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
    x_g = x_g.tensor.reshape(batch_size, num_tokens, x_g.shape[-1])
    f_g = f_g.tensor.reshape(batch_size, num_tokens, f_g.shape[-1])
    x_g_proj, f_g_proj = layer(x_g, f_g)

    x_proj_g = x_proj_g.tensor.reshape(batch_size, num_tokens, x_proj_g.shape[-1])
    f_proj_g = f_proj_g.tensor.reshape(batch_size, num_tokens, f_proj_g.shape[-1])

    assert x_g_proj.shape == x_proj_g.shape, "Shape mismatch"
    assert f_g_proj.shape == f_proj_g.shape, "Shape Mismatch"

    assert torch.allclose(
        f_proj_g, f_g_proj, atol=1e-5
    ), "Projection model not equivariant"
    assert torch.allclose(
        x_proj_g, x_g_proj, atol=1e-5
    ), "Projection model not equivariant"


def test_equivariance_layer_norm(mock_data, so3_group):
    batch_size = 2
    num_tokens = 10

    input_inv_dimension = 50
    input_vector_multiplicty = 5

    output_inv_dimension = input_inv_dimension
    output_vector_multiplicty = input_vector_multiplicty

    layer = NormActivationLayer(
        input_inv_dimension=input_inv_dimension,
        input_vector_multiplicity=input_vector_multiplicty,
    )

    f = torch.randn(batch_size, num_tokens, input_inv_dimension)
    x = torch.randn(batch_size, num_tokens, 3 * input_vector_multiplicty)

    ### apply projection
    x_proj, f_proj = layer(x, f)

    # Type-0 (invariant) and Type-1 (vector) representations
    trivial_repr = so3_group.irrep(0)
    type1_representation = so3_group.irrep(1)

    x = x.reshape(batch_size * num_tokens, x.shape[-1])
    f = f.reshape(batch_size * num_tokens, f.shape[-1])

    # Field types
    vector_type = escnn.nn.FieldType(
        so3_group, input_vector_multiplicty * [type1_representation]
    )
    invariant_type = escnn.nn.FieldType(so3_group, input_inv_dimension * [trivial_repr])

    output_vector_type = escnn.nn.FieldType(
        so3_group, output_vector_multiplicty * [type1_representation]
    )
    output_invariant_type = escnn.nn.FieldType(
        so3_group, output_inv_dimension * [trivial_repr]
    )

    # Wrap tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    f = escnn.nn.GeometricTensor(f, invariant_type)

    x_proj = x_proj.reshape(batch_size * num_tokens, x_proj.shape[-1])
    f_proj = f_proj.reshape(batch_size * num_tokens, f_proj.shape[-1])
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
    x_g = x_g.tensor.reshape(batch_size, num_tokens, x_g.shape[-1])
    f_g = f_g.tensor.reshape(batch_size, num_tokens, f_g.shape[-1])
    x_g_proj, f_g_proj = layer(x_g, f_g)

    x_proj_g = x_proj_g.tensor.reshape(batch_size, num_tokens, x_proj_g.shape[-1])
    f_proj_g = f_proj_g.tensor.reshape(batch_size, num_tokens, f_proj_g.shape[-1])

    assert x_g_proj.shape == x_proj_g.shape, "Shape mismatch"
    assert f_g_proj.shape == f_proj_g.shape, "Shape Mismatch"

    assert torch.allclose(
        f_proj_g, f_g_proj, atol=1e-5
    ), "Projection model not equivariant"
    assert torch.allclose(
        x_proj_g, x_g_proj, atol=1e-5
    ), "Projection model not equivariant"


def test_tensor_product_layer_equivariant(so3_group):
    batch_size = 2
    num_tokens = 10

    input_inv_dimension = 50
    input_vector_multiplicty = 5

    output_inv_dimension = 50
    output_vector_multiplicty = 5

    layer = TensorProductLayer(
        input_inv_dimension=input_inv_dimension,
        input_vector_multiplicity=input_vector_multiplicty,
        output_inv_dimension=output_inv_dimension,
        output_vector_multiplicity=output_vector_multiplicty,
    )

    f = torch.randn(batch_size, num_tokens, input_inv_dimension)
    x = torch.randn(batch_size, num_tokens, 3 * input_vector_multiplicty)

    ### apply projection
    x_proj, f_proj = layer(x, f)

    # Type-0 (invariant) and Type-1 (vector) representations
    trivial_repr = so3_group.irrep(0)
    type1_representation = so3_group.irrep(1)

    x = x.reshape(batch_size * num_tokens, x.shape[-1])
    f = f.reshape(batch_size * num_tokens, f.shape[-1])

    # Field types
    vector_type = escnn.nn.FieldType(
        so3_group, input_vector_multiplicty * [type1_representation]
    )
    invariant_type = escnn.nn.FieldType(so3_group, input_inv_dimension * [trivial_repr])

    output_vector_type = escnn.nn.FieldType(
        so3_group, output_vector_multiplicty * [type1_representation]
    )
    output_invariant_type = escnn.nn.FieldType(
        so3_group, output_inv_dimension * [trivial_repr]
    )

    # Wrap tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    f = escnn.nn.GeometricTensor(f, invariant_type)

    x_proj = x_proj.reshape(batch_size * num_tokens, x_proj.shape[-1])
    f_proj = f_proj.reshape(batch_size * num_tokens, f_proj.shape[-1])
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
    x_g = x_g.tensor.reshape(batch_size, num_tokens, x_g.shape[-1])
    f_g = f_g.tensor.reshape(batch_size, num_tokens, f_g.shape[-1])
    x_g_proj, f_g_proj = layer(x_g, f_g)

    x_proj_g = x_proj_g.tensor.reshape(batch_size, num_tokens, x_proj_g.shape[-1])
    f_proj_g = f_proj_g.tensor.reshape(batch_size, num_tokens, f_proj_g.shape[-1])

    assert x_g_proj.shape == x_proj_g.shape, "Shape mismatch"
    assert f_g_proj.shape == f_proj_g.shape, "Shape Mismatch"

    assert torch.allclose(
        f_proj_g, f_g_proj, atol=1e-5
    ), "Projection model not equivariant"
    assert torch.allclose(
        x_proj_g, x_g_proj, atol=1e-5
    ), "Projection model not equivariant"
