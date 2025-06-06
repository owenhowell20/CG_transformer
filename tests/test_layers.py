import pytest
import torch
from fixtures import mock_data, so3_group
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.layers import (
    RegularTensorProductLayer,
    RegularLinearProjection,
    RegularNormActivation,
    RegularBatchNorm,
)
import escnn
from src.utils import regular_rep_irreps


def test_tensor_product_layer():
    B = 2
    N = 4

    max_input_harmonic = 2
    input_multiplicity = 3

    output_max_harmonic = 2
    output_multiplicity = 3

    layer = RegularTensorProductLayer(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
        output_max_harmonic=output_max_harmonic,
        output_multiplicity=output_multiplicity,
    )

    inputs = torch.randn(B, N, input_multiplicity * (max_input_harmonic + 1) ** 2)
    outputs = layer(inputs)

    assert outputs.shape == (
        B,
        N,
        output_multiplicity * (output_max_harmonic + 1) ** 2,
    ), "output shape mismatch"

    ### Check values
    assert torch.isfinite(outputs).all(), "output contains NaNs or Infs"


def test_linear_layer():
    B = 2
    N = 4

    max_input_harmonic = 2
    input_multiplicity = 2

    output_max_harmonic = 2
    output_multiplicity = 3

    layer = RegularLinearProjection(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
        output_max_harmonic=output_max_harmonic,
        output_multiplicity=output_multiplicity,
    )

    inputs = torch.randn(B, N, input_multiplicity * (max_input_harmonic + 1) ** 2)
    outputs = layer(inputs)

    assert outputs.shape == (
        B,
        N,
        output_multiplicity * (output_max_harmonic + 1) ** 2,
    ), "output shape mismatch"

    ### Check values
    assert torch.isfinite(outputs).all(), "output contains NaNs or Infs"


def test_batch_norm_layer():
    B = 2
    N = 4

    max_input_harmonic = 2
    input_multiplicity = 2

    layer = RegularBatchNorm(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
    )

    inputs = torch.randn(B, N, input_multiplicity * (max_input_harmonic + 1) ** 2)
    outputs = layer(inputs)

    assert outputs.shape == (
        B,
        N,
        input_multiplicity * (max_input_harmonic + 1) ** 2,
    ), "output shape mismatch"

    ### Check values
    assert torch.isfinite(outputs).all(), "output contains NaNs or Infs"


def test_norm_activation_layer():
    B = 2
    N = 4

    max_input_harmonic = 2
    input_multiplicity = 2

    layer = RegularNormActivation(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
    )

    inputs = torch.randn(B, N, input_multiplicity * (max_input_harmonic + 1) ** 2)
    outputs = layer(inputs)

    assert outputs.shape == (
        B,
        N,
        input_multiplicity * (max_input_harmonic + 1) ** 2,
    ), "output shape mismatch"

    ### Check values
    assert torch.isfinite(outputs).all(), "output contains NaNs or Infs"


def test_equivariance_norm_activation(mock_data, so3_group):
    batch_size = 2
    num_tokens = 4

    max_input_harmonic = 2
    input_multiplicity = 2

    max_output_harmonic = max_input_harmonic
    output_multiplicity = input_multiplicity

    inputs = torch.randn(
        batch_size, num_tokens, input_multiplicity * (max_input_harmonic + 1) ** 2
    )

    layer = RegularNormActivation(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
    )
    outputs = layer(inputs)

    input_rep = regular_rep_irreps(
        l_max=max_input_harmonic + 1,
        multiplicity=input_multiplicity,
        so3_group=so3_group,
    )
    output_rep = regular_rep_irreps(
        l_max=max_output_harmonic + 1,
        multiplicity=output_multiplicity,
        so3_group=so3_group,
    )

    input_type = escnn.nn.FieldType(so3_group, input_rep)

    output_type = escnn.nn.FieldType(so3_group, output_rep)

    inputs = inputs.reshape(batch_size * num_tokens, inputs.shape[-1])
    outputs = outputs.reshape(batch_size * num_tokens, outputs.shape[-1])

    inputs = escnn.nn.GeometricTensor(inputs, input_type)
    outputs = escnn.nn.GeometricTensor(outputs, output_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation
    inputs_g = inputs.transform(g).tensor
    outputs_g = outputs.transform(g).tensor

    ### apply model to transform
    inputs_g = inputs_g.reshape(batch_size, num_tokens, inputs_g.shape[-1])
    outputs_g = outputs_g.reshape(batch_size, num_tokens, outputs_g.shape[-1])

    g_outputs = layer(inputs_g)
    g_outputs = g_outputs.reshape(batch_size, num_tokens, g_outputs.shape[-1])

    assert outputs_g.shape == g_outputs.shape, "Shape mismatch"
    assert torch.allclose(
        outputs_g, g_outputs, atol=1e-5
    ), "Projection model not equivariant"


def test_equivariance_batch_norm(mock_data, so3_group):
    batch_size = 2
    num_tokens = 4

    max_input_harmonic = 2
    input_multiplicity = 2

    max_output_harmonic = max_input_harmonic
    output_multiplicity = input_multiplicity

    inputs = torch.randn(
        batch_size, num_tokens, input_multiplicity * (max_input_harmonic + 1) ** 2
    )

    layer = RegularBatchNorm(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
    )
    outputs = layer(inputs)

    input_rep = regular_rep_irreps(
        l_max=max_input_harmonic + 1,
        multiplicity=input_multiplicity,
        so3_group=so3_group,
    )
    output_rep = regular_rep_irreps(
        l_max=max_output_harmonic + 1,
        multiplicity=output_multiplicity,
        so3_group=so3_group,
    )

    input_type = escnn.nn.FieldType(so3_group, input_rep)

    output_type = escnn.nn.FieldType(so3_group, output_rep)

    inputs = inputs.reshape(batch_size * num_tokens, inputs.shape[-1])
    outputs = outputs.reshape(batch_size * num_tokens, outputs.shape[-1])

    inputs = escnn.nn.GeometricTensor(inputs, input_type)
    outputs = escnn.nn.GeometricTensor(outputs, output_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation
    inputs_g = inputs.transform(g).tensor
    outputs_g = outputs.transform(g).tensor

    ### apply model to transform
    inputs_g = inputs_g.reshape(batch_size, num_tokens, inputs_g.shape[-1])
    outputs_g = outputs_g.reshape(batch_size, num_tokens, outputs_g.shape[-1])

    g_outputs = layer(inputs_g)
    g_outputs = g_outputs.reshape(batch_size, num_tokens, g_outputs.shape[-1])

    assert outputs_g.shape == g_outputs.shape, "Shape mismatch"
    assert torch.allclose(
        outputs_g, g_outputs, atol=1e-5
    ), "Projection model not equivariant"


def test_equivariance_linear_projection(mock_data, so3_group):
    batch_size = 2
    num_tokens = 4

    max_input_harmonic = 2
    input_multiplicity = 2

    max_output_harmonic = max_input_harmonic
    output_multiplicity = input_multiplicity

    inputs = torch.randn(
        batch_size, num_tokens, input_multiplicity * (max_input_harmonic + 1) ** 2
    )

    layer = RegularLinearProjection(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
        output_max_harmonic=max_output_harmonic,
        output_multiplicity=output_multiplicity,
    )
    outputs = layer(inputs)

    input_rep = regular_rep_irreps(
        l_max=max_input_harmonic + 1,
        multiplicity=input_multiplicity,
        so3_group=so3_group,
    )
    output_rep = regular_rep_irreps(
        l_max=max_output_harmonic + 1,
        multiplicity=output_multiplicity,
        so3_group=so3_group,
    )

    input_type = escnn.nn.FieldType(so3_group, input_rep)

    output_type = escnn.nn.FieldType(so3_group, output_rep)

    inputs = inputs.reshape(batch_size * num_tokens, inputs.shape[-1])
    outputs = outputs.reshape(batch_size * num_tokens, outputs.shape[-1])

    inputs = escnn.nn.GeometricTensor(inputs, input_type)
    outputs = escnn.nn.GeometricTensor(outputs, output_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation
    inputs_g = inputs.transform(g).tensor
    outputs_g = outputs.transform(g).tensor

    ### apply model to transform
    inputs_g = inputs_g.reshape(batch_size, num_tokens, inputs_g.shape[-1])
    outputs_g = outputs_g.reshape(batch_size, num_tokens, outputs_g.shape[-1])

    g_outputs = layer(inputs_g)
    g_outputs = g_outputs.reshape(batch_size, num_tokens, g_outputs.shape[-1])

    assert outputs_g.shape == g_outputs.shape, "Shape mismatch"
    assert torch.allclose(
        outputs_g, g_outputs, atol=1e-5
    ), "Projection model not equivariant"


def test_equivariance_tensor_product(mock_data, so3_group):
    batch_size = 2
    num_tokens = 4

    max_input_harmonic = 2
    input_multiplicity = 2

    max_output_harmonic = max_input_harmonic
    output_multiplicity = input_multiplicity

    inputs = torch.randn(
        batch_size, num_tokens, input_multiplicity * (max_input_harmonic + 1) ** 2
    )

    layer = RegularTensorProductLayer(
        input_max_harmonic=max_input_harmonic,
        input_multiplicity=input_multiplicity,
        output_max_harmonic=max_output_harmonic,
        output_multiplicity=output_multiplicity,
    )
    outputs = layer(inputs)

    input_rep = regular_rep_irreps(
        l_max=max_input_harmonic + 1,
        multiplicity=input_multiplicity,
        so3_group=so3_group,
    )
    output_rep = regular_rep_irreps(
        l_max=max_output_harmonic + 1,
        multiplicity=output_multiplicity,
        so3_group=so3_group,
    )

    input_type = escnn.nn.FieldType(so3_group, input_rep)

    output_type = escnn.nn.FieldType(so3_group, output_rep)

    inputs = inputs.reshape(batch_size * num_tokens, inputs.shape[-1])
    outputs = outputs.reshape(batch_size * num_tokens, outputs.shape[-1])

    inputs = escnn.nn.GeometricTensor(inputs, input_type)
    outputs = escnn.nn.GeometricTensor(outputs, output_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation
    inputs_g = inputs.transform(g).tensor
    outputs_g = outputs.transform(g).tensor

    ### apply model to transform
    inputs_g = inputs_g.reshape(batch_size, num_tokens, inputs_g.shape[-1])
    outputs_g = outputs_g.reshape(batch_size, num_tokens, outputs_g.shape[-1])

    g_outputs = layer(inputs_g)
    g_outputs = g_outputs.reshape(batch_size, num_tokens, g_outputs.shape[-1])

    assert outputs_g.shape == g_outputs.shape, "Shape mismatch"
    assert torch.allclose(
        outputs_g, g_outputs, atol=1e-5
    ), "Projection model not equivariant"
