import pytest


from torch.utils.data import DataLoader
import torch
import sys
import os
import wandb
from escnn.group import SO3
from escnn.gspaces import no_base_space
from torch import optim
import escnn


from .data import mock_data, mock_tensor_data
from .test_flags import mock_flags
from learning_benchmarks.geometric_recall.models import GRDSE3HyperHyena

from learning_benchmarks.geometric_recall.models import (
    GRDSE3Hyena,
    GRDHyena,
    GRDStandard,
    GRDSE3HyperHyena,
)
import torch


def test_model(mock_data, mock_flags):
    x, f = mock_data
    batch_size, num_tokens, _ = x.shape
    FLAGS = mock_flags
    x = x.to(FLAGS.device)

    model = GRDSE3HyperHyena(
        sequence_length=num_tokens,
        positional_encoding_dimension=f.shape[0],
        input_multiplicity_1=8,
        input_harmonic_1=3,
        hidden_multiplicity_1=8,
        hidden_harmonic_1=3,
        hidden_multiplicity_2=8,
        hidden_harmonic_2=3,
        hidden_multiplicity_3=8,
        hidden_harmonic_3=3,
    )

    if isinstance(model, torch.nn.Module):
        model.to(FLAGS.device)

    y = model(x)

    print(y.shape)
    assert y.shape == (batch_size, 3), "dimension mismatch"


def test_baseline_model(mock_data, mock_flags):
    batch_size, num_tokens = 1, 10
    FLAGS = mock_flags

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(batch_size, 2 * num_tokens + 1, 3, device=device)

    model = GRDSE3Hyena(
        sequence_length=2 * num_tokens + 1,
        positional_encoding_dimension=32,
        input_dimension_1=64,
        input_dimension_2=128,
        input_dimension_3=32,
        vector_attention_type="FFT",
    ).to(FLAGS.device)

    if isinstance(model, torch.nn.Module):
        model.to(FLAGS.device)

    y = model(x)
    assert y.shape == (batch_size, 3), "dimension mismatch"


def test_equivariant_model_equivariant(mock_data, mock_flags):
    so3 = no_base_space(SO3())
    x, f = mock_data
    batch_size, num_tokens, _ = x.shape
    FLAGS = mock_flags
    x = x.to(FLAGS.device)

    model = GRDSE3HyperHyena(
        sequence_length=num_tokens,
        input_multiplicity_1=8,
        input_harmonic_1=3,
        hidden_multiplicity_1=8,
        hidden_harmonic_1=3,
        hidden_multiplicity_2=8,
        hidden_harmonic_2=3,
        hidden_multiplicity_3=8,
        hidden_harmonic_3=3,
    ).to(FLAGS.device)

    if isinstance(model, torch.nn.Module):
        model.to(FLAGS.device)

    ### output
    y = model(x)  ### y~(b,3)

    ### Type-1 (vector) representations
    type1_representation = so3.irrep(1)

    ### reshape input
    x = x.reshape(batch_size * num_tokens, 3)  # (BN,3)

    # Field types
    vector_type = escnn.nn.FieldType(so3, [type1_representation])

    # Wrap tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    y = escnn.nn.GeometricTensor(y, vector_type)

    ### apply G transformation
    g = so3.fibergroup.sample()

    # Apply the transformation to the vector features (x)
    x_g = x.transform(g).tensor
    y_g = y.transform(g).tensor

    # ### apply model to transform
    x_g = x_g.reshape(batch_size, num_tokens, 3)
    x_g = x_g.to(FLAGS.device)
    z = model(x_g)

    assert y.shape == z.shape, "dimension mismatch"
    assert torch.allclose(y_g, z, atol=1e-5), "Full model not equivariant"


def test_baseline_equivariant_model_equivariant(mock_data, mock_flags):
    so3 = no_base_space(SO3())
    x, f = mock_data
    batch_size, num_tokens, _ = x.shape
    FLAGS = mock_flags
    x = x.to(FLAGS.device)

    model = GRDSE3Hyena(
        sequence_length=num_tokens,
        positional_encoding_dimension=32,
        input_dimension_1=64,
        input_dimension_2=128,
        input_dimension_3=32,
    ).to(FLAGS.device)

    if isinstance(model, torch.nn.Module):
        model.to(FLAGS.device)

    ### output
    y = model(x)  ### y~(b,3)

    ### Type-1 (vector) representations
    type1_representation = so3.irrep(1)

    ### reshape input
    x = x.reshape(batch_size * num_tokens, 3)  # (BN,3)

    # Field types
    vector_type = escnn.nn.FieldType(so3, [type1_representation])

    # Wrap tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    y = escnn.nn.GeometricTensor(y, vector_type)

    ### apply G transformation
    g = so3.fibergroup.sample()

    # Apply the transformation to the vector features (x)
    x_g = x.transform(g).tensor
    y_g = y.transform(g).tensor

    # ### apply model to transform
    x_g = x_g.reshape(batch_size, num_tokens, 3)
    x_g = x_g.to(FLAGS.device)
    z = model(x_g)

    assert y.shape == z.shape, "dimension mismatch"
    assert torch.allclose(y_g, z, atol=1e-5), "Full model not equivariant"
