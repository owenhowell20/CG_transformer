import pytest
import torch
from fixtures import so3_group
import escnn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.models import GATr_model


def test_full_model(so3_group):
    input_inv_dimension = 3
    input_vector_multiplicity = 4

    output_inv_multiplicity = 2
    output_vec_multiplicity = 3

    batch_size = 5
    N = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(
        batch_size,
        N,
        input_inv_dimension + 3 * input_vector_multiplicity,
        device=device,
    )

    model = GATr_model(
        input_inv_dimension=input_inv_dimension,
        input_vector_multiplicity=input_vector_multiplicity,
        output_inv_dimension=output_inv_multiplicity,
        output_vector_multiplicity=output_vec_multiplicity,
    ).to(device)

    x = model(x)
    assert x.shape[0] == batch_size, "batch dimensions should be the same"
    assert x.shape[1] == N, "token dimensions should be the same"
    assert (
        x.shape[2] == output_inv_multiplicity + output_vec_multiplicity * 3  # vec + inv
    ), "Coordonates must be dimension three"
    assert isinstance(x, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(x).any(), "Output contains NaN values"


# def test_full_model_equivariant(so3_group):
#
#     input_multiplicity = 3
#     max_input_harmonic = 2
#
#     hidden_multiplicity = 2
#     max_hidden_harmonic = 3
#
#     output_multiplicity = 2
#     max_output_harmonic = 3
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     batch_size = 5
#     N = 10
#
#     x = torch.randn(
#         batch_size, N, input_multiplicity * (max_input_harmonic + 1) ** 2, device=device
#     )
#
#     model = SE3HyperHyenaOperator(
#         input_multiplicity=input_multiplicity,
#         input_max_harmonic=max_input_harmonic,
#         hidden_multiplicity=hidden_multiplicity,
#         hidden_max_harmonic=max_hidden_harmonic,
#         output_multiplicity=output_multiplicity,
#         output_max_harmonic=max_output_harmonic,
#     ).to(device)
#
#     x_model = model(x)
#
#     input_rep = regular_rep_irreps(
#         l_max=max_input_harmonic + 1,
#         multiplicity=input_multiplicity,
#         so3_group=so3_group,
#     )
#
#     output_rep = regular_rep_irreps(
#         l_max=max_output_harmonic + 1,
#         multiplicity=output_multiplicity,
#         so3_group=so3_group,
#     )
#
#     # Field types
#     type_in = escnn.nn.FieldType(so3_group, input_rep)
#     type_out = escnn.nn.FieldType(so3_group, output_rep)
#
#     # Wrap tensors
#     x = x.reshape(batch_size * N, x.shape[-1])
#     x = escnn.nn.GeometricTensor(x, type_in)
#     x_model = x_model.reshape(batch_size * N, x_model.shape[-1])
#
#     x_model = escnn.nn.GeometricTensor(x_model, type_out)
#
#     ### apply G transformation
#     g = so3_group.fibergroup.sample()
#
#     # Apply the transformation to the vector features (x)
#     x_g = x.transform(g).tensor
#     x_model_g = x_model.transform(g).tensor
#
#     ### reshape x_g back
#     x_g = x_g.reshape(batch_size, N, x_g.shape[-1])
#     x_model_g = x_model_g.reshape(batch_size, N, x_model_g.shape[-1])
#
#     ### now compute model
#     x_g_model = model(x_g)
#
#     assert x_g_model.shape == x_model_g.shape, "shapes not the same"
#     assert torch.allclose(x_model_g, x_g_model, atol=1e-5), "Max absolute error:" + str(
#         (x_model_g - x_g_model).abs().max().item()
#     )
