import pytest
import torch
from fixtures import mock_data, so3_group

import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.projections import EquivariantGating

from escnn.group import SO3
from escnn.gspaces import no_base_space
from src.SE3Hyena import VectorLongConv, VectorSelfAttention
import escnn
from src.projections import LinearProjection


def test_small_gamma_gate(mock_data, so3_group):
    """test gamma gating"""
    x, f = mock_data  # Extract x and f from the fixture
    hidden_dim = f.shape[2]

    device = x.device

    gate = EquivariantGating(
        input_inv_dimension=hidden_dim, input_vector_multiplicity=1
    ).to(device)
    ### apply gamma gate
    m_eqv, m_inv = gate(x, f)

    assert isinstance(m_eqv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(m_eqv).any(), "Output contains NaN values"

    assert isinstance(m_inv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(m_inv).any(), "Output contains NaN values"


### test gamma gate equivariant
def test_small_gamma_gate_equivariant(mock_data, so3_group):
    """test gamma gating"""
    x, f = mock_data  # Extract x and f from the fixture
    batch_size, num_tokens, hidden_dim = f.shape
    feature_dim = hidden_dim

    device = x.device

    gate = EquivariantGating(
        input_inv_dimension=hidden_dim, input_vector_multiplicity=1
    ).to(device)
    ### apply gamma gate
    m_eqv, m_inv = gate(x, f)

    ###### NOW, transform the inputs by g and compute the forward

    ### Type-0 (invariant) and Type-1 (vector) representations
    trivial_repr = so3_group.irrep(0)
    type1_representation = so3_group.irrep(1)

    ### reshape inputs
    x = x.reshape(batch_size * num_tokens, 3)  # (BN,3)
    f = f.reshape(batch_size * num_tokens, feature_dim)  # (BN,feature_dim)

    # Field types
    vector_type = escnn.nn.FieldType(so3_group, [type1_representation])
    invariant_type = escnn.nn.FieldType(so3_group, hidden_dim * [trivial_repr])

    # Wrap tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    f = escnn.nn.GeometricTensor(f, invariant_type)

    # apply the G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation to the input features
    x_g = x.transform(g).tensor
    f_g = f.transform(g).tensor

    ### reshape back
    x_g = x_g.reshape(batch_size, num_tokens, 3)  # (BN,3)
    f_g = f_g.reshape(batch_size, num_tokens, feature_dim)

    ### apply gamma gate
    m_g_eqv, m_g_inv = gate(x_g, f_g)

    assert torch.allclose(m_eqv, m_g_eqv), "gating not invariant"
    assert torch.allclose(m_inv, m_g_inv), "gating not invariant"


### Test that the gamma gate is equivariant
def test_gamma_gate_equivariant(mock_data, so3_group):
    """test gamma gating"""

    feature_dim = 512
    hidden_dim = 512
    output_dim = hidden_dim

    x, f = mock_data  # Extract x and f from the fixture

    device = x.device

    q_model = LinearProjection(
        input_inv_dimension=feature_dim,
        input_vector_multiplicity=1,
        output_inv_dimension=hidden_dim,
        output_vector_multiplicity=1,
    ).to(device)
    k_model = LinearProjection(
        input_inv_dimension=feature_dim,
        input_vector_multiplicity=1,
        output_inv_dimension=hidden_dim,
        output_vector_multiplicity=1,
    ).to(device)
    v_model = LinearProjection(
        input_inv_dimension=feature_dim,
        input_vector_multiplicity=1,
        output_inv_dimension=hidden_dim,
        output_vector_multiplicity=1,
    ).to(device)
    gate = EquivariantGating(
        input_inv_dimension=hidden_dim, input_vector_multiplicity=1
    ).to(device)

    ### vector self attention
    attn_model = VectorSelfAttention().to(device)

    batch_size = x.shape[0]
    num_tokens = x.shape[1]

    #### compute the forward
    q_eqv, q_inv = q_model(x, f)
    k_eqv, k_inv = k_model(x, f)
    v_eqv, v_inv = v_model(x, f)

    ### reshape into (b,N,3) and (b,N,3)
    q_eqv = q_eqv.view(batch_size, num_tokens, 3)
    k_eqv = k_eqv.view(batch_size, num_tokens, 3)
    v_eqv = v_eqv.view(batch_size, num_tokens, 3)

    q_inv = q_inv.view(batch_size, num_tokens, hidden_dim)
    k_inv = k_inv.view(batch_size, num_tokens, hidden_dim)
    v_inv = v_inv.view(batch_size, num_tokens, hidden_dim)

    u_eqv = attn_model.forward(q_eqv, k_eqv, v_eqv)
    u_inv = q_inv  ### (b, N, d)

    ### apply gamma gate
    m_eqv, m_inv = gate(u_eqv, u_inv)

    ### m_inv and m_eqv should be dimension 1 i.e. (b,N,1) shaped
    ### apply sigmoid to 1 dimension
    u_eqv = torch.sigmoid(m_eqv) * u_eqv
    u_inv = torch.sigmoid(m_inv) * u_inv

    ###### NOW, transform the inputs by g and compute the forward

    ### Type-0 (invariant) and Type-1 (vector) representations
    trivial_repr = so3_group.irrep(0)
    type1_representation = so3_group.irrep(1)

    ### reshape inputs
    x = x.reshape(batch_size * num_tokens, 3)  # (BN,3)
    f = f.reshape(batch_size * num_tokens, feature_dim)  # (BN,feature_dim)

    # Field types
    vector_type = escnn.nn.FieldType(so3_group, [type1_representation])
    invariant_type = escnn.nn.FieldType(so3_group, feature_dim * [trivial_repr])
    output_invariant_type = escnn.nn.FieldType(so3_group, output_dim * [trivial_repr])

    # Wrap tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    f = escnn.nn.GeometricTensor(f, invariant_type)

    # apply the G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation to the input features
    x_g = x.transform(g).tensor
    f_g = f.transform(g).tensor

    ### reshape back
    x_g = x_g.reshape(batch_size, num_tokens, 3)  # (BN,3)
    f_g = f_g.reshape(batch_size, num_tokens, feature_dim)

    ### now apply model projections to the transformed inputs
    q_g_eqv, q_g_inv = q_model(x_g, f_g)
    k_g_eqv, k_g_inv = k_model(x_g, f_g)
    v_g_eqv, v_g_inv = v_model(x_g, f_g)

    ### reshape into (b,N,3) and (b,N,3)
    q_g_eqv = q_g_eqv.view(batch_size, num_tokens, 3)
    k_g_eqv = k_g_eqv.view(batch_size, num_tokens, 3)
    v_g_eqv = v_g_eqv.view(batch_size, num_tokens, 3)

    q_g_inv = q_g_inv.view(batch_size, num_tokens, hidden_dim)
    k_g_inv = k_g_inv.view(batch_size, num_tokens, hidden_dim)
    v_g_inv = v_g_inv.view(batch_size, num_tokens, hidden_dim)

    u_g_eqv = attn_model.forward(q_g_eqv, k_g_eqv, v_g_eqv)
    u_g_inv = q_g_inv  ### (b, N, d)

    ### apply gamma gate
    m_g_eqv, m_g_inv = gate(u_g_eqv, u_g_inv)

    ### m_inv and m_eqv should be dimension 1 i.e. (b,N,1) shaped
    ### apply sigmoid to 1 dimension
    u_g_eqv = torch.sigmoid(m_g_eqv) * u_g_eqv
    u_g_inv = torch.sigmoid(m_g_inv) * u_g_inv

    assert u_g_inv.shape == u_inv.shape, "dimension mismatch"
    assert u_g_eqv.shape == u_eqv.shape, "dimension mismatch"

    ### Now, we need to transform the original outputs by g
    ### first wrap u_inv and u_eqv as geometric tensors
    u_eqv = u_eqv.view(batch_size * num_tokens, 3)
    u_inv = u_inv.view(batch_size * num_tokens, hidden_dim)

    u_eqv = escnn.nn.GeometricTensor(u_eqv, vector_type)
    u_inv = escnn.nn.GeometricTensor(u_inv, output_invariant_type)

    ### apply g transformation
    u_inv_g = u_inv.transform(g).tensor
    u_eqv_g = u_eqv.transform(g).tensor

    ### reshape back
    u_eqv_g = u_eqv_g.view(batch_size, num_tokens, 3)
    u_inv_g = u_inv_g.view(batch_size, num_tokens, hidden_dim)

    assert torch.allclose(
        u_g_inv, u_inv_g, atol=1e-4
    ), "Invariant gating is not equivariant"
    assert torch.allclose(
        u_g_eqv, u_eqv_g, atol=1e-4
    ), "Equivariant gating is not equivariant"


def test_gamma_gate(mock_data, so3_group):
    """test gamma gating"""
    hidden_dim = 1024
    feature_dim = 512

    x, f = mock_data  # Extract x and f from the fixture
    device = x.device

    q_model = LinearProjection(
        input_inv_dimension=feature_dim,
        input_vector_multiplicity=1,
        output_inv_dimension=hidden_dim,
        output_vector_multiplicity=1,
    ).to(device)
    k_model = LinearProjection(
        input_inv_dimension=feature_dim,
        input_vector_multiplicity=1,
        output_inv_dimension=hidden_dim,
        output_vector_multiplicity=1,
    ).to(device)
    v_model = LinearProjection(
        input_inv_dimension=feature_dim,
        input_vector_multiplicity=1,
        output_inv_dimension=hidden_dim,
        output_vector_multiplicity=1,
    ).to(device)
    gate = EquivariantGating(
        input_inv_dimension=hidden_dim, input_vector_multiplicity=1
    ).to(device)

    attn_model = VectorSelfAttention().to(device)

    b = x.shape[0]
    N = x.shape[1]

    # Ensure output shape is correct
    q_eqv, q_inv = q_model(x, f)
    k_eqv, k_inv = k_model(x, f)
    v_eqv, v_inv = v_model(x, f)

    ### reshape into (b,N,3) and (b,N,3)
    q_eqv = q_eqv.view(b, N, 3)
    k_eqv = k_eqv.view(b, N, 3)
    v_eqv = v_eqv.view(b, N, 3)

    q_inv = q_inv.view(b, N, hidden_dim)
    k_inv = k_inv.view(b, N, hidden_dim)
    v_inv = v_inv.view(b, N, hidden_dim)

    u_eqv = attn_model.forward(q_eqv, k_eqv, v_eqv)
    u_inv = q_inv  ### (b, N, d)

    ### apply gamma gate
    m_eqv, m_inv = gate(u_eqv, u_inv)

    assert isinstance(m_eqv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(m_eqv).any(), "Output contains NaN values"

    assert isinstance(m_inv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(m_inv).any(), "Output contains NaN values"

    ### m_inv and m_eqv should be dimension 1 i.e. (b,N,1) shaped
    ### apply sigmoid to 1 dimension
    u_eqv = torch.sigmoid(m_eqv) * u_eqv
    u_inv = torch.sigmoid(m_inv) * u_inv

    assert isinstance(u_eqv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(u_eqv).any(), "Output contains NaN values"

    assert isinstance(u_inv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(u_inv).any(), "Output contains NaN values"
