import pytest
import torch
from fixtures import mock_data, so3_group
from escnn.group import SO3
from escnn.gspaces import no_base_space
import escnn
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.projections import LinearProjection
from src.SE3Hyena import VectorLongConv, VectorSelfAttention


# def test_VectorConv_translation(mock_data, so3_group):
#
#     hidden_dim = 1024
#     q_model = HybridProjection(feature_dim=512, output_dim=hidden_dim, group=so3_group)
#     k_model = HybridProjection(feature_dim=512, output_dim=hidden_dim, group=so3_group)
#     v_model = HybridProjection(feature_dim=512, output_dim=hidden_dim, group=so3_group)
#
#     conv_model = VectorLongConv()
#
#     x, f = mock_data  # Extract x and f from the fixture
#     batch_size, num_tokens, _ = x.shape
#
#     ### repeat same t along each 3 dim, i.e. for each batch sample different 3 vector
#     t = torch.randn(batch_size, 3).unsqueeze(1).repeat(1, num_tokens, 1)
#     assert t.shape == (batch_size, num_tokens, 3), "mean translation not right shape"
#     x_trans = x + t
#
#     # Ensure output shape is correct
#     q_eqv, q_inv = q_model(x, f)
#     k_eqv, k_inv = k_model(x, f)
#     v_eqv, v_inv = v_model(x, f)
#
#     q_eqv_trans, q_inv_trans = q_model(x_trans, f)
#     k_eqv_trans, k_inv_trans = k_model(x_trans, f)
#     v_eqv_trans, v_inv_trans = v_model(x_trans, f)
#
#     ### reshape into (b,N,3) and (b,N,3)
#     q_eqv = q_eqv.view(batch_size, num_tokens, 3)
#     k_eqv = k_eqv.view(batch_size, num_tokens, 3)
#     v_eqv = v_eqv.view(batch_size, num_tokens, 3)
#
#     ### reshape into (b,N,3) and (b,N,3)
#     q_eqv_trans = q_eqv_trans.view(batch_size, num_tokens, 3)
#     k_eqv_trans = k_eqv_trans.view(batch_size, num_tokens, 3)
#     v_eqv_trans = v_eqv_trans.view(batch_size, num_tokens, 3)
#
#     ### check that projections are correct
#     assert torch.allclose(
#         q_eqv + t, q_eqv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         k_eqv + t, k_eqv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         v_eqv + t, v_eqv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#
#     assert torch.allclose(
#         q_inv, q_inv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         k_inv, k_inv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         v_inv, v_inv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#
#     ### Now, check that vector convolution model is correct
#
#     ### apply vector self attention
#     u_eqv = conv_model.forward(q_eqv, k_eqv)
#     u_eqv_trans = conv_model.forward(q_eqv_trans, k_eqv_trans)
#
#     assert torch.allclose(
#         u_eqv, u_eqv_trans, atol=1e-4
#     ), "Vector convolution model is not translational invariant"
#
#
# ### test that vector attn module is translational equivariant
# def test_VectorAttn_translation(mock_data, so3_group):
#
#     hidden_dim = 1024
#     q_model = HybridProjection(feature_dim=512, output_dim=hidden_dim, group=so3_group)
#     k_model = HybridProjection(feature_dim=512, output_dim=hidden_dim, group=so3_group)
#     v_model = HybridProjection(feature_dim=512, output_dim=hidden_dim, group=so3_group)
#     x, f = mock_data  # Extract x and f from the fixture
#     batch_size, num_tokens, _ = x.shape
#
#     ### repeat same t along each 3 dim, i.e. for each batch sample different 3 vector
#     t = torch.randn(batch_size, 3).unsqueeze(1).repeat(1, num_tokens, 1)
#     assert t.shape == (batch_size, num_tokens, 3), "mean translation not right shape"
#     x_trans = x + t
#
#     ### projections
#     q_eqv, q_inv = q_model(x, f)
#     k_eqv, k_inv = k_model(x, f)
#     v_eqv, v_inv = v_model(x, f)
#
#     ### translated projections
#     q_eqv_trans, q_inv_trans = q_model(x_trans, f)
#     k_eqv_trans, k_inv_trans = k_model(x_trans, f)
#     v_eqv_trans, v_inv_trans = v_model(x_trans, f)
#
#     attn_model = VectorSelfAttention()
#
#     ### reshape into (b,N,3) and (b,N,3)
#     q_eqv = q_eqv.view(batch_size, num_tokens, 3)
#     k_eqv = k_eqv.view(batch_size, num_tokens, 3)
#     v_eqv = v_eqv.view(batch_size, num_tokens, 3)
#
#     ### reshape into (b,N,3) and (b,N,3)
#     q_eqv_trans = q_eqv_trans.view(batch_size, num_tokens, 3)
#     k_eqv_trans = k_eqv_trans.view(batch_size, num_tokens, 3)
#     v_eqv_trans = v_eqv_trans.view(batch_size, num_tokens, 3)
#
#     ### check that projections are correct
#     assert torch.allclose(
#         q_eqv + t, q_eqv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         k_eqv + t, k_eqv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         v_eqv + t, v_eqv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#
#     assert torch.allclose(
#         q_inv, q_inv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         k_inv, k_inv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#     assert torch.allclose(
#         v_inv, v_inv_trans, atol=1e-4
#     ), "Projection Model not translation equivariant"
#
#     ### apply vector self attention
#     u_eqv = attn_model.forward(q_eqv, k_eqv, v_eqv)
#     u_eqv_trans = attn_model.forward(q_eqv_trans, k_eqv_trans, v_eqv_trans)
#     assert torch.allclose(
#         u_eqv, u_eqv_trans, atol=1e-4
#     ), "Vector Attention model is not translational invariant"
#


def test_VectorSelfAttn_equivariant(mock_data, so3_group):
    """test equivariance of vector self attention module"""

    feature_dim = 512
    hidden_dim = 512
    output_dim = hidden_dim

    x, f = mock_data

    batch_size = x.shape[0]
    num_tokens = x.shape[1]

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

    attn_model = VectorSelfAttention().to(device)

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

    assert u_eqv_g.shape == u_g_eqv.shape, "dimension mismatch"
    assert u_inv_g.shape == u_g_inv.shape, "dimension mismatch"

    assert torch.allclose(
        u_g_inv, u_inv_g, atol=1e-4
    ), "Vector self attention is not equivariant"
    assert torch.allclose(
        u_g_eqv, u_eqv_g, atol=1e-4
    ), "Vector self attention is not equivariant"


def test_VectorLongConv_equivariant(mock_data):
    x, f = mock_data
    batch_size, num_tokens, _ = x.shape
    device = x.device

    # Initialize the model
    conv_model = VectorLongConv().to(device)

    so3 = no_base_space(SO3())
    type1_representation = so3.irrep(1)

    q = torch.randn_like(x)
    k = torch.randn_like(x)

    # Perform forward pass
    u_eqv = conv_model.forward(q, k)

    q = q.reshape(batch_size * num_tokens, 3)  # (BN, 3)
    k = k.reshape(batch_size * num_tokens, 3)  # (BN, 3)

    vector_type = escnn.nn.FieldType(so3, [type1_representation])

    q = escnn.nn.GeometricTensor(q, vector_type)
    k = escnn.nn.GeometricTensor(k, vector_type)

    # Sample a random group transformation
    g = so3.fibergroup.sample()

    # Apply the transformation to the vector features (q, k)
    q_g = q.transform(g)
    k_g = k.transform(g)

    q_g = q_g.tensor.reshape(batch_size, num_tokens, 3)
    k_g = k_g.tensor.reshape(batch_size, num_tokens, 3)

    # Perform forward pass with transformed inputs
    u_g_eqv = conv_model.forward(q_g, k_g)

    # Check shapes
    assert u_g_eqv.shape == u_eqv.shape, "Dimension mismatch"
    assert u_eqv.shape == (batch_size, num_tokens, 3)

    # Reshape and wrap the output tensors
    u_eqv = u_eqv.reshape(batch_size * num_tokens, 3)
    u_eqv = escnn.nn.GeometricTensor(u_eqv, vector_type)

    u_eqv_g = u_eqv.transform(g).tensor
    u_eqv_g = u_eqv_g.reshape(batch_size, num_tokens, 3)

    assert u_g_eqv.shape == u_eqv_g.shape, "Shape mismatch"
    assert torch.allclose(
        u_eqv_g, u_g_eqv, atol=1e-3
    ), "Vector convolution not equivariant"


def test_VectorConv_equivariant(mock_data, so3_group):
    feature_dim = 512
    hidden_dim = 512

    x, f = mock_data  # Extract x and f from the fixture
    batch_size = x.shape[0]  ### batch size
    num_tokens = x.shape[1]  ### num tokens
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

    ### vector conv for attention
    conv_model = VectorLongConv().to(device)

    ### query and key
    q_eqv, q_inv = q_model(x, f)
    k_eqv, k_inv = k_model(x, f)

    # Type-0 (invariant) and Type-1 (vector) representations
    trivial_repr = so3_group.irrep(0)
    type1_representation = so3_group.irrep(1)

    x = x.reshape(batch_size * num_tokens, 3)  # (BN,3)
    f = f.reshape(batch_size * num_tokens, hidden_dim)  # (BN,feature_dim)

    # Field types of inputs
    vector_type = escnn.nn.FieldType(so3_group, [type1_representation])
    invariant_type = escnn.nn.FieldType(so3_group, feature_dim * [trivial_repr])
    output_invariant_type = escnn.nn.FieldType(so3_group, hidden_dim * [trivial_repr])

    ### Wrap input tensors
    x = escnn.nn.GeometricTensor(x, vector_type)
    f = escnn.nn.GeometricTensor(f, invariant_type)

    ### get random G transformation
    g = so3_group.fibergroup.sample()

    ### Apply the g transformation to the vector features (x)
    x_g = x.transform(g)

    ### Apply the g transformation to the invariant features (f)
    f_g = f.transform(g)

    ### model to transformed features
    x_g = x_g.tensor.reshape(batch_size, num_tokens, 3)
    f_g = f_g.tensor.reshape(batch_size, num_tokens, feature_dim)
    q_g_eqv, q_g_inv = q_model(x_g, f_g)
    k_g_eqv, k_g_inv = k_model(x_g, f_g)

    ### reshape back into (b,N,3) and (b,N,3)
    q_eqv = q_eqv.view(batch_size, num_tokens, 3)
    k_eqv = k_eqv.view(batch_size, num_tokens, 3)

    ### reshape into (b,N,3) and (B,N,3)
    q_g_eqv = q_g_eqv.view(batch_size, num_tokens, 3)
    k_g_eqv = k_g_eqv.view(batch_size, num_tokens, 3)

    ### apply vector self attention
    u_eqv = conv_model.forward(q_eqv, k_eqv)

    ### apply vector self attention to transformed
    u_g_eqv = conv_model.forward(q_g_eqv, k_g_eqv)

    ### now, apply the g transform to u_eqv
    assert u_eqv.shape == (batch_size, num_tokens, 3)
    u_eqv = u_eqv.reshape(batch_size * num_tokens, 3)
    u_eqv = escnn.nn.GeometricTensor(u_eqv, vector_type)

    ### apply g transform to u_eqv
    u_eqv_g = u_eqv.transform(g)

    ## reshape
    u_eqv_g = u_eqv_g.tensor.reshape(batch_size, num_tokens, 3)

    assert u_g_eqv.shape == u_eqv_g.shape, "Shape mismatch"
    assert torch.allclose(
        u_eqv_g, u_g_eqv, atol=1e-3
    ), "vector convolution not equivariant"


def test_VectorConv(mock_data):
    group = no_base_space(SO3())

    x, f = mock_data  # Extract x and f from the fixture
    batch_size = x.shape[0]  ### batch size
    num_tokens = x.shape[1]  ### num tokens
    device = x.device

    hidden_dim = 1024
    feature_dim = 512
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

    conv_model = VectorLongConv().to(device)

    x, f = mock_data  # Extract x and f from the fixture
    b = x.shape[0]
    N = x.shape[1]

    # Ensure output shape is correct
    q_eqv, q_inv = q_model(x, f)
    k_eqv, k_inv = k_model(x, f)
    v_eqv, v_inv = v_model(x, f)

    assert q_eqv.shape[0] == k_eqv.shape[0], "Mismatch in batch and token dimensions"
    assert q_eqv.shape[1] == k_eqv.shape[1], "Mismatch in feature dimension"

    ### reshape into (b,N,3) and (b,N,3)
    q_eqv = q_eqv.view(b, N, 3)
    k_eqv = k_eqv.view(b, N, 3)

    ### apply vector self attention
    u_eqv = conv_model.forward(q_eqv, k_eqv)

    assert u_eqv.shape[2] == 3, "resultant dimension should be three"
    assert isinstance(u_eqv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(u_eqv).any(), "Output contains NaN values"


def test_VectorAttn(mock_data):
    group = no_base_space(SO3())
    hidden_dim = 1024
    feature_dim = 512

    x, f = mock_data  # Extract x and f from the fixture
    b = x.shape[0]
    N = x.shape[1]

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

    # Ensure output shape is correct
    q_eqv, q_inv = q_model(x, f)
    k_eqv, k_inv = k_model(x, f)
    v_eqv, v_inv = v_model(x, f)

    assert q_eqv.shape[0] == k_eqv.shape[0], "Mismatch in batch and token dimensions"
    assert q_eqv.shape[1] == k_eqv.shape[1], "Mismatch in feature dimension"

    attn_model = VectorSelfAttention().to(device)

    ### reshape into (b,N,3) and (b,N,3)
    q_eqv = q_eqv.view(b, N, 3)
    k_eqv = k_eqv.view(b, N, 3)
    v_eqv = v_eqv.view(b, N, 3)

    ### apply vector self attention
    u_eqv = attn_model.forward(q_eqv, k_eqv, v_eqv)

    assert u_eqv.shape[2] == 3, "resultant dimension should be three"
    assert isinstance(u_eqv, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(u_eqv).any(), "Output contains NaN values"
