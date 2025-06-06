import torch
from fixtures import so3_group
from src.nonlinearities import normalize_features, softmax_features

import escnn
from src.utils import regular_rep_irreps


def test_equivariance_normilize_features(so3_group):
    batch_size = 4
    num_heads = 8
    seq_len = 256
    harmonic = 3
    channels = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    f = torch.randn(
        batch_size, num_heads, seq_len, 2 * harmonic + 1, channels, device=device
    )

    f_out = normalize_features(f)

    rep = channels * [so3_group.irrep(harmonic)]
    input_type = escnn.nn.FieldType(so3_group, rep)
    f = f.permute(0, 1, 2, 4, 3).reshape(
        batch_size * num_heads * seq_len, (2 * harmonic + 1) * channels
    )
    f_out = f_out.permute(0, 1, 2, 4, 3).reshape(
        batch_size * num_heads * seq_len, (2 * harmonic + 1) * channels
    )

    f = escnn.nn.GeometricTensor(f, input_type)
    f_out = escnn.nn.GeometricTensor(f_out, input_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation
    f_g = f.transform(g).tensor
    f_out_g = f_out.transform(g).tensor

    ### apply model to transform
    f_g = f_g.reshape(
        batch_size, num_heads, seq_len, channels, 2 * harmonic + 1
    ).permute(0, 1, 2, 4, 3)

    f_g_out = normalize_features(f_g)
    f_g_out = f_g_out.permute(0, 1, 2, 4, 3).reshape(
        batch_size * num_heads * seq_len, (2 * harmonic + 1) * channels
    )

    assert f_out_g.shape == f_g_out.shape, "Shape mismatch"
    assert torch.allclose(
        f_out_g,
        f_g_out,
        atol=1e-3,
    )


def test_equivariance_softmax_features(so3_group):
    batch_size = 4
    num_heads = 8
    seq_len = 256
    harmonic = 3
    channels = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    f = torch.randn(
        batch_size, num_heads, seq_len, 2 * harmonic + 1, channels, device=device
    )

    f_out = softmax_features(f)

    rep = channels * [so3_group.irrep(harmonic)]
    input_type = escnn.nn.FieldType(so3_group, rep)
    f = f.permute(0, 1, 2, 4, 3).reshape(
        batch_size * num_heads * seq_len, (2 * harmonic + 1) * channels
    )
    f_out = f_out.permute(0, 1, 2, 4, 3).reshape(
        batch_size * num_heads * seq_len, (2 * harmonic + 1) * channels
    )

    f = escnn.nn.GeometricTensor(f, input_type)
    f_out = escnn.nn.GeometricTensor(f_out, input_type)

    # apply G transformation
    g = so3_group.fibergroup.sample()

    # Apply the transformation
    f_g = f.transform(g).tensor
    f_out_g = f_out.transform(g).tensor

    ### apply model to transform
    f_g = f_g.reshape(
        batch_size, num_heads, seq_len, channels, 2 * harmonic + 1
    ).permute(0, 1, 2, 4, 3)

    f_g_out = softmax_features(f_g)
    f_g_out = f_g_out.permute(0, 1, 2, 4, 3).reshape(
        batch_size * num_heads * seq_len, (2 * harmonic + 1) * channels
    )

    assert f_out_g.shape == f_g_out.shape, "Shape mismatch"
    assert torch.allclose(
        f_out_g,
        f_g_out,
        atol=1e-3,
    )
