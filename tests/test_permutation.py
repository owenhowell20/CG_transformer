import sys
import os
import torch
from fixtures import mock_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.attention import ScalarSelfAttention, ScalarLongConv, ScalarFFTConv1D
from src.SE3Hyena import VectorSelfAttention


### test FFT scalar conv for permutation invariance
def test_scalar_FFT_conv(mock_data):
    # Extract x and f from the fixture
    x, f = mock_data
    assert x.device == f.device

    batch_size, num_tokens, d = f.shape

    attention = ScalarFFTConv1D(dimension=d, kernel_size=3).to(x.device)

    q = f
    k = f
    v = f

    ### apply attention
    attn = attention(q, k)

    ### apply permutation to f
    perm = torch.randperm(num_tokens, device=x.device)  # random perm of [0, ..., N-1]
    f_perm = f[:, perm, :]

    q_perm = f_perm
    k_perm = f_perm
    v_perm = f_perm

    attn_perm = attention(q_perm, k_perm)
    perm_attn = attn[:, perm, :]

    assert attn_perm.shape == perm_attn.shape, "Attention Shapes do not match"
    assert not torch.allclose(
        attn_perm, perm_attn, atol=1e-4
    ), "Permutation Attentions do not match"


def test_se3_hyena_attention(mock_data):
    # Extract x and f from the fixture
    x, f = mock_data
    assert x.device == f.device

    batch_size, num_tokens, d = f.shape

    attention = VectorSelfAttention().to(device=f.device)

    q = x
    k = x
    v = x

    ### apply attention
    attn = attention(q, k, v)

    ### apply permutation to f
    perm = torch.randperm(num_tokens, device=x.device)  # random perm of [0, ..., N-1]
    x_perm = x[:, perm, :]

    q_perm = x_perm
    k_perm = x_perm
    v_perm = x_perm

    attn_perm = attention(q_perm, k_perm, v_perm)
    perm_attn = attn[:, perm, :]

    assert torch.allclose(
        attn_perm, perm_attn, atol=1e-4
    ), "Permutation Attentions do not match"


def test_transformer_permutation(mock_data):
    # Extract x and f from the fixture
    x, f = mock_data
    assert x.device == f.device

    batch_size, num_tokens, d = f.shape

    attention = ScalarSelfAttention().to(device=f.device)

    q = f
    k = f
    v = f

    ### apply attention
    attn = attention(q, k, v)

    ### apply permutation to f
    perm = torch.randperm(num_tokens, device=x.device)  # random perm of [0, ..., N-1]
    f_perm = f[:, perm, :]

    q_perm = f_perm
    k_perm = f_perm
    v_perm = f_perm

    attn_perm = attention(q_perm, k_perm, v_perm)
    perm_attn = attn[:, perm, :]

    assert torch.allclose(
        attn_perm, perm_attn, atol=1e-4
    ), "Permutation Attentions do not match"
