import torch
import torch.nn as nn


class VectorCrossProduct(nn.Module):
    """
    Module to compute the vector cross product between two 3D vectors.

    Given two input tensors `u` and `v` of shape (B, N, 3), representing batches
    of 3D vectors, this module computes their cross product `u × v` along the last
    dimension, returning a tensor of the same shape.

    Inputs:
        u (Tensor): Input tensor of shape (B, N, 3)
        v (Tensor): Input tensor of shape (B, N, 3)

    Returns:
        Tensor: Cross product `u × v`, shape (B, N, 3)
    """

    def __init__(self):
        super(VectorCrossProduct, self).__init__()

    def forward(self, u, v):
        cross_product = torch.cross(u, v, dim=-1)
        return cross_product


class VectorDotProduct(nn.Module):
    def __init__(self):
        super(VectorDotProduct, self).__init__()

    def forward(self, u, v):
        """u~ (b,N,d) , v ~ (b,N,d)
        Compute the dot product u v ~ ( b, N ,d )
        """

        # Compute the dot product between u and v along the last dimension (d)
        dot_product = torch.sum(u * v, dim=-1, keepdim=True)  # Shape will be (b, N, 1)

        # In this case, return it with the same shape as f (b, N, d)
        dot_product = dot_product.expand(
            -1, -1, f.size(2)
        )  # Broadcasting to shape (b, N, d)
        assert f.shape == dot_product.shape
        return f + dot_product


class VectorLongConv(nn.Module):
    """
    SE(3)-equivariant long-range vector convolution module.

    This module performs an equivariant convolution between two input vector fields
    (`q` and `k`) using a factorized cross-product representation. The operation
    subtracts out the mean of each input to enforce translational equivariance, then
    applies a structured cross-product transformation in Fourier space using FFT-based
    convolution for efficiency.

    The convolution is defined using precomputed low-rank tensor decompositions (L, H, P),
    which are stored as buffers and applied in a batched fashion.

    Args:
        None

    Inputs:
        q (Tensor): Query tensor of shape (B, N, 3), representing a sequence of 3D vectors.
        k (Tensor): Key tensor of shape (B, N, 3), same shape as `q`.

    Returns:
        Tensor: Output tensor of shape (B, N, 3), representing the convolved vector field.
    """

    def __init__(self):
        super(VectorLongConv, self).__init__()
        # L cross-product tensor factorized:
        l_reduced = torch.FloatTensor(
            [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
        )

        self.register_buffer("l_reduced", l_reduced, persistent=False)

        h_reduced = torch.FloatTensor(
            [[0, 1, 0], [0, 0, -1], [-1, 0, 0], [0, 0, 1], [1, 0, 0], [0, -1, 0]]
        )

        self.register_buffer("h_reduced", h_reduced, persistent=False)

        # P cross-product tensor factorized:
        p_reduced = torch.FloatTensor(
            [[0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0]]
        )

        self.register_buffer("p_reduced", p_reduced, persistent=False)

    def forward(self, q, k):
        B, N, D = q.shape  # (batch, sequence, 3)

        ### translational equivariance: q
        q_t_mean = torch.mean(q, dim=1)
        assert q_t_mean.shape == (q.shape[0], q.shape[2]), "Mean translation mismatch"

        ### subtract off the mean of q
        q_t_copy = q_t_mean.unsqueeze(1).repeat(1, q.shape[1], 1)
        assert q_t_copy.shape == q.shape
        q = q - q_t_copy

        ### translational equivariance: k
        k_t_mean = torch.mean(k, dim=1)
        assert k_t_mean.shape == (k.shape[0], k.shape[2]), "Mean translation mismatch"

        ### subtract off the mean of k
        k_t_copy = k_t_mean.unsqueeze(1).repeat(1, k.shape[1], 1)
        assert k_t_copy.shape == k.shape
        k = k - k_t_copy

        # Batchify L, H, P reduced matrices
        l_reduced = self.l_reduced[None, None].repeat(B, N, 1, 1)
        h_reduced = self.h_reduced[None, None].repeat(B, N, 1, 1)
        p_reduced = self.p_reduced[None, None].repeat(B, N, 1, 1)

        # Expand inputs with reduced L, H, P matrices
        q_expd = torch.matmul(l_reduced, q.unsqueeze(-1)).squeeze(-1)
        k_expd = torch.matmul(h_reduced, k.unsqueeze(-1)).squeeze(-1)

        # FFT-based convolution
        fft_q = torch.fft.rfft(q_expd, n=N, dim=1)
        fft_k = torch.fft.rfft(k_expd, n=N, dim=1)
        fft_conv_qk = torch.fft.irfft(fft_q * fft_k, n=N, dim=1)

        # Reduce to vector product and normalize
        u = torch.matmul(p_reduced, fft_conv_qk.unsqueeze(-1)).squeeze(-1) / N

        return u


class VectorSelfAttention(nn.Module):
    """
    SE(3)-equivariant vector self-attention mechanism.

    This module computes a cross-product-based self-attention between query, key, and
    value vector fields, designed to be equivariant under 3D translations and rotations.
    Attention weights are derived from the norm of cross products between query and key
    vectors, and applied to the value vectors via a second cross product.

    All inputs are mean-centered to enforce translational equivariance.

    Args:
        None

    Inputs:
        q (Tensor): Query tensor of shape (B, N, 3), where B is the batch size,
                    N is the sequence length, and 3 is the spatial dimension.
        k (Tensor): Key tensor of shape (B, N, 3), same shape as `q`.
        v (Tensor): Value tensor of shape (B, N, 3), same shape as `q`.

    Returns:
        Tensor: Output tensor of shape (B, N, 3), representing the attention-modulated
                vector field, equivariant under SE(3) transformations.
    """

    def __init__(self):
        super(VectorSelfAttention, self).__init__()

    def forward(self, q, k, v):
        # q, k, v: (batch, sequence, dim=3)
        B, N, D = q.shape

        ### translational equivariance: q
        q_t_mean = torch.mean(q, dim=1)
        assert q_t_mean.shape == (q.shape[0], q.shape[2]), "Mean translation mismatch"

        ### subtract off the mean of q
        q_t_copy = q_t_mean.unsqueeze(1).repeat(1, q.shape[1], 1)
        assert q_t_copy.shape == q.shape
        q = q - q_t_copy

        ### translational equivariance: k
        k_t_mean = torch.mean(k, dim=1)
        assert k_t_mean.shape == (k.shape[0], k.shape[2]), "Mean translation mismatch"

        ### subtract off the mean of k
        k_t_copy = k_t_mean.unsqueeze(1).repeat(1, k.shape[1], 1)
        assert k_t_copy.shape == k.shape
        k = k - k_t_copy

        ### translational equivariance: v
        v_t_mean = torch.mean(v, dim=1)
        assert v_t_mean.shape == (v.shape[0], v.shape[2]), "Mean translation mismatch"

        ### subtract off the mean of x
        v_t_copy = v_t_mean.unsqueeze(1).repeat(1, v.shape[1], 1)
        assert v_t_copy.shape == v.shape
        v = v - v_t_copy

        # Cross product matrix C: (B, N, N, 3)
        q_expd, k_expd = q.unsqueeze(2), k.unsqueeze(1)
        C = torch.cross(q_expd, k_expd, dim=-1)

        # \eta(C) matrix of norms and softmax: (B, N, N, 1)
        eta_C = C.norm(dim=-1, keepdim=True)
        sm_eta_C = nn.functional.softmax(eta_C / (N**0.5), dim=2)

        # S matrix: (B, N, N, 3)
        S = sm_eta_C * C

        # Compute u = S x v: (B, N, 3)
        pre_u = torch.cross(S, v.unsqueeze(2), dim=-1)
        u = pre_u.mean(dim=2)

        return u
