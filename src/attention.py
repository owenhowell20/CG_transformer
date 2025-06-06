import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, device="cpu"):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.device = device

        assert (
            input_dim % num_heads == 0
        ), "Input dimension must be divisible by number of heads"
        self.head_dim = input_dim // num_heads
        self.embed_dim = num_heads * self.head_dim

        # Learnable linear projections
        self.q_proj = nn.Linear(input_dim, self.embed_dim).to(device)
        self.k_proj = nn.Linear(input_dim, self.embed_dim).to(device)
        self.v_proj = nn.Linear(input_dim, self.embed_dim).to(device)
        self.out_proj = nn.Linear(self.embed_dim, input_dim).to(device)

    def forward(self, x):
        """
        Multi-Head Self-Attention

        :param x: Input tensor of shape (batch, seq_len, input_dim)
        :return: Output tensor of shape (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Project input to queries, keys, values
        q = self.q_proj(x)  # (batch, seq_len, num_heads * head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (batch, num_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(
            attn_weights, v
        )  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )  # (batch, seq_len, num_heads * head_dim)
        out = self.out_proj(attn_output)  # (batch, seq_len, input_dim)

        return out


class ScalarSelfAttention(nn.Module):
    def __init__(self):
        super(ScalarSelfAttention, self).__init__()

    def forward(self, q, k, v):
        """

        Standard Scalar Self-Attention, no learnable weights

        :param q: Queries of shape (batch, sequence, d)
        :param k: Keys of shape (batch, sequence, d)
        :param v: Values of shape (batch, sequence, d)

        :return: Output of shape (batch, sequence, d)
        """

        # Compute attention scores using dot product (batch, seq, seq)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, seq, seq)

        # Scale by sqrt(d) for stability
        d = q.size(-1)
        attn_scores = attn_scores / d**0.5

        # Softmax over keys dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq, seq)

        # Weighted sum of values
        output = torch.matmul(attn_weights, v)  # (batch, seq, d)

        return output


class ScalarLongConv(nn.Module):
    def __init__(self, dimension, kernel_size, stride=1):
        super(ScalarLongConv, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.dimension = dimension

        # Each feature dimension has its own kernel: (d, kernel_size)
        self.conv_filter = nn.Parameter(torch.randn(self.dimension, kernel_size))

    def forward(self, q, k):
        """
        q, k: (batch, seq_len, d)
        returns: (batch, seq_len, d)
        """
        B, N, D = q.shape
        q_conv = q.permute(0, 2, 1)  # (B, D, N)

        # Modulate filters using mean over k
        k_mean = k.mean(dim=1)  # (B, D)
        filter_mod = self.conv_filter.unsqueeze(0) * k_mean.unsqueeze(-1)  # (B, D, K)

        # Apply grouped conv per batch manually
        out_list = []
        for b in range(B):
            filt = filter_mod[b].unsqueeze(1)  # (D, 1, K)
            x = q_conv[b].unsqueeze(0)  # (1, D, N)
            y = F.conv1d(x, filt, stride=self.stride, padding=self.padding, groups=D)
            out_list.append(y)

        out = torch.cat(out_list, dim=0)  # (B, D, N)
        return out.permute(0, 2, 1)  # (B, N, D)


class ScalarFFTConv1D(nn.Module):
    def __init__(self, dimension, kernel_size):
        super(ScalarFFTConv1D, self).__init__()
        self.dimension = dimension
        self.kernel_size = kernel_size

        # Base learnable kernel (shared across batch)
        self.weight = nn.Parameter(torch.randn(self.dimension, kernel_size))  # (D, K)

        # Project key to a single modulation scalar per dimension
        self.key_proj = nn.Linear(self.dimension, self.dimension)

    def forward(self, q, k):
        """
        :param q: Query tensor of shape (B, N, D)
        :param k: Key tensor of shape (B, N, D)
        :return: Output tensor of shape (B, N, D)
        """
        B, N, D = q.shape
        K = self.kernel_size

        # Summarize key across sequence to modulate filters
        k_mod = self.key_proj(k.mean(dim=1))  # (B, D)
        k_mod = k_mod.unsqueeze(-1)  # (B, D, 1)

        # Modulate the base weight
        dynamic_weight = self.weight.unsqueeze(0) * k_mod  # (B, D, K)

        # Pad to match FFT size
        padded_weight = torch.zeros(B, D, N, device=q.device)
        padded_weight[:, :, :K] = dynamic_weight  # (B, D, N)

        # FFT
        q_f = torch.fft.rfft(q, dim=1)  # (B, N//2+1, D)
        w_f = torch.fft.rfft(padded_weight, dim=2)  # (B, D, N//2+1)

        # Permute for broadcast: (B, D, F) → (B, F, D)
        w_f = w_f.permute(0, 2, 1)  # (B, F, D)

        # Element-wise product in frequency domain
        y_f = q_f * w_f  # (B, F, D)

        # Inverse FFT
        y = torch.fft.irfft(y_f, n=N, dim=1)  # (B, N, D)

        return y


if __name__ == "__main__":
    b, N, d = 64, 256, 32
    q = torch.randn(b, N, d)
    k = torch.randn(b, N, d)

    conv = ScalarLongConv(dimension=d, kernel_size=7)
    fft_conv = ScalarFFTConv1D(dimension=d, kernel_size=7)

    out = fft_conv(q, k)
    assert out.shape == (b, N, d), "dimension mismatch"

    out = conv(q, k)
    assert out.shape == (b, N, d), "dimension mismatch"
