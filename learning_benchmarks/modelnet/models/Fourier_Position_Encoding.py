import torch
import torch.nn as nn
import numpy as np


class FourierPositionalEncoding(nn.Module):
    def __init__(self, in_dim=3, num_bands=16, max_freq=10.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.freq_bands = 2 ** torch.linspace(0, np.log2(max_freq), num_bands)

    def forward(self, xyz):
        # xyz: [B, N, 3]
        xb = xyz.unsqueeze(-1) * self.freq_bands.to(xyz.device)  # [B, N, 3, num_bands]
        sin = torch.sin(xb)  # [B, N, 3, num_bands]
        cos = torch.cos(xb)  # [B, N, 3, num_bands]
        fourier_feat = torch.cat([sin, cos], dim=-1)  # [B, N, 3, 2*num_bands]
        fourier_feat = fourier_feat.reshape(
            xyz.shape[0], xyz.shape[1], -1
        )  # [B, N, 3*2*num_bands]
        out = torch.cat([xyz, fourier_feat], dim=-1)  # [B, N, 3+3*2*num_bands]
        return out
