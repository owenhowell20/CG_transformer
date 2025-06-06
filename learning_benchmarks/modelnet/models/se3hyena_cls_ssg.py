import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, root_dir)

from src.models import SE3HyenaOperator
from src.projections import (
    HybridProjection,
    HybridLayerNorm,
    HybridRelu,
    InvariantProjection,
    InvariantUpProjection,
    InvariantPointCloudEmbedding,
)

from escnn.group import SO3
from escnn.gspaces import no_base_space
from Fourier_Position_Encoding import FourierPositionalEncoding


class get_model(nn.Module):
    def __init__(
        self, num_class, normal_channel=False, fourier_pos_enc=True, num_bands=16
    ):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.fourier_pos_enc = fourier_pos_enc

        if fourier_pos_enc:
            self.pos_enc = FourierPositionalEncoding(
                in_dim=3, num_bands=num_bands, max_freq=10.0
            )
            hyena_in_channel = 3 + 2 * 3 * num_bands
            if normal_channel:
                hyena_in_channel += 3
        else:
            hyena_in_channel = in_channel

        group = no_base_space(SO3())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hyena1 = SE3HyenaOperator(
            input_dimension=hyena_in_channel,
            output_dimension=64,
            group=group,
            device=device,
        )
        self.norm1 = nn.LayerNorm(64)
        self.hyena2 = SE3HyenaOperator(
            input_dimension=64 + hyena_in_channel,
            output_dimension=128,
            group=group,
            device=device,
        )
        self.norm2 = nn.LayerNorm(128)
        self.hyena3 = SE3HyenaOperator(
            input_dimension=128 + hyena_in_channel,
            output_dimension=256,
            group=group,
            device=device,
        )
        self.norm3 = nn.LayerNorm(256)

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        # x: [B, in_channel, N]
        x = x.transpose(1, 2).contiguous()  # [B, N, in_channel]

        # pos enc
        if self.fourier_pos_enc:
            xyz = x[:, :, :3]
            if x.shape[2] > 3:
                normals = x[:, :, 3:]
                posenc = self.pos_enc(xyz)
                pos_in = torch.cat([posenc, normals], dim=-1)
            else:
                pos_in = self.pos_enc(xyz)
        else:
            pos_in = x

        f1 = pos_in
        x1, f1 = self.hyena1(x, f1)
        f1 = self.norm1(f1)
        x1 = F.relu(x1)

        f2 = torch.cat([f1, pos_in], dim=-1)
        x2, f2 = self.hyena2(x1, f2)
        f2 = self.norm2(f2)
        x2 = F.relu(x2)

        f3 = torch.cat([f2, pos_in], dim=-1)
        x3, f3 = self.hyena3(x2, f3)
        f3 = self.norm3(f3)
        x3 = F.relu(x3)

        x = f3.mean(dim=1)  # [B, 256]
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        x = F.log_softmax(x, -1)
        return x, None


class get_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, trans_feat):
        return F.nll_loss(pred, target)
