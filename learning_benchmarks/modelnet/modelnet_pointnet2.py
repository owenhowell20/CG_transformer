import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import PointNetConv as PointConv


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class ModelNetPointNet2(nn.Module):
    """PointNet++ for ModelNet classification"""

    def __init__(
        self,
        num_classes=40,  # Default to ModelNet40
        set_abstraction_ratio_1=0.5,
        set_abstraction_ratio_2=0.25,
        set_abstraction_radius_1=0.2,
        set_abstraction_radius_2=0.4,
        feature_dim_1=128,
        feature_dim_2=256,
        feature_dim_3=1024,
        dropout=0.3,
    ):
        super(ModelNetPointNet2, self).__init__()

        self.num_classes = num_classes

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            MLP([3, 64, 64, feature_dim_1]),
        )

        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            MLP([feature_dim_1 + 3, 128, 128, feature_dim_2]),
        )

        self.sa3_module = GlobalSetAbstraction(
            MLP([feature_dim_2 + 3, 256, 512, feature_dim_3])
        )

        # Final MLP for classification - similar to SE3Hyena model
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim_3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        ).to(self.device)

    def forward(self, G):
        # For PyG data objects
        x = G.x if hasattr(G, "x") else None
        pos = G.pos
        batch = G.batch

        # First set abstraction layer
        x, pos, batch = self.sa1_module(x, pos, batch)

        # Second set abstraction layer
        x, pos, batch = self.sa2_module(x, pos, batch)

        # Global set abstraction (pooling)
        x, pos, batch = self.sa3_module(x, pos, batch)

        # Classification head
        logits = self.classifier(x)

        return logits

    def predict(self, G):
        """For compatibility with ModelNetSE3Hyena interface"""
        return self.forward(G)
