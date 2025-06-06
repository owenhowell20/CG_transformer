import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, parent_dir)


def knn(x, k):
    """
    x: (batch_size, num_points, num_dims)
    k: number of neighbors
    return: (batch_size, num_points, k, num_dims)
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_edge_feature(x, k=20):
    """
    x: (batch_size, num_points, num_dims)
    return: (batch_size, num_points, k, 2*num_dims)
    """
    batch_size, num_points, num_dims = x.shape
    idx = knn(x, k=k)  # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.reshape(batch_size * num_points, -1)
    feature = x[idx].view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat(
        (feature - x, x), dim=3
    )  # (batch_size, num_points, k, 2*num_dims)
    return feature


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        batch_size, num_points, num_dims = x.shape
        x = get_edge_feature(x, k=self.k)  # (batch_size, num_points, k, 2*num_dims)
        x = x.permute(0, 3, 1, 2)  # (batch_size, 2*num_dims, num_points, k)
        x = self.conv(x)  # (batch_size, out_channels, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, out_channels, num_points)
        x = x.permute(0, 2, 1)  # (batch_size, num_points, out_channels)
        return x


class ModelNetDGCNN(nn.Module):
    def __init__(
        self, num_classes=40, k=20, emb_dims=1024, dropout=0.5  # Default to ModelNet40
    ):
        super(ModelNetDGCNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Edge convolution layers
        self.edge_conv1 = EdgeConv(3, 64, k=k)
        self.edge_conv2 = EdgeConv(64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, k=k)

        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(512, emb_dims),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(emb_dims, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, data):
        # Get points from data
        if hasattr(data, "pos"):
            points = data.pos
        else:
            points = data.x[:, :3]

        # Get batch indices
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(points.size(0), dtype=torch.long, device=points.device)

        # Convert to dense batch if needed
        if len(points.shape) == 2:  # (num_points_total, 3)
            batch_size = batch.max().item() + 1
            num_points = points.size(0) // batch_size

            # Reshape points to (batch_size, num_points, 3)
            points_list = []
            for i in range(batch_size):
                mask = batch == i
                points_batch = points[mask]
                if points_batch.size(0) < num_points:
                    # Pad with zeros if needed
                    padding = torch.zeros(
                        num_points - points_batch.size(0),
                        points_batch.size(1),
                        device=points.device,
                    )
                    points_batch = torch.cat([points_batch, padding], 0)
                points_list.append(points_batch)
            points = torch.stack(points_list, 0)

        # Apply edge convolutions
        x1 = self.edge_conv1(points)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        # Concatenate and global pooling
        x = torch.cat([x1, x2, x3, x4], dim=2)
        x = x.max(dim=1)[0]  # Global max pooling

        # Classification
        logits = self.mlp(x)

        return logits


class ModelNetDGCNNLoss(nn.Module):
    def __init__(self):
        super(ModelNetDGCNNLoss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        return F.cross_entropy(pred, target)
