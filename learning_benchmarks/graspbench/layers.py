import torch
import torch.nn as nn


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
