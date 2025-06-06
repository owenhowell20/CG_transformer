import torch


def normalize_features(f: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize features along the 2l+1 dimension.

    Args:
        f: Input tensor of shape [b, h, N, 2l+1, m] where:
            b: batch size
            h: number of heads
            N: number of nodes
            2l+1: spherical harmonics dimension
            m: multiplicity
        eps: Small constant to avoid division by zero

    Returns:
        Normalized tensor of same shape as input [b, h, N, 2l+1, m]
    """
    norm = torch.linalg.norm(f, dim=3, keepdim=True)  # shape: [b, h, N, 1, m]
    return f / (norm + eps)  # avoid division by zero


def softmax_features(f: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax over the 2l+1 dimension while maintaining equivariance.
    First computes magnitudes, applies softmax, then scales the original values.

    Args:
        f: Input tensor of shape [b, h, N, 2l+1, m] where:
            b: batch size
            h: number of heads
            N: number of nodes
            2l+1: spherical harmonics dimension
            m: multiplicity

    Returns:
        Tensor of same shape as input [b, h, N, 2l+1, m] with softmax applied over dim=3
    """
    # Compute magnitudes along the 2l+1 dimension
    magnitudes = torch.linalg.norm(f, dim=3, keepdim=True)  # [b, h, N, 1, m]

    # Apply softmax to magnitudes
    softmax_weights = torch.softmax(magnitudes, dim=3)  # [b, h, N, 1, m]

    # Scale original values by softmax weights
    return f * softmax_weights  # [b, h, N, 2l+1, m]
