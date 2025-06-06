import torch
import torch.nn.functional as F


def compute_loss(prediction, target):
    """Cosine distance loss"""
    ### [b,N,3] and [b,N,3] prediction and target
    pred_flat = prediction.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    cos_dist = 1 - cos_sim
    return cos_dist.mean()


def compute_loss_mse(pred, values):
    """
    Computes the MSE loss between predicted and true vectors, no normalization.

    Args:
        pred (torch.Tensor): Predicted vectors with shape (b, 3)
        values (torch.Tensor): True vectors with shape (b, 3)

    Returns:
        torch.Tensor: The mean squared error loss.
    """
    loss = F.mse_loss(pred, values)
    return loss.mean()


def compute_loss_l2(pred, values):
    """
    Computes the mean L2 distance between predicted vectors (pred) and true values (values).

    Args:
        pred (torch.Tensor): Predicted vectors with shape (b, N, 3)
        values (torch.Tensor): True values with shape (b, N, 3)

    Returns:
        torch.Tensor: The mean L2 loss
    """

    # Compute squared difference
    diff = pred - values

    # Compute squared Euclidean distance for each vector
    squared_distance = torch.sum(diff**2, dim=-1)  # Sum over the last dimension (3)

    # Compute the mean L2 loss across the batch and N
    loss = torch.mean(squared_distance)  # Mean over the batch and N

    return loss
