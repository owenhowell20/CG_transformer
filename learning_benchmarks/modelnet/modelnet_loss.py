import torch
import torch.nn.functional as F


def get_classification_loss(pred, target, **kwargs):
    """
    Standard cross-entropy loss for classification

    Args:
        pred: Model predictions (B, num_classes)
        target: Ground truth labels (B,)

    Returns:
        loss: Cross entropy loss
        accuracy: Classification accuracy
    """
    loss = F.cross_entropy(pred, target)

    # Calculate accuracy
    _, pred_idx = torch.max(pred, dim=1)
    correct = (pred_idx == target).float().sum()
    accuracy = correct / target.shape[0]

    return loss, accuracy
