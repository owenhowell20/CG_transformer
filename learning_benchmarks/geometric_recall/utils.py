import torch


def random_rotation_batch(batch_size, device="cpu"):
    """Generate a batch of random 3D rotation matrices using QR decomposition."""
    A = torch.randn(batch_size, 3, 3, device=device)
    Q, _ = torch.linalg.qr(A)

    # Ensure det(Q) = +1 for all
    dets = torch.det(Q)
    Q[dets < 0, :, 0] *= -1
    return Q  # Shape: (batch_size, 3, 3)
