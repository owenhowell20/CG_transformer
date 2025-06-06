from typing import List
import torch
import math
from scipy.spatial.transform import Rotation as R


def to_np(x):
    return x.cpu().detach().numpy()


def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def skew_symmetric(v):
    # v: [B, 3]
    B = v.shape[0]
    zero = torch.zeros(B, 1, device=v.device)
    vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    return torch.cat(
        [
            torch.cat([zero, -vz, vy], dim=1),
            torch.cat([vz, zero, -vx], dim=1),
            torch.cat([-vy, vx, zero], dim=1),
        ],
        dim=1,
    ).view(B, 3, 3)


def regular_rep_irreps(l_max: int, multiplicity: int, so3_group=None):
    rep = []
    for i in range(l_max):
        rep = rep + multiplicity * [so3_group.irrep(i)]

    return rep


def random_rotation_matrix():
    """Generate a random 3D rotation matrix."""
    a = torch.randn(3)
    q = a / a.norm()  # Normalize for axis
    theta = torch.randn(1) * 2 * torch.pi  # Random angle
    K = torch.tensor([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])
    R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


### convert three orthgonal 3-vectors to rot matrix
def vectors_to_so3(v1, v2, v3):
    V = torch.stack([v1, v2, v3], dim=-1)  # shape [..., 3, 3]
    U, _, Vt = torch.linalg.svd(V)
    R = U @ Vt
    # Ensure det(R) == 1
    det = torch.det(R)
    mask = det < 0
    if mask.any():
        U[..., :, -1][mask] *= -1
        R = U @ Vt
    return R


def axis_angle_to_matrix(rotvec):
    # rotvec: [B, 3]
    theta = torch.norm(rotvec, dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
    axis = rotvec / theta  # Normalize to unit axis
    K = skew_symmetric(axis)  # [B, 3, 3]

    eye = torch.eye(3, device=rotvec.device).unsqueeze(0)  # [1, 3, 3]
    sin_theta = torch.sin(theta).unsqueeze(-1)  # [B, 1, 1]
    cos_theta = torch.cos(theta).unsqueeze(-1)  # [B, 1, 1]

    # Rodrigues' rotation formula: R = I + sinθ K + (1 - cosθ) K^2
    K2 = torch.matmul(K, K)
    return eye + sin_theta * K + (1 - cos_theta) * K2


def construct_so3_frame_from_flat(v123_flat):
    """
    Construct batch of SO(3) frames from [b, 9] input.
    Each row is [v1_x, v1_y, v1_z, v2_x, ..., v3_z]
    Returns: [b, 3, 3] rotation matrices
    """
    b = v123_flat.shape[0]
    v123 = v123_flat.view(b, 3, 3)  # shape [b, 3, 3]
    v1, v2, v3 = v123[:, 0], v123[:, 1], v123[:, 2]  # each [b, 3]

    e1 = normalize(v1)  # [b, 3]

    # e2: orthogonalize v2 against e1
    proj = (e1 * v2).sum(dim=1, keepdim=True) * e1  # [b, 3]
    e2 = normalize(v2 - proj)

    # e3: cross product to ensure right-handedness
    e3 = torch.cross(e1, e2, dim=1)

    R = torch.stack([e1, e2, e3], dim=-1)  # [b, 3, 3]
    return R


def random_rotation_matrices(batch_size):
    r = R.random(batch_size)
    return torch.tensor(r.as_matrix(), dtype=torch.float32)  # shape [b, 3, 3]


def compute_SO3_dim(multiplicities: List[int]) -> int:
    return sum(mult * (2 * i + 1) for i, mult in enumerate(multiplicities))


def positional_encoding(n: int, d: int, device="cpu") -> torch.Tensor:
    """
    Generate sinusoidal positional encoding for sequence length n and feature dimension d.
    Works for even or odd d.
    Returns a tensor of shape (n, d).
    """
    position = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(
        1
    )  # (n, 1)
    div_term = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / d)
    )  # (ceil(d/2),)

    pe = torch.zeros((n, d), device=device)
    pe[:, 0::2] = torch.sin(position * div_term)

    if d % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])

    return pe


def random_rotation_batch(batch_size, device="cpu"):
    """Generate a batch of random 3D rotation matrices using QR decomposition."""
    A = torch.randn(batch_size, 3, 3, device=device)
    Q, _ = torch.linalg.qr(A)

    # Ensure det(Q) = +1 for all
    dets = torch.det(Q)
    Q[dets < 0, :, 0] *= -1
    return Q  # Shape: (batch_size, 3, 3)


def sampleSO3():
    """
    Samples a random rotation matrix from the special orthogonal group SO(3).
    Returns:
        A (3, 3) numpy array representing a random rotation matrix.
    """

    random_rotation = R.random()
    return torch.tensor(random_rotation.as_matrix(), dtype=torch.float64)
