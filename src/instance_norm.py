import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from e3nn.o3 import Irreps


class InstanceNorm(nn.Module):
    """Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.

    Parameters
    ----------
    harmonic_order : int
        harmonic order of the representation
    multiplicity : int
        multiplicity of the representation
    eps : float
        avoid division by zero
    """

    def __init__(
        self,
        harmonic_order,
        multiplicity,
        eps=1e-5,
    ):
        super().__init__()

        self.harmonic_order = harmonic_order
        self.multiplicity = multiplicity
        self.eps = eps

    def forward(self, input):
        """evaluate
        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, N, 2*l+1, m)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, N, 2*l+1, m)``
        """
        batch, N, _, _ = input.shape

        # Take the norm over the 2*l+1 dimension and keep dimensions for broadcasting
        norm = torch.norm(input, p=2, dim=2, keepdim=True)
        norm = norm.clamp(min=self.eps)

        # Normalize
        output = input / norm

        return output
