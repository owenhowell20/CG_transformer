import torch.nn as nn
import torch
from equivariant_attention.modules import get_basis_and_r
from src.performer_pytorch import softmax_kernel


def RegularEquivariantAttention(
    q, k, v, projection_matrix, l_max=3, num_random_features=256, eps=1e-5
):
    outputs = []
    for l in range(l_max):
        q_slice = q[:, :, :, l**2 : l**2 + 2 * l + 1, :]
        k_slice = k[:, :, :, l**2 : l**2 + 2 * l + 1, :]
        v_slice = v[:, :, :, l**2 : l**2 + 2 * l + 1, :]

        projection_matrix_slice = projection_matrix[
            :, l**2 : l**2 + 2 * l + 1, :
        ].flatten(-2)
        output = HarmonicEquivariantAttention(
            q_slice, k_slice, v_slice, projection_matrix_slice, num_random_features, eps
        )
        outputs.append(output)

    return torch.cat(outputs, dim=-2)


def HarmonicEquivariantAttention(
    q, k, v, projection_matrix_slice, num_random_features=256, eps=1e-5
):
    """q,k,v ~[b, N, h, 2l+1, C] features
    I.e. features have N tokens, each with h heads of size (2l+1)xC
    prjection matrix is (num_random_features, dim)
    """
    harmonic_dim = q.shape[3]
    channels = q.shape[4]
    dim = harmonic_dim * channels

    ### reshape
    q = q.flatten(-2)  # Flattens last two dims (2l+1 and C) into one
    k = k.flatten(-2)  # Flattens last two dims (2l+1 and C) into one
    v = v.flatten(-2)  # Flattens last two dims (2l+1 and C) into one

    phi_queries = softmax_kernel(
        q, projection_matrix=projection_matrix_slice, is_query=True
    )
    phi_keys = softmax_kernel(
        k, projection_matrix=projection_matrix_slice, is_query=False
    )

    # key values contraction
    kv = torch.einsum("bhld,bhlm->bhdm", phi_keys, v)
    z = 1 / (torch.einsum("bhld,bhd->bhl", phi_queries, phi_keys.sum(dim=2)) + eps)
    output = torch.einsum("bhld,bhdm->bhlm", phi_queries, kv)
    output = output * z.unsqueeze(-1)

    output = output.reshape(
        output.shape[0], output.shape[1], output.shape[2], harmonic_dim, channels
    )

    return output


class EquivariantRandomFeatures(nn.Module):
    def __init__(
        self,
        input_invariant_multplicity: int,
        input_vector_multiplicity: int,
        num_layers: int,
        num_channels: int,
        num_degrees: int = 4,
        div: float = 4,
        n_heads: int = 1,
        si_m="1x1",
        si_e="att",
        x_ij="add",
    ):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = 1
        self.div = div
        self.n_heads = n_heads
        self.si_m, self.si_e = si_m, si_e
        self.x_ij = x_ij

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        return 1
