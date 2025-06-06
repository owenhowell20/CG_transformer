import torch
from contextlib import nullcontext
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch

import torch
from torch_geometric.data import Data, Batch

import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.from_se3cnn import utils_steerable


### @profile
def get_basis_pyg(G, max_degree, compute_gradients):
    """Precompute the SE(3)-equivariant weight basis, W_J^lk(x)

    This is called by get_basis_and_r().

    Args:
        G: torch_geometric.data
        max_degree: non-negative int for degree of highest feature type
        compute_gradients: boolean, whether to compute gradients during basis construction
    Returns:
        dict of equivariant bases. Keys are in the form 'd_in,d_out'. Values are
        tensors of shape (batch_size, 1, 2*d_out+1, 1, 2*d_in+1, number_of_bases)
        where the 1's will later be broadcast to the number of output and input
        channels
    """
    if compute_gradients:
        context = nullcontext()
    else:
        context = torch.no_grad()

    with context:
        cloned_d = torch.clone(G.edge_attr)

        if G.edge_attr.requires_grad:
            cloned_d.requires_grad_()
            # log_gradient_norm(cloned_d, 'Basis computation flow')

        # Relative positional encodings (vector)
        r_ij = utils_steerable.get_spherical_from_cartesian_torch(cloned_d)
        # Spherical harmonic basis
        Y = utils_steerable.precompute_sh(r_ij, 2 * max_degree)
        device = Y[0].device

        basis = {}
        for d_in in range(max_degree + 1):
            for d_out in range(max_degree + 1):
                K_Js = []
                for J in range(abs(d_in - d_out), d_in + d_out + 1):
                    # Get spherical harmonic projection matrices
                    Q_J = utils_steerable._basis_transformation_Q_J(J, d_in, d_out)
                    Q_J = Q_J.float().T.to(device)

                    # Create kernel from spherical harmonics
                    K_J = torch.matmul(Y[J], Q_J)
                    K_Js.append(K_J)

                # Reshape so can take linear combinations with a dot product
                size = (-1, 1, 2 * d_out + 1, 1, 2 * d_in + 1, 2 * min(d_in, d_out) + 1)
                basis[f"{d_in},{d_out}"] = torch.stack(K_Js, -1).view(*size).to(device)
        return basis


# @profile
def get_r_pyg(G):
    """Compute internodal distances"""
    # Access the edge attributes (similar to 'edata['d']' in DGL)
    cloned_d = torch.clone(G.edge_attr)

    # Check if edge_attr requires gradients
    if G.edge_attr.requires_grad:
        cloned_d.requires_grad_()
        # log_gradient_norm(cloned_d, 'Neural networks flow')

    # Calculate the Euclidean distance (internodal distances)
    distances = torch.sqrt(cloned_d**2)
    return distances


def sort_keys(list_of_basis_dicts):
    """
    Stack the basis tensors from a list of dicts into a single dict,
    stacking along the first (batch) dimension.

    Args:
        list_of_basis_dicts: List[Dict[str, Tensor]], each dict has keys like '0,0', '1,1', etc.,
                             and values of shape [1, :, :, :, :, :]

    Returns:
        Dict[str, Tensor]: A new dict where each value is a tensor of shape [B, :, :, :, :, :]
    """
    stacked_basis = {}
    keys = list_of_basis_dicts[0].keys()

    for key in keys:
        stacked_basis[key] = torch.cat(
            [basis_dict[key] for basis_dict in list_of_basis_dicts], dim=0
        )

    return stacked_basis


# @profile
def get_basis_and_r_pyg(G, max_degree, compute_gradients=False):
    """Return equivariant weight basis (basis) and internodal distances (r).

    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function
    can be shared as input across all SE(3)-Transformer layers in a model.

    Args:
        G: batch PyG graph
        max_degree: non-negative int for degree of highest feature-type
        compute_gradients: controls whether to compute gradients during basis construction
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
        vector of relative distances, ordered according to edge ordering of G
    """
    basis = get_basis_pyg(G, max_degree, compute_gradients)
    r = get_r_pyg(G)
    return basis, r


### generate the W matrices for the point cloud
# @profile
def generate_matrices(
    point_cloud_features: torch.tensor,
    max_degree: int = 5,
    compute_gradients: bool = True,
):
    """Args:
     point cloud data: (b,N,3)
     max_degree: maximum degree harmonic
     compute_gradients: specifify gradient computation

    Returns:
        dict of equivariant bases. Keys are in the form 'd_in,d_out'. Values are
        tensors of shape (batch_size, 1, 2*d_out+1, 1, 2*d_in+1, number_of_bases)
        where the 1's will later be broadcast to the number of output and input
        channels"""

    ### convert the pointcloud to PyG format
    assert point_cloud_features.shape[2] == 3, "point cloud has wrong shape"

    ### (b, N,3) point cloud coordonates
    point_cloud = point_cloud_features[:, :, 0:3]
    batch_size = point_cloud.shape[0]
    N = point_cloud.shape[1]  ### number of points

    # A fully connected graph has edges between every pair of nodes
    i, j = torch.combinations(
        torch.arange(N), r=2
    ).T  # All unique pairs (i, j) where i != j
    edge_index = torch.stack([i, j], dim=0)  # Shape: (2, N*(N-1)/2)

    # Step 2: Create the batch (identifying which nodes belong to which graph)
    # We will use a tensor that assigns each node in the batch to its respective graph
    batch = torch.arange(batch_size).repeat_interleave(N)

    all_basis = []
    all_r = []

    for graph_id in range(batch_size):
        # Slice the nodes for the current graph
        x_graph = point_cloud[graph_id]

        # For fully connected graphs, we use the same edge_index for each graph
        data = Data(
            x=x_graph, edge_index=edge_index + graph_id * N, batch=batch == graph_id
        )

        # Step 4: Add edge attributes (e.g., random values)
        edge_attr = torch.randn(
            edge_index.size(1)
        )  # Random edge attributes, same number as edges
        data.edge_attr = edge_attr  # Assign edge attributes to the data object

        basis, r = get_basis_and_r_pyg(
            data,
            max_degree=max_degree,
            compute_gradients=compute_gradients,
        )
        all_basis.append(basis)  # basis is a dict[str -> Tensor]
        all_r.append(r)  # r is a tensor of shape [E]

    basis = sort_keys(all_basis)
    r = torch.stack(all_r, dim=0)  # shape: [b, E]
    return basis, r
