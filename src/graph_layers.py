import torch
from haiku import dropout
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from e3nn.o3 import Irreps
from e3nn.o3 import FullyConnectedTensorProduct, spherical_harmonics
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree

from typing import List

import torch
from torch_geometric.utils import to_dense_adj
from torch import linalg


import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data


def spectral_decomposition(graph_adj_matrix, low_rank=None):
    """Computes the Laplacian and low-rank spectral decomposition of the graph adjecency matrix"""
    batch_size, N, _ = graph_adj_matrix.shape

    # Step 1: Degree matrix (diagonal matrix of node degrees)
    degree_matrix = torch.diag_embed(graph_adj_matrix.sum(dim=-1))
    laplacian = degree_matrix - graph_adj_matrix

    if low_rank is None:
        # Full eigen-decomposition
        eigs, U = torch.linalg.eigh(laplacian)
        return U, eigs
    else:
        # Low-rank approx with LOBPCG
        U_list, eigs_list = [], []

        for i in range(batch_size):
            L = laplacian[i]  # [N, N]
            # Random initial guess for eigenvectors
            X = torch.randn(N, low_rank, device=L.device, dtype=L.dtype)

            # Get the largest r eigenvalues and eigenvectors using LOBPCG
            eigvals, eigvecs = torch.lobpcg(L, X=X, largest=True)
            eigs_list.append(eigvals)  # [low_rank]
            U_list.append(eigvecs)  # [N, low_rank]

        eigs = torch.stack(eigs_list, dim=0)  # [batch_size, low_rank]
        U = torch.stack(U_list, dim=0)  # [batch_size, N, low_rank]

        return U, eigs


def PyG_GraphFourierTransform(G, rank: int = 1):
    """Compute the Fourier transform on each graph in a PyG batched graph at rank r.

    Inputs:
        G: PyG batched graph, a list of Data objects.
        rank: int, rank of low-rank approximation of each graph adjacency matrix.

    Returns:
        U: [n_g, r] matrix, the low-rank approximation of each graph in the batch.
    """

    # Now loop over each graph in the batch (access the individual graphs using batch)

    U_list = []
    batch_size = G.num_graphs  # Number of graphs in the batch

    for i in range(batch_size):
        # Get the adjacency matrix for the i-th graph
        start = G.batch == i
        end = G.batch == i
        graph = G[start]  # Extract the i-th graph from the batch

        graph_adj_matrix = to_dense_adj(
            graph.edge_index, max_num_nodes=graph.num_nodes
        )[
            0
        ]  # [N, N]

        # Perform the spectral decomposition for the current graph
        u_g, _ = spectral_decomposition(graph_adj_matrix, low_rank=rank)

        U_list.append(u_g)

    # Concatenate all graph results (U) into one tensor
    U = torch.cat(U_list, dim=0)  # [n_g, r], where n_g is the total number of nodes

    return U


class PEGLayer(nn.Module):
    def __init__(
        self,
        node_input_dimension: int,
        node_output_dimension: int,
        emb_dims: int = 256,
        dropout: float = 0.1,
    ):
        super(PEGLayer, self).__init__()
        self.node_input_dimension = node_input_dimension
        self.node_output_dimension = node_output_dimension

        ### node embedding weights
        self.W_node = nn.Linear(node_input_dimension, node_output_dimension)

        # MLP for embedding scalar distances
        self.distance_embedding_layer = nn.Sequential(
            nn.Linear(1, emb_dims),
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
            nn.Linear(256, 1),
        )

    def compute_distances(self, positions, edge_index):
        """
        Compute the distances between nodes (for simplicity, using L2 distance between nodes).
        Args:
            positions: spatial positions of nodes tensor of shape (num_nodes, 3)
            edge_index: Edge indices (2, num_edges)

        Returns:
            torch.Tensor: Distances for each edge (num_edges, 1)
        """
        row, col = edge_index

        ### Compute pairwise Euclidean distances
        dist = torch.norm(
            positions[row] - positions[col], dim=-1, keepdim=True
        )  # L2 distance
        return dist

    def compute_normalized_adjacency(self, edge_index, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float)  # shape: (num_nodes,)

        deg[deg == 0] = (
            1  # Pretend nodes with zero degree have self-loops with degree 1 for normalization
        )
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # shape: (num_edges,)
        return norm

    def forward(self, G):
        """
        Forward pass for the PEG layer.
        Args:
            G: a torch_geometric.data.Batch graph containing nodes and edges.

        Returns:
            torch.Tensor: embedding of shape (num_nodes, node_output_dimension + 3).
        """

        pos = G.pos  ### node positions (num_nodes, 3)
        x = G.x  ### Node features (num_nodes, node_input_dimension)
        edge_index = G.edge_index  ### Edge indices (2, num_edges)

        ### compute distance matrix
        distances = self.compute_distances(pos, edge_index)  # Shape: (num_edges, 1)

        # Pass distances through the distance embedding layer
        distance_embeddings = self.distance_embedding_layer(
            distances
        )  # Shape: (num_edges, 1)

        ### now compute the node embeddings (num_nodes, node_output_dimension)
        x_proj = x @ self.W_node

        ### Now, compute \hat{A}, the normilized graph adj matrix
        norm = self.compute_normalized_adjacency(edge_index, num_nodes=x.size(0))

        ### Now, compute Hamard product of distance embeddings with \hat{A}
        scaled_edge_weights = (
            norm.view(-1, 1) * distance_embeddings
        )  # Shape: (num_edges, 1)

        # Message Passing: Aggregate features
        row, col = edge_index
        aggr_node_features = torch.zeros_like(
            x_proj
        )  # (num_nodes, node_output_dimension)

        # For each edge (i, j), we scale the node features and sum them at the nodes
        for i in range(len(row)):
            i_idx = row[i]  # Start node
            j_idx = col[i]  # End node

            # Scale the node features by the edge weight
            aggr_node_features[i_idx] += scaled_edge_weights[i] * x_proj[j_idx]
            aggr_node_features[j_idx] += scaled_edge_weights[i] * x_proj[i_idx]

        ###Now, aggr_node_features ~ (N, d_out) apply elementwise softmax
        aggr_node_features = torch.relu(aggr_node_features)
        return torch.cat([aggr_node_features, pos], dim=-1)


class HyenaGraphConv(nn.Module):
    def __init__(self, node_dimension: int, order: int = 1, low_rank=None):
        """
        Args:
            node_dimension List(int): Sequence of node feature dimensions
            order int= 1 : number of rounds
            low_rank (None or int): If None, use all eigenvectors/eigenvalues.
        """
        super(HyenaGraphConv, self).__init__()
        self.order = order
        self.node_dimension = node_dimension

        self.input_proj = torch.nn.Linear(self.node_dimension, self.node_dimension)

        self.spectral_layers = nn.ModuleList()
        for i in range(self.order):
            self.spectral_layers.append(
                AdaptiveSpectralConvLayer(
                    node_dimension=self.node_dimension,
                    output_dimension=self.node_dimension,
                    low_rank=self.low_rank,
                )
            )

    def forward(self, G, U):
        """
        Forward pass for the AdaptiveSpectralConvLayer.
        Args:
            G: a batched torch_geometric.data.Batch graph
            U: precomputed graph fourier matrix


        Returns:
            torch.Tensor: The output tensor after applying spectral convolution.
        """

        ### One key thing: only have to compute U once for each graph!
        x = G.node  ### what

        z = self.input_proj(x)

        ### compute the order + 1 projections u
        for k in range(self.order):
            u = self.input_proj(x)

            print("START:", z.shape)
            z = self.spectral_layers[k].forward(G, U)
            print("end:", z.shape)

            print(k, u.shape, z.shape)

        return z


class AdaptiveSpectralConvLayer(nn.Module):
    def __init__(self, node_dimension, output_dimension, low_rank=None):
        """
        Initializes the SpectralConvLayer.

        Args:
            node_dimension (int): Input feature dimension per node.
            output_dimension (int): Output feature dimension per node.
            low_rank (None or int): If None, use all eigenvectors/eigenvalues.
        """
        super(AdaptiveSpectralConvLayer, self).__init__()
        self.node_dimension = node_dimension
        self.output_dimension = output_dimension
        self.low_rank = low_rank

        # Filter: applied in the Fourier (spectral) space, of size (self.low_rank, self.node_dimension, self.output_dimension )
        self.filter = nn.Parameter(
            torch.randn(self.low_rank, self.node_dimension, self.output_dimension)
        )

    @staticmethod
    def pad_node_features(x, batch, max_num_nodes=None):
        """
        Args:
            x: [total_nodes, input_dim]
            batch: [total_nodes] - graph id for each node
            max_num_nodes: int or None - maximum nodes to pad to (optional)

        Returns:
            padded_x: [batch_size, max_num_nodes, input_dim]
        """
        batch_size = batch.max().item() + 1
        input_dim = x.size(-1)

        if max_num_nodes is None:
            num_nodes_per_graph = torch.bincount(batch)
            max_num_nodes = num_nodes_per_graph.max().item()

        padded_x = torch.zeros(batch_size, max_num_nodes, input_dim, device=x.device)

        node_ptr = 0
        for i in range(batch_size):
            num_nodes = (batch == i).sum().item()
            padded_x[i, :num_nodes, :] = x[batch == i]

        return padded_x

    def forward(self, G, U=None):
        """
        Forward pass for the AdaptiveSpectralConvLayer.
        Args:
        x: node features [ b_g , node_dim], where b_g is the PyG batch dim, i.e. graphs smooshed together
        graph_adj_matrix: graph adj matrix [b,n,N]
        batch: a batched torch_geometric.data.Batch graph

           U (optional): if the matrix U is already computed,

        Returns:
            torch.Tensor: The output tensor after applying spectral convolution.
        """

        if U is not None:
            ### compute the rank-r graph fourier transform matrix for each graph in batch
            U, eigs = self.spectral_decomposition(
                graph_adj_matrix, low_rank=self.low_rank
            )

        # Pad features, don't want to do this...
        x_padded = G.x  ###self.pad_node_features(x, batch)  ## # [b,N, d]

        ### fourier transform
        hat_x = torch.einsum("bnr,bnd->brd", U, x_padded)

        # Fourier-space convolution (low-rank), hat_x: (r, node_dimension), spectral filter: (r, node_dimension, output_dimension)
        hat_x = torch.einsum("brd, rdo -> bro", hat_x, self.filter)

        ### inverse fourier transform U^{T} = U^{-1} is orthogonal
        x = torch.einsum("bnr,bro->bno", U, hat_x)

        return x


class GraphConvLayer(MessagePassing):
    def __init__(self, node_dimension, edge_dimension, out_channels):
        super(GraphConvLayer, self).__init__(aggr="add")  # Use "add" aggregation
        self.node_dimension = node_dimension
        self.edge_dimension = edge_dimension
        self.out_channels = out_channels

        ### liner output dim
        self.lin = torch.nn.Linear(
            self.node_dimension + self.edge_dimension, out_channels
        )

        # Nonlinearity
        self.activation = torch.nn.ReLU()

    def forward(self, G):
        """G: a batched torch_geometric.data.Batch graph"""

        x = G.x  # [N, node_dimension], node features
        edge_index = G.edge_index  ### [ N , edge_dimension]
        edge_attr = G.edge_attr  ### [E, 2]
        pos = G.pos  ### [N, 3]

        ### Standard message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Apply linear transformation to node features
        out = self.lin(out)
        out = self.activation(out)
        # mask is [B, N_max], useful for masking padded nodes
        return out

    def message(self, x_j, edge_attr):
        """Define message computation"""
        # Concatenate node features and edge features along the feature dimension
        return torch.cat([x_j, edge_attr], dim=-1)  # Concatenate on feature dimension

    def update(self, aggr_out):
        """Aggregate node features"""
        return aggr_out  # Just return the aggregated message


class SE3GNN(torch.nn.Module):
    """SE3 equivariant graph msg passing"""

    def __init__(self, node_dimension, edge_dimension, out_channels, group=None):
        super().__init__()
        self.node_dimension = node_dimension
        self.edge_dimension = edge_dimension
        self.out_channels = out_channels
        self.group = group

        self.lin = torch.nn.Linear(
            self.node_dimension + self.edge_dimension, self.out_channels
        )  # readout

    def forward(self, G):
        """G: a batched torch_geometric.data.Batch graph"""

        x = G.x  # [N, F], node features
        pos = G.pos  # [N, 3], 3D coordinates
        edge_index = G.edge_index
        edge_attr = G.edge_attr

        row, col = edge_index
        edge_vec = pos[row] - pos[col]  # edge vectors r_{ij} ~ [E, 3]

        agg = torch.cat([x, edge_attr], dim=-1)

        # Readout to scalar prediction
        return self.lin(agg)  # [N, self.out_channels]


class SpectralConvLayer(nn.Module):
    def __init__(
        self, graph_adj_matrix, node_dimension, output_dimension, low_rank=None
    ):
        """
        Initializes the SpectralConvLayer.

        Args:
            graph_adj_matrix (torch.Tensor): The adjacency matrix of the graph, shape [N, N].
            node_dimension (int): Input feature dimension per node.
            output_dimension (int): Output feature dimension per node.
            low_rank (None or int): If None, use all eigenvectors/eigenvalues.
        """
        super(SpectralConvLayer, self).__init__()
        self.node_dimension = node_dimension
        self.output_dimension = output_dimension

        laplacian, U, eigs = self.spectral_decomposition(graph_adj_matrix)

        if low_rank is not None:
            eigvecs = U[:, :low_rank]

        self.register_buffer("U", eigvecs)  # [N, r] or [N, N]
        self.register_buffer("U_inv", eigvecs.t())  # orthogonal, so U_inv = U^T

        # Filter: applied in the original (node) space
        self.filter = nn.Linear(self.node_dimension, self.output_dimension)

    def spectral_decomposition(graph_adj_matrix):
        """
        Computes the Laplacian of the graph and performs spectral decomposition.

        Args:
            graph_adj_matrix (torch.Tensor): The adjacency matrix of the graph. Must be square matrix

        Returns:
            laplacian (torch.Tensor): The Laplacian matrix.
            U (torch.Tensor): The eigenvectors (matrix of eigenvectors).
            eigs (torch.Tensor): The eigenvalues (diagonal elements).
        """
        # Step 1: Compute the degree matrix D (diagonal)
        degree_matrix = torch.diag(
            graph_adj_matrix.sum(dim=1)
        )  # D is diagonal with row sums of A

        # Step 2: Compute the Laplacian matrix L = D - A
        laplacian = degree_matrix - graph_adj_matrix

        # Step 3: Compute the eigenvectors and eigenvalues of the Laplacian matrix
        # Using `torch.linalg.eigh` for symmetric matrices (Laplacian is symmetric)
        eigs, U = torch.linalg.eigh(laplacian)

        return laplacian, U, eigs

    def forward(self, x):
        """
        Forward pass for the SpectralConvLayer.

        Args:
            x (torch.Tensor): The input tensor representing node features of dimension [ b, N, d]

        Returns:
            torch.Tensor: The output tensor after applying spectral convolution.
        """
        batch_size = x.shape[0]
        num_nodes = x.shape[1]

        assert x.shape[2] == self.node_dimension, "Assert Node dimension not correct"

        ### take fourier transformation of x
        x_hat = torch.einsum("nm,bmx->bnx", self.U_inv, x)

        ### multiply with filter
        x_hat = torch.einsum("nfg,bnf->bng", self.filter.weight, x_hat)

        ### inverse fourier transform
        x = torch.einsum("xy,bnx->bny", self.U, x_hat)

        assert x.shape[0] == batch_size
        assert x.shape[1] == num_nodes
        assert x.shape[2] == self.output_dimension

        return x
