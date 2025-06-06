import sys
import os
import torch
from torch import nn
from escnn.group import SO3
from escnn.gspaces import no_base_space
import math
from torch_geometric.utils import to_dense_batch


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, parent_dir)
from src.models import SE3HyenaOperator, SE3HyperHyenaOperator
from src.projections import (
    LinearProjection,
    TensorProductLayer,
    NormActivationLayer,
    BatchNormLayer,
)
from src.graph_layers import GraphConvLayer
from src.utils import positional_encoding
from torch import nn


import numpy as np
import torch

from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import (
    GConvSE3,
    GNormSE3,
    get_basis_and_r,
    GSE3Res,
    GMaxPooling,
    GAvgPooling,
)
from equivariant_attention.fibers import Fiber


class qm9_SE3Hyena(nn.Module):
    """SE(3)-Hyenea for qm9"""

    """Model consists of SE3GraphConv --> SE3Hyena"""

    def __init__(
        self,
        positional_encoding_dimension=16,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
        node_feature_dimension=11,
        edge_feature_dimension=4,
        kernel_size=3,
        scalar_attention_type="FFT",
        output_dimension=1,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the network
        self.positional_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3
        self.node_feature_dimension = node_feature_dimension
        self.edge_feature_dimension = edge_feature_dimension

        self.output_dimension = output_dimension
        self.scalar_attention_type = scalar_attention_type
        self.kernel_size = kernel_size

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.graph_conv = GraphConvLayer(
            node_dimension=self.node_feature_dimension,
            edge_dimension=self.edge_feature_dimension,
            out_channels=self.positional_encoding_dimension,
        )

        self.hyena_layer1 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=2 * self.input_dimension_1,
            hidden_vector_multiplicity=3,
            output_inv_dimension=self.input_dimension_1,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.mlp1 = LinearProjection(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=1,
            output_inv_dimension=self.input_dimension_2,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.hyena_layer2 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=1,
            hidden_inv_dimension=2 * self.input_dimension_3,
            hidden_vector_multiplicity=5,
            output_inv_dimension=self.input_dimension_3,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.mlp2 = LinearProjection(
            input_inv_dimension=self.input_dimension_3,
            input_vector_multiplicity=1,
            output_inv_dimension=self.input_dimension_3,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.hyena_layer3 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_3,
            input_vector_multiplicity=1,
            hidden_inv_dimension=2 * self.input_dimension_3,
            hidden_vector_multiplicity=5,
            output_inv_dimension=self.input_dimension_3,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.output_layer = LinearProjection(
            input_inv_dimension=self.input_dimension_3,
            input_vector_multiplicity=1,
            output_inv_dimension=self.output_dimension,
            output_vector_multiplicity=0,
        ).to(self.device)

    def sinusoidal_positional_encoding(self, mask: torch.Tensor, d_emb: int):
        B, N = mask.shape
        # Create position indices [0, 1, ..., N-1] for each batch
        pos_idx = torch.arange(N, device=mask.device).unsqueeze(0).repeat(B, 1)
        pos_idx = pos_idx.masked_fill(~mask, 0)  # or any neutral index for padding

        # Standard sinusoidal PE
        div_term = torch.exp(
            torch.arange(0, d_emb, 2, device=mask.device)
            * (-torch.log(torch.tensor(10000.0)) / d_emb)
        )
        pe = torch.zeros(B, N, d_emb, device=mask.device)
        pe[:, :, 0::2] = torch.sin(pos_idx.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(pos_idx.unsqueeze(-1) * div_term)
        return pe

    def forward(self, G):
        ### For a batched graph (DataBatch object)
        x, mask = to_dense_batch(G.pos, G.batch)

        ### graph conv: (node features, edge features) --> node features
        f = self.graph_conv(G)
        f, mask_2 = to_dense_batch(f, G.batch)  ### convert f to batch form

        ### positional encoding on mask
        pe = self.sinusoidal_positional_encoding(
            mask, self.positional_encoding_dimension
        )

        ### need learned encoding on node features?
        ### f = self.project_f(f)  # [B, N, d_emb]
        f = pe + f

        assert mask.shape == mask_2.shape, "masks same shape!"

        assert (
            self.positional_encoding_dimension == f.shape[2]
        ), "Feature Dimension incorrect"
        assert x.shape[2] == 3, "Coordonates wrong shape"
        assert f.shape[0] == x.shape[0]
        assert f.shape[1] == x.shape[1]

        batch_size = x.shape[0]
        num_tokens = x.shape[1]

        x, f = self.hyena_layer1.forward(x, f)

        # Save the inputs for residual
        x_resid, f_resid = x, f

        x_out, f_out = self.mlp1(x, f)
        x_out = x_out.reshape(batch_size, num_tokens, 3)
        f_out = f_out.reshape(batch_size, num_tokens, f.shape[-1])

        # Add residual
        x = x_resid + x_out
        f = f_resid + f_out

        x, f = self.hyena_layer2.forward(x, f)

        # Save the inputs for residual
        x_resid, f_resid = x, f

        x_out, f_out = self.mlp2(x, f)
        x_out = x_out.reshape(batch_size, num_tokens, 3)
        f_out = f_out.reshape(batch_size, num_tokens, f.shape[-1])

        # Add residual
        x = x_resid + x_out
        f = f_resid + f_out

        x, f = self.hyena_layer3.forward(x, f)

        ### take mean pool over N dim: [x,f] is (b,N, 3+d) Mean pooling over token dimension
        x = x.mean(dim=1).unsqueeze(1)
        f = f.mean(dim=1).unsqueeze(1)

        ### Apply the linear layer to the last dimension (b,3+d)-->(b, self.output_dimension ) i.e. output is all invariant features
        out, _ = self.output_layer(x, f)

        if self.output_dimension == 1:
            out = out.squeeze(1)  ### get rid of 1 dim

        return out.squeeze(1)


""" Baselines taken from se3-transformer"""


class TFN(nn.Module):
    """SE(3) equivariant GCN"""

    def __init__(
        self,
        num_layers: int,
        atom_feature_size: int,
        num_channels: int,
        num_nlayers: int = 1,
        num_degrees: int = 4,
        edge_dim: int = 4,
        **kwargs
    ):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels * num_degrees
        self.edge_dim = edge_dim

        self.fibers = {
            "in": Fiber(1, atom_feature_size),
            "mid": Fiber(num_degrees, self.num_channels),
            "out": Fiber(1, self.num_channels_out),
        }

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks

    def _build_gcn(self, fibers, out_dim):
        block0 = []
        fin = fibers["in"]
        for i in range(self.num_layers - 1):
            block0.append(
                GConvSE3(
                    fin, fibers["mid"], self_interaction=True, edge_dim=self.edge_dim
                )
            )
            block0.append(GNormSE3(fibers["mid"], num_layers=self.num_nlayers))
            fin = fibers["mid"]
        block0.append(
            GConvSE3(
                fibers["mid"],
                fibers["out"],
                self_interaction=True,
                edge_dim=self.edge_dim,
            )
        )

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(block0), nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {"0": G.ndata["f"]}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        return h


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(
        self,
        num_layers: int,
        atom_feature_size: int,
        num_channels: int,
        num_nlayers: int = 1,
        num_degrees: int = 4,
        edge_dim: int = 4,
        div: float = 4,
        pooling: str = "avg",
        n_heads: int = 1,
        **kwargs
    ):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {
            "in": Fiber(1, atom_feature_size),
            "mid": Fiber(num_degrees, self.num_channels),
            "out": Fiber(1, num_degrees * self.num_channels),
        }

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers["in"]
        for i in range(self.num_layers):
            Gblock.append(
                GSE3Res(
                    fin,
                    fibers["mid"],
                    edge_dim=self.edge_dim,
                    div=self.div,
                    n_heads=self.n_heads,
                )
            )
            Gblock.append(GNormSE3(fibers["mid"]))
            fin = fibers["mid"]
        Gblock.append(
            GConvSE3(
                fibers["mid"],
                fibers["out"],
                self_interaction=True,
                edge_dim=self.edge_dim,
            )
        )

        # Pooling
        if self.pooling == "avg":
            Gblock.append(GAvgPooling())
        elif self.pooling == "max":
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(
            nn.Linear(self.fibers["out"].n_features, self.fibers["out"].n_features)
        )
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers["out"].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {"0": G.ndata["f"]}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h


def model_selection(FLAGS):
    """Select model given flags"""

    if FLAGS.model == "SE3Hyena":
        model = qm9_SE3Hyena(
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
            output_dimension=1,
            kernel_size=7,
            scalar_attention_type="Standard",
            node_feature_dimension=11,
            edge_feature_dimension=4,
        )
    elif FLAGS.model == "SE3-Transformer":
        model = SE3Transformer(
            num_layers=1,
            atom_feature_size=1,
            num_channels=1,
            num_nlayers=1,
            num_degrees=4,
            edge_dim=4,
        )

    elif FLAGS.model == "TFN":
        model = TFN(
            num_layers=1,
            atom_feature_size=1,
            num_channels=1,
            num_nlayers=1,
            num_degrees=4,
            edge_dim=4,
        )

    else:
        raise ValueError

    return model
