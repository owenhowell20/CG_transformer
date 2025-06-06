from escnn.group import SO3
from escnn.gspaces import no_base_space
from torch_geometric.utils import to_dense_batch
import torch
from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias, GConvSE3
from torch import nn

from equivariant_attention.modules import (
    GConvSE3,
    GNormSE3,
    get_basis_and_r,
    GSE3Res,
    GMaxPooling,
    GAvgPooling,
)
from equivariant_attention.fibers import Fiber

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import (
    embed_point,
    embed_scalar,
    extract_point,
    extract_scalar,
    embed_translation,
    extract_translation,
)

import sys
import os

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models import SE3HyenaOperator, StandardAttention, HyenaAttention
from src.projections import LinearProjection, BatchNormLayer, NormActivationLayer
from src.utils import positional_encoding


class nbody_Standard(nn.Module):
    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension=16,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
    ):
        super().__init__()
        # Build the network
        self.sequence_length = sequence_length
        self.positional_encoding_dimension = positional_encoding_dimension
        self.token_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### positional encodings
        self.invariant_features = positional_encoding(
            self.sequence_length, self.positional_encoding_dimension, device=self.device
        ).to(self.device)

        ### token embeddings
        self.token_type_embedding = nn.Embedding(3, self.token_encoding_dimension).to(
            self.device
        )

        self.group = no_base_space(SO3())  ### group
        self.input_layer = nn.Linear(
            self.positional_encoding_dimension + 3, self.positional_encoding_dimension
        )

        ### se3 hyena operators
        self.hyena_layer1 = StandardAttention(
            input_dimension=self.positional_encoding_dimension,
            hidden_dimension=self.positional_encoding_dimension,
            output_dimension=self.input_dimension_1,
            num_heads=8,  ### default 8 heads
        ).to(self.device)

        self.mlp1 = nn.Linear(
            self.input_dimension_1,
            self.input_dimension_1,
        ).to(self.device)

        self.hyena_layer2 = StandardAttention(
            input_dimension=self.input_dimension_1,
            hidden_dimension=self.input_dimension_1,
            output_dimension=self.input_dimension_2,
            num_heads=8,  ### default 8 heads
        ).to(self.device)

        self.mlp2 = nn.Linear(self.input_dimension_2, self.input_dimension_2).to(
            self.device
        )

        self.hyena_layer3 = StandardAttention(
            input_dimension=self.input_dimension_2,
            hidden_dimension=self.input_dimension_2,
            output_dimension=self.input_dimension_3,
            num_heads=8,  ### default 8 heads
        ).to(self.device)

        self.output_proj = nn.Linear(self.input_dimension_3, 3).to(self.device)

    def forward(self, G):
        # For a batched graph (DataBatch object)
        f_t, mask = to_dense_batch(
            G.x, G.batch
        )  # Node features, positions (x), tensor shape: [batch_size, num_nodes, feature_size]
        x_t = f_t[:, :, 0:3].to(self.device)  ### [b, N, 3]
        v_t = f_t[:, :, 3:6].to(self.device)  ### [b, N, 3]
        c = f_t[:, :, -1].to(self.device)  ### [ b, N ]

        ### stack x and v: state vector
        s_t = torch.cat([x_t, v_t], dim=1)
        batch_size, num_tokens, _ = s_t.shape
        assert s_t.shape[2] == 3, "dimension mismatch"

        ### stack charges: invariant features
        pos_enc = self.invariant_features.unsqueeze(0).repeat(s_t.shape[0], 1, 1)
        charge_enc = (
            torch.cat([c, c], dim=1)
            .unsqueeze(-1)
            .expand(-1, -1, self.positional_encoding_dimension)
        )  ### (b,2N,1) ### encoding here
        f = pos_enc + charge_enc

        ### now, apply the first SE(3)-Hyena Operator Layer
        assert s_t.shape[0] == f.shape[0], "batch sizes not equal"
        assert s_t.shape[1] == f.shape[1], "num tokens not equal"
        assert (
            f.shape[2] == self.positional_encoding_dimension
        ), "wrong invariant features dimensions"

        ### stack s_t and f
        ### s_t ~ [b,2N,3]  f ~ [b,2N , d ] -- > x~[b,2N,d+3]
        x = torch.cat([s_t, f], dim=-1)
        x = self.input_layer(x)
        assert x.shape[2] == self.positional_encoding_dimension, x.shape

        x = self.hyena_layer1.forward(x)

        # Save the inputs for residual
        x_resid = x

        # Apply MLP
        x = self.mlp1(x)

        # Add residual
        x = x_resid + x

        x = self.hyena_layer2.forward(x)

        # Save the inputs for residual
        x_resid = x

        # Apply MLP
        x = self.mlp2(x)

        # Add residual
        x = x + x_resid
        x = self.hyena_layer3.forward(x)

        x = self.output_proj(x)

        N = x.shape[1]
        x_t = x[:, 0 : int(N / 2), :].permute(1, 0, 2)
        v_t = x[:, int(N / 2) : N, :].permute(1, 0, 2)
        return x_t, v_t


class nbody_Hyena(nn.Module):
    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension=16,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
    ):
        super().__init__()
        # Build the network
        self.sequence_length = sequence_length
        self.positional_encoding_dimension = positional_encoding_dimension
        self.token_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### positional encodings
        self.invariant_features = positional_encoding(
            self.sequence_length, self.positional_encoding_dimension, device=self.device
        ).to(self.device)

        ### token embeddings
        self.token_type_embedding = nn.Embedding(3, self.token_encoding_dimension).to(
            self.device
        )

        self.norm = nn.LayerNorm(
            self.positional_encoding_dimension + 3
        )  # Pre-attn LayerNorm

        ### se3 hyena operators
        self.hyena_layer1 = HyenaAttention(
            input_dimension=self.positional_encoding_dimension + 3,
            hidden_dimension=self.positional_encoding_dimension + 3,
            output_dimension=self.input_dimension_1,
            device=self.device,
        )

        self.mlp1 = nn.Linear(
            self.input_dimension_1,
            self.input_dimension_1,
        ).to(self.device)

        self.hyena_layer2 = HyenaAttention(
            input_dimension=self.input_dimension_1,
            hidden_dimension=self.input_dimension_1,
            output_dimension=self.input_dimension_2,
            device=self.device,
        )

        self.mlp2 = nn.Linear(self.input_dimension_2, self.input_dimension_2).to(
            self.device
        )

        self.hyena_layer3 = HyenaAttention(
            input_dimension=self.input_dimension_2,
            hidden_dimension=self.input_dimension_2,
            output_dimension=self.input_dimension_3,
            device=self.device,
        ).to(self.device)

        self.output_proj = nn.Linear(self.input_dimension_3, 3).to(self.device)

    def forward(self, G):
        # For a batched graph (DataBatch object)
        f_t, mask = to_dense_batch(
            G.x, G.batch
        )  # Node features, positions (x), tensor shape: [batch_size, num_nodes, feature_size]
        x_t = f_t[:, :, 0:3].to(self.device)  ### [b, N, 3]
        v_t = f_t[:, :, 3:6].to(self.device)  ### [b, N, 3]
        c = f_t[:, :, -1].to(self.device)  ### [ b, N ]

        ### stack n and v
        s_t = torch.cat([x_t, v_t], dim=1)
        batch_size, num_tokens, _ = s_t.shape
        assert s_t.shape[2] == 3, "dimension mismatch"

        ### stack charges: invariant features
        pos_enc = self.invariant_features.unsqueeze(0).repeat(s_t.shape[0], 1, 1)
        charge_enc = (
            torch.cat([c, c], dim=1)
            .unsqueeze(-1)
            .expand(-1, -1, self.positional_encoding_dimension)
        )  ### (b,2N,1) ### encoding here
        f = pos_enc + charge_enc

        ### now, apply the first SE(3)-Hyena Operator Layer
        assert s_t.shape[0] == f.shape[0], "batch sizes not equal"
        assert s_t.shape[1] == f.shape[1], "num tokens not equal"
        assert (
            f.shape[2] == self.positional_encoding_dimension
        ), "wrong invariant features dimensions"

        ### stack s_t and f

        ### s_t ~ [b,2N,3]
        ### f ~ [b,2N , d ] -- > x~[b,2N,d+3]
        x = torch.cat([s_t, f], dim=-1)
        assert x.shape[2] == self.positional_encoding_dimension + 3

        ### layer norm
        x = self.norm(x)
        x = self.hyena_layer1.forward(x)

        # Save the inputs for residual
        x_resid = x

        # Apply MLP
        x = self.mlp1(x)

        # Add residual
        x = x_resid + x

        x = self.hyena_layer2.forward(x)

        # Save the inputs for residual
        x_resid = x

        # Apply MLP
        x = self.mlp2(x)

        # Add residual
        x = x + x_resid
        x = self.hyena_layer3.forward(x)

        x = self.output_proj(x)

        N = x.shape[1]
        x_t = x[:, 0 : int(N / 2), :].permute(1, 0, 2)
        v_t = x[:, int(N / 2) : N, :].permute(1, 0, 2)
        return x_t, v_t


class nbody_SE3Hyenea(nn.Module):
    # """SE(3) equivariant Hyenea for nbody
    #
    # Intial positions and velocities are intput to equivariant branch
    # One hot encoded charges are input to invariant branch (with positional encodings? )
    # The SE3 hyena model consists of two SE3-hyena operators, followed by one output equivaraiant MLP
    # The gating uses hidden dimension of 8 for both equivariant and invariant features
    # We use adam optimizer with a learning rate set to 0.0001 and weight decay of 0.00001
    #
    #
    # """

    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension=16,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
    ):
        super().__init__()
        # Build the network
        self.sequence_length = sequence_length
        self.positional_encoding_dimension = positional_encoding_dimension
        self.token_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### positional encodings
        self.invariant_features = positional_encoding(
            self.sequence_length, self.positional_encoding_dimension, device=self.device
        ).to(self.device)

        ### token embeddings for charge
        self.token_type_embedding = nn.Embedding(3, self.token_encoding_dimension).to(
            self.device
        )

        ### se3 hyena operators
        self.hyena_layer1 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_inv_dimension=self.input_dimension_1,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.mlp1 = LinearProjection(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=1,
            output_inv_dimension=self.input_dimension_1,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.batch_norm_1 = BatchNormLayer(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=1,
        ).to(self.device)

        self.nonlinear_1 = NormActivationLayer(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=1,
        ).to(self.device)

        self.hyena_layer2 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=1,
            output_inv_dimension=self.input_dimension_2,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.mlp2 = LinearProjection(
            input_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=1,
            output_inv_dimension=self.input_dimension_2,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.hyena_layer3 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=1,
            output_inv_dimension=self.input_dimension_3,
            output_vector_multiplicity=1,
        ).to(self.device)

    def forward(self, G):
        # For a batched graph (DataBatch object)
        f_t, mask = to_dense_batch(
            G.x, G.batch
        )  # Node features, positions (x), tensor shape: [batch_size, num_nodes, feature_size]
        x_t = f_t[:, :, 0:3].to(self.device)  ### [b,N, 3]
        v_t = f_t[:, :, 3:6].to(self.device)  ### [b,N, 3]
        c = f_t[:, :, -1].to(self.device)  ### [ b,N ]

        ### stack n and v
        s_t = torch.cat([x_t, v_t], dim=1)
        batch_size, num_tokens, _ = s_t.shape

        ### positional and charge encodings
        pos_enc = self.invariant_features.unsqueeze(0).repeat(s_t.shape[0], 1, 1)
        charge_enc = (
            torch.cat([c, c], dim=1)
            .unsqueeze(-1)
            .expand(-1, -1, self.positional_encoding_dimension)
        )
        f = pos_enc + charge_enc

        s_t_resid = s_t
        s_t, cs = self.hyena_layer1.forward(s_t, f)
        s_t = s_t + s_t_resid

        s_t_resid = s_t
        s_t, cs = self.mlp1(s_t, cs)
        s_t, cs = self.batch_norm_1(s_t, cs)
        s_t, cs = self.nonlinear_1(s_t, cs)
        s_t = s_t + s_t_resid

        s_t_resid = s_t
        s_t, cs = self.hyena_layer2.forward(s_t, cs)
        s_t = s_t + s_t_resid

        s_t_resid = s_t
        s_t, cs = self.hyena_layer3.forward(s_t, cs)
        s_t = s_t + s_t_resid

        N = s_t.shape[1]
        x_t = s_t[:, 0 : int(N / 2), :].permute(1, 0, 2)
        v_t = s_t[:, int(N / 2) : N, :].permute(1, 0, 2)
        return x_t, v_t


class nbody_SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        num_degrees: int = 4,
        div: float = 4,
        n_heads: int = 1,
        si_m="1x1",
        si_e="att",
        x_ij="add",
    ):
        """
        Args:
            num_layers: number of attention layers
            num_channels: number of channels per degree
            num_degrees: number of degrees (aka types) in hidden layer, count start from type-0
            div: (int >= 1) keys, queries and values will have (num_channels/div) channels
            n_heads: (int >= 1) for multi-headed attention
            si_m: ['1x1', 'att'] type of self-interaction in hidden layers
            si_e: ['1x1', 'att'] type of self-interaction in final layer
            x_ij: ['add', 'cat'] use relative position as edge feature
        """
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

        self.fibers = {
            "in": Fiber(dictionary={1: 1}),
            "mid": Fiber(self.num_degrees, self.num_channels),
            "out": Fiber(dictionary={1: 2}),
        }

        self.Gblock = self._build_gcn(self.fibers)
        print(self.Gblock)

    def _build_gcn(self, fibers):
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
                    learnable_skip=True,
                    skip="cat",
                    selfint=self.si_m,
                    x_ij=self.x_ij,
                )
            )
            Gblock.append(GNormBias(fibers["mid"]))
            fin = fibers["mid"]
        Gblock.append(
            GSE3Res(
                fibers["mid"],
                fibers["out"],
                edge_dim=self.edge_dim,
                div=1,
                n_heads=min(self.n_heads, 2),
                learnable_skip=True,
                skip="cat",
                selfint=self.si_e,
                x_ij=self.x_ij,
            )
        )
        return nn.ModuleList(Gblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)
        h_enc = {"1": G.ndata["v"]}
        for layer in self.Gblock:
            h_enc = layer(h_enc, G=G, r=r, basis=basis)

        return h_enc["1"]


class nbody_TFN(nn.Module):
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
        print(self.block0)
        print(self.block1)
        print(self.block2)

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


### current setup: pass [x,v] as vectors and [c, positional encoding] as scalars
class nbody_GATr(nn.Module):
    """SE(3) equivariant Hyenea for nbody"""

    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension=16,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
        blocks=3,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_c = 1
        self.out_c = 1

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # This GATr's architecture is different from SE3Hyena
        self.gatr = GATr(
            in_mv_channels=self.in_c,
            out_mv_channels=self.out_c,
            hidden_mv_channels=input_dimension_3,  # GATr used 16
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=input_dimension_1,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        ).to(self.device)

    def forward(self, G):
        # For a batched graph (DataBatch object)
        f_t, mask = to_dense_batch(
            G.x, G.batch
        )  # Node features, positions (x), tensor shape: [batch_size, num_nodes, feature_size]

        # Build one multivector holding masses, points, and velocities for each object
        masses = f_t[:, :, -1:].to(self.device)  # (batchsize, objects, 1)
        masses = embed_scalar(masses)  # (batchsize, objects, 16)
        points = f_t[:, :, 0:3].to(self.device)  # (batchsize, objects, 3)
        points = embed_point(points)  # (batchsize, objects, 16)
        velocities = f_t[:, :, 3:6].to(self.device)  # (batchsize, objects, 3)
        velocities = embed_translation(velocities)  # (batchsize, objects, 16)
        multivector = masses + points + velocities  # (batchsize, objects, 16)

        # Insert channel dimension
        multivector = multivector.unsqueeze(2)  # (batchsize, objects, 1, 16)

        # Pass data through GATr
        # mask = BlockDiagonalMask.from_seqlens(torch.bincount(inputs.batch).tolist())
        embedded_outputs, _ = self.gatr(
            multivector, scalars=None
        )  # (batchsize, num_points, 1, 16)

        # Extract scalar and aggregate outputs from point cloud
        v_t = extract_translation(embedded_outputs).squeeze(
            2
        )  # (batchsize, num_points, 3)
        x_t = extract_point(embedded_outputs).squeeze(2)
        return x_t.reshape(-1, 1, 3), v_t.reshape(-1, 1, 3)
