import pytest
import torch
from fixtures import mock_data, so3_group, mock_dgl_graph_with_edges, mock_ridataset
from escnn.group import SO3
from escnn.gspaces import no_base_space
import escnn
import sys
import os
from copy import deepcopy
import dgl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.models import SE3Transformer, TFN
from src.utils import random_rotation_matrix


def test_TFN(mock_ridataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TFN(
        num_layers=1,
        num_channels=8,
        num_degrees=4,
    ).to(device)

    for i in range(2):
        G, x_T, v_T = mock_ridataset[i]
        assert isinstance(G, dgl.DGLGraph)

        # Set required node features
        G.ndata["c"] = torch.ones(G.num_nodes(), 1)  # Scalar features [N, 1]
        G.ndata["v"] = torch.cat(
            [G.ndata["x"], G.ndata["v"]], dim=1
        )  # Vector features [N, 6]

        # Set required edge features
        src, dst = G.edges()
        G.edata["d"] = G.ndata["x"][dst] - G.ndata["x"][src]  # [E, 3]
        G.edata["r"] = torch.sqrt(
            torch.sum(G.edata["d"] ** 2, -1, keepdim=True)
        )  # [E, 1]
        G.edata["w"] = G.edata["w"].unsqueeze(-1)  # [E, 1] -> [E, 1, 1]

        output = model(G)

        vel = output["1"]
        assert vel.shape[1] == 3, "Wrong shape for vector output"


def test_se3transformer(mock_ridataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SE3Transformer(
        input_invariant_multplicity=0,
        input_vector_multiplicity=1,
        num_layers=1,
        num_channels=8,
        num_degrees=4,
        div=4,  ### div must divide num_channels
        n_heads=1,
        si_m="1x1",
        si_e="att",
        x_ij="add",
    ).to(device)

    for i in range(2):
        G, x_T, v_T = mock_ridataset[i]
        assert isinstance(G, dgl.DGLGraph)

        # Set required node features
        G.ndata["0"] = torch.ones(G.num_nodes(), 1)  # Scalar features [N, 1]
        G.ndata["1"] = torch.cat(
            [G.ndata["x"], G.ndata["v"]], dim=1
        )  # Vector features [N, 6]

        # Set required edge features
        src, dst = G.edges()
        G.edata["d"] = G.ndata["x"][dst] - G.ndata["x"][src]  # [E, 3]
        G.edata["r"] = torch.sqrt(
            torch.sum(G.edata["d"] ** 2, -1, keepdim=True)
        )  # [E, 1]
        G.edata["w"] = G.edata["w"].unsqueeze(-1)  # [E, 1] -> [E, 1, 1]

        output = model(G)

        vel = output["1"]
        assert vel.shape[1] == 3, "Wrong shape for vector output"


# def test_se3transformer_equivariant(mock_dgl_graph_with_edges):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     G = mock_dgl_graph_with_edges.to(device)
#     G_rot = deepcopy(G).to(device)

#     model = SE3Transformer(
#         input_invariant_multplicity=0,
#         input_vector_multiplicity=1,
#         num_layers=1,
#         num_channels=1,
#         num_degrees=4,
#         div=1,  ### div must divide num_channels
#         n_heads=1,
#         si_m="1x1",
#         si_e="att",
#         x_ij="add",
#     ).to(device)

#     ### now, apply rotation to graph G
#     # Apply random rotation
#     R = random_rotation_matrix()  # 3x3

#     # Rotate vector-valued node features
#     for key in ["x", "v", "d"]:
#         if key in G_rot.ndata:
#             G_rot.ndata[key] = G_rot.ndata[key] @ R.T

#     # Rotate edge features
#     if "d" in G_rot.edata:
#         G_rot.edata["d"] = G_rot.edata["d"] @ R.T

#     # Update edge distances after rotating node positions
#     src, dst = G_rot.edges()
#     G_rot.edata["r"] = (G_rot.ndata["x"][dst] - G_rot.ndata["x"][src]).norm(
#         dim=1, keepdim=True
#     )

#     output = model(G)
#     rot_output = model(G)
