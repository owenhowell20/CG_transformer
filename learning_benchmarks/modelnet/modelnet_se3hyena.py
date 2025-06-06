import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import knn_graph

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, parent_dir)

from torch_geometric.data import Data, Batch
from escnn.group import SO3
from escnn.gspaces import no_base_space
from src.models import SE3HyenaOperator
from src.projections import (
    LinearProjection,
    NormActivationLayer,
    BatchNormLayer,
    TensorProductLayer,
    EquivariantGating,
)
from src.utils import positional_encoding


class ModelNetSE3Hyena(nn.Module):
    """SE(3) equivariant Hyena for ModelNet classification"""

    def __init__(
        self,
        num_classes=40,  # Default to ModelNet40
        sequence_length=1024,  # Max number of points
        positional_encoding_dimension=64,
        input_dimension_1=128,
        input_dimension_2=128,
        input_dimension_3=128,
        positional_encoding_type="pos_and_learn",  ### pos_only, none
        kernel_size=3,
        scalar_attention_type="Standard",
        vector_attention_type="FFT",
    ):
        super(ModelNetSE3Hyena, self).__init__()

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.positional_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3

        self.positional_encoding_type = positional_encoding_type

        mlp_input_1 = 64
        mlp_input_2 = 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ### Embedding layer to convert 3D coordinates to higher dimension (of invariant type)
        # self.learned_positional_embedding = LinearProjection(
        #     input_inv_dimension=1,
        #     input_vector_multiplicity=1,
        #     output_inv_dimension=self.positional_encoding_dimension,
        #     output_vector_multiplicity=1,
        # ).to(self.device)

        self.hyena_layer1 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=input_dimension_1,
            hidden_vector_multiplicity=3,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
            scalar_attention_type=scalar_attention_type,
            vector_attention_type=vector_attention_type,
        ).to(self.device)

        self.mlp1 = LinearProjection(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.norm1 = BatchNormLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        ).to(self.device)

        self.relu1 = NormActivationLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        ).to(self.device)

        self.hyena_layer2 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=input_dimension_2,
            hidden_vector_multiplicity=3,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
            scalar_attention_type=scalar_attention_type,
            vector_attention_type=vector_attention_type,
        ).to(self.device)

        self.mlp2 = LinearProjection(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.norm2 = BatchNormLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        ).to(self.device)

        self.relu2 = NormActivationLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        ).to(self.device)

        self.hyena_layer3 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=self.input_dimension_3,
            hidden_vector_multiplicity=5,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
            scalar_attention_type=scalar_attention_type,
            vector_attention_type=vector_attention_type,
        ).to(self.device)

        self.mlp3 = LinearProjection(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=0,
        ).to(self.device)

        ### Final MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(self.positional_encoding_dimension, mlp_input_1),
            nn.BatchNorm1d(mlp_input_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_input_1, mlp_input_2),
            nn.BatchNorm1d(mlp_input_2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_input_2, num_classes),
        ).to(self.device)

        # Initialize positional encodings
        self.register_buffer(
            "sinusoidal_pos_enc",
            positional_encoding(
                sequence_length, positional_encoding_dimension, self.device
            ),
        )

    def forward(self, G):
        # For a batched graph
        batch_size = G.batch.max().item() + 1

        ### center each graph
        graphs = G.to_data_list()
        for i, graph in enumerate(graphs):
            graph.pos = graph.pos - graph.pos.mean(dim=0, keepdim=True)
        G = Batch.from_data_list(graphs)

        # Get points and batch information
        pos = G.pos  # [total_points, 3]
        batch = G.batch  # [total_points]

        # Convert to dense batch for sequence processing
        x, mask = to_dense_batch(pos, batch)  # [batch_size, max_points, 3]

        # Get batch size and number of points
        batch_size, num_points, _ = x.shape
        f = torch.zeros(
            batch_size, num_points, self.positional_encoding_dimension, device=x.device
        )

        x_res, f_res = x, f
        x, f = self.hyena_layer1.forward(x, f)
        x, f = x + x_res, f + f_res

        x_resid, f_resid = x, f
        x, f = self.mlp1(x, f)
        x, f = self.norm1(x, f)
        x, f = self.relu1(x, f)
        x, f = x_resid + x, f_resid + f

        x_res, f_res = x, f
        x, f = self.hyena_layer2.forward(x, f)
        x, f = x + x_res, f + f_res

        x_resid, f_resid = x, f
        x, f = self.mlp2(x, f)
        x, f = self.norm2(x, f)
        x, f = self.relu2(x, f)
        x, f = x_resid + x, f_resid + f

        x_res, f_res = x, f
        x, f = self.hyena_layer3.forward(x, f)
        x, f = x + x_res, f + f_res

        _, f = self.mlp3(x, f)

        # Global max pooling over points
        f = f.max(dim=1)[0]  ### [batch_size, input_dimension_3]

        # Classification head
        logits = self.classifier(f)
        return logits
