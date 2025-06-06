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

from graspbench.layers import get_edge_feature, EdgeConv
from torch_geometric.data import Data, Batch
from escnn.group import SO3
from escnn.gspaces import no_base_space
from src.models import SE3HyenaOperator, StandardAttention, SE3HyperHyenaOperator
from src.projections import (
    TensorProductLayer,
    LinearProjection,
    NormActivationLayer,
    BatchNormLayer,
)
from src.utils import (
    normalize,
    construct_so3_frame_from_flat,
    axis_angle_to_matrix,
    vectors_to_so3,
    positional_encoding,
    skew_symmetric,
)


class SE3HyenaNormal(nn.Module):
    """SE(3) equivariant Hyena for ModelNet classification"""

    def __init__(
        self,
        sequence_length=1024,  # Max number of points
        positional_encoding_dimension=128,
        input_dimension_1=128,
        input_dimension_2=64,
        input_dimension_3=32,
        positional_encoding_type="pos_only",  ### pos_only, none
        kernel_size=3,
        scalar_attention_type="Standard",
    ):
        super(SE3HyenaNormal, self).__init__()

        self.sequence_length = sequence_length
        self.positional_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3

        self.positional_encoding_type = positional_encoding_type

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the group
        self.group = no_base_space(SO3())

        if self.positional_encoding_type != "pos_only":
            # Embedding layer to convert 3D coordinates to higher dimension (of invariant type)
            self.embedding = LinearProjection(
                input_inv_dimension=0,
                input_vector_multiplicity=1,
                output_inv_dimension=self.positional_encoding_dimension,
                output_vector_multiplicity=0,
            )

        # Hyena operators
        self.hyena_layer1 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=2 * input_dimension_1,
            hidden_vector_multiplicity=128,
            output_inv_dimension=self.input_dimension_1,
            output_vector_multiplicity=64,
        )

        self.mlp1 = LinearProjection(
            input_inv_dimension=input_dimension_1,
            input_vector_multiplicity=64,
            output_inv_dimension=input_dimension_1,
            output_vector_multiplicity=64,
        )
        # self.norm1 = HybridLayerNorm(
        #     feature_dim=input_dimension_1, group=self.group, device=self.device
        # )

        self.relu1 = TensorProductLayer(
            input_inv_dimension=input_dimension_1,
            input_vector_multiplicity=64,
            output_inv_dimension=input_dimension_1,
            output_vector_multiplicity=64,
        )

        # Second Hyena layer
        self.hyena_layer2 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=64,
            hidden_inv_dimension=2 * self.input_dimension_2,
            hidden_vector_multiplicity=32,
            output_inv_dimension=self.input_dimension_2,
            output_vector_multiplicity=32,
        )

        self.mlp2 = LinearProjection(
            input_inv_dimension=input_dimension_2,
            input_vector_multiplicity=32,
            output_inv_dimension=input_dimension_2,
            output_vector_multiplicity=32,
        )

        self.relu2 = TensorProductLayer(
            input_inv_dimension=input_dimension_2,
            input_vector_multiplicity=16,
            output_inv_dimension=input_dimension_2,
            output_vector_multiplicity=16,
        )

        # Third Hyena layer
        self.hyena_layer3 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=16,
            hidden_inv_dimension=2 * self.input_dimension_3,
            hidden_vector_multiplicity=16,
            output_inv_dimension=self.input_dimension_3,
            output_vector_multiplicity=8,
        )

        self.mlp3 = LinearProjection(
            input_inv_dimension=input_dimension_3,
            input_vector_multiplicity=8,
            output_inv_dimension=0,
            output_vector_multiplicity=1,
        )

        ### Initialize positional encodings
        self.register_buffer(
            "sinusoidal_pos_enc",
            positional_encoding(
                sequence_length, positional_encoding_dimension, self.device
            ),
        )

    def forward(self, x):
        # Get batch size and number of points
        batch_size, num_points, _ = x.shape

        ### Apply embedding to get initial features
        x = x.reshape(batch_size * num_points, 3)

        if self.positional_encoding_type == "pos_only":
            x = x.reshape(batch_size, num_points, 3)

            # Add positional encoding to node features
            if num_points <= self.sequence_length:
                pos_enc = (
                    self.sinusoidal_pos_enc[:num_points, :]
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
            else:
                pos_enc = positional_encoding(
                    num_points, self.positional_encoding_dimension, x.device
                )
                pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

            # Add positional encoding to features
            f = pos_enc

        else:
            f = self.embedding(
                x
            )  # [batch_size, max_points, positional_encoding_dimension]
            f = f.reshape(batch_size, num_points, self.positional_encoding_dimension)
            x = x.reshape(batch_size, num_points, 3)

            # Add positional encoding to node features
            if num_points <= self.sequence_length:
                pos_enc = (
                    self.sinusoidal_pos_enc[:num_points, :]
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
            else:
                pos_enc = positional_encoding(
                    num_points, self.positional_encoding_dimension, x.device
                )
                pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

            # Add positional encoding to features
            f = pos_enc + f

        x, f = self.hyena_layer1(x, f)

        # Save for residual and apply MLP
        x_resid, f_resid = x, f
        x, f = self.mlp1(x, f)
        x, f = self.relu1(x, f)
        x = x.reshape(batch_size, num_points, x.shape[-1])
        f = f.reshape(batch_size, num_points, f.shape[-1])
        x, f = x_resid + x, f_resid + f

        x, f = self.hyena_layer2.forward(x, f)

        x_resid, f_resid = x, f
        x, f = self.mlp2(x, f)
        x, f = self.relu2(x, f)
        x = x.reshape(batch_size, num_points, x.shape[-1])
        f = f.reshape(batch_size, num_points, f.shape[-1])
        x, f = x_resid + x, f_resid + f
        x, f = self.hyena_layer3.forward(x, f)
        x = F.normalize(x, p=2, dim=-1)  # L2 normalization along the last dimension
        return x


### first baseline
class StandardNormal(nn.Module):
    def __init__(
        self,
        sequence_length=1024,  # Max number of points
        positional_encoding_dimension=128,
        input_dimension_1=128,
        input_dimension_2=64,
        input_dimension_3=32,
        positional_encoding_type="pos_only",  ### pos_only, none
        kernel_size=3,
        scalar_attention_type="Standard",
    ):
        super(StandardNormal, self).__init__()
        self.positional_encoding_dimension = positional_encoding_dimension
        self.token_encoding_dimension = positional_encoding_dimension
        self.sequence_length = sequence_length

        self.input_dimension = self.positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### positional encodings
        self.invariant_features = positional_encoding(
            self.sequence_length, self.positional_encoding_dimension, device=self.device
        )

        ### token embeddings
        self.token_type_embedding = nn.Embedding(3, self.token_encoding_dimension).to(
            self.device
        )

        self.emb_position = nn.Linear(3, self.token_encoding_dimension).to(self.device)

        ### se3 hyena operators
        self.hyena_layer1 = StandardAttention(
            input_dimension=self.positional_encoding_dimension,
            output_dimension=self.input_dimension_1,
            device=self.device,
        )

        # MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dimension_1, input_dimension_1),
            nn.BatchNorm1d(input_dimension_1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hyena_layer2 = StandardAttention(
            input_dimension=self.input_dimension_1,
            output_dimension=self.input_dimension_2,
        ).to(self.device)

        # MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(input_dimension_2, input_dimension_2),
            nn.BatchNorm1d(input_dimension_2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hyena_layer3 = StandardAttention(
            input_dimension=self.input_dimension_2,
            output_dimension=self.input_dimension_3,
        ).to(self.device)

        # MLP
        self.mlp3 = nn.Sequential(
            nn.Linear(input_dimension_3, input_dimension_3),
            nn.BatchNorm1d(input_dimension_3),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        ### final output projection
        self.proj = nn.Linear(self.input_dimension_3, 3).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ### input sequence x~(b,N,3)
        batch_size, num_tokens, _ = x.shape
        ### Expand tensor to shape (b, N, d)
        f = self.invariant_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine with positional encodings
        x_emb = self.emb_position(x)
        f = f + x_emb
        f = self.hyena_layer1.forward(f)

        # Save the inputs for residual
        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp1(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        f = self.hyena_layer2.forward(f)

        # Save the inputs for residual
        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp2(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        ### now, apply the third SE(3)-Hyena Operator Layer
        f = self.hyena_layer3.forward(f)

        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp3(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        ### final projection
        f = self.proj(f)

        ### normilize to be unit vector: x ~ [b,N,3]
        f = F.normalize(f, p=2, dim=-1)  # L2 normalization along the last dimension

        return f


### first baseline
class StandardGrasp(nn.Module):
    def __init__(
        self,
        sequence_length=1024,  # Max number of points
        positional_encoding_dimension=128,
        input_dimension_1=128,
        input_dimension_2=64,
        input_dimension_3=32,
        use_normals: bool = True,  ### use the normals in making predictions
        positional_encoding_type="pos_only",  ### pos_only, none
        kernel_size=3,
        scalar_attention_type="Standard",
    ):
        super(StandardGrasp, self).__init__()
        self.positional_encoding_dimension = positional_encoding_dimension
        self.token_encoding_dimension = positional_encoding_dimension
        self.sequence_length = sequence_length
        self.use_normals = use_normals

        self.input_dimension = self.positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### positional encodings
        self.invariant_features = positional_encoding(
            self.sequence_length, self.positional_encoding_dimension, device=self.device
        )

        ### token embeddings
        if self.use_normals == True:
            self.token_type_embedding = nn.Embedding(
                6, self.token_encoding_dimension
            ).to(self.device)
        else:
            self.token_type_embedding = nn.Embedding(
                3, self.token_encoding_dimension
            ).to(self.device)

        if self.use_normals == True:
            self.emb_position = nn.Linear(6, self.token_encoding_dimension).to(
                self.device
            )
        else:
            self.emb_position = nn.Linear(3, self.token_encoding_dimension).to(
                self.device
            )

        ### se3 hyena operators
        self.hyena_layer1 = StandardAttention(
            input_dimension=self.positional_encoding_dimension,
            output_dimension=self.input_dimension_1,
        ).to(self.device)

        # MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dimension_1, input_dimension_1),
            nn.BatchNorm1d(input_dimension_1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hyena_layer2 = StandardAttention(
            input_dimension=self.input_dimension_1,
            output_dimension=self.input_dimension_2,
        ).to(self.device)

        # MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(input_dimension_2, input_dimension_2),
            nn.BatchNorm1d(input_dimension_2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hyena_layer3 = StandardAttention(
            input_dimension=self.input_dimension_2,
            output_dimension=self.input_dimension_3,
        ).to(self.device)

        # MLP
        self.mlp3 = nn.Sequential(
            nn.Linear(input_dimension_3, input_dimension_3),
            nn.BatchNorm1d(input_dimension_3),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        ### compute prob. density on nodes ~ [b,N,1]
        self.mlp_density = nn.Sequential(
            nn.Linear(input_dimension_3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        ### compute a rotation via axis-angle param: ~[b,3] ---convert--> [b,SO(3)] via axis-angle
        self.mlp_rot = nn.Sequential(
            nn.Linear(input_dimension_3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )

        ### compute the depth estimation ~[b,1]
        self.mlp_depth = nn.Sequential(
            nn.Linear(input_dimension_3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, n: torch.tensor) -> torch.Tensor:
        if self.use_normals == True:
            x = torch.cat([x, n], dim=-1)

        batch_size, num_tokens, _ = x.shape
        f = self.invariant_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine with positional encodings
        x_emb = self.emb_position(x)
        f = f + x_emb
        f = self.hyena_layer1.forward(f)

        # Save the inputs for residual
        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp1(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        f = self.hyena_layer2.forward(f)

        # Save the inputs for residual
        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp2(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        ### now, apply the third SE(3)-Hyena Operator Layer
        f = self.hyena_layer3.forward(f)

        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp3(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        ### final projection to prob dist
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f_density = self.mlp_density(f)
        f_density = f_density.reshape(batch_size, num_tokens, 1)
        f_density = F.softmax(f_density, dim=1)  ### ~ [ b, N, 3]

        ### max pool over N dimension:
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = torch.mean(f, dim=1)  ### [b, d]

        axis_angle = self.mlp_rot(f)
        rot = axis_angle_to_matrix(axis_angle)  ### [b,3,3]

        dist = self.mlp_depth(f)  ### [b,1]
        return f_density.squeeze(-1), rot, dist.squeeze(-1)


class SE3HyenaGrasp(nn.Module):
    """SE(3) equivariant Hyena for ModelNet classification"""

    def __init__(
        self,
        sequence_length=1024,  # Max number of points
        positional_encoding_dimension=128,
        input_dimension_1=128,
        input_dimension_2=64,
        input_dimension_3=32,
        use_normals: bool = True,
        positional_encoding_type="pos_only",  ### pos_only, none
        kernel_size=3,
        scalar_attention_type="Standard",
    ):
        super(SE3HyenaGrasp, self).__init__()

        self.sequence_length = sequence_length
        self.positional_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3
        self.use_normals = use_normals
        self.positional_encoding_type = positional_encoding_type

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the group
        self.group = no_base_space(SO3())

        if self.positional_encoding_type != "pos_only":
            # Embedding layer to convert 3D coordinates to higher dimension (of invariant type)
            self.embedding = LinearProjection(
                input_inv_dimension=0,
                input_vector_multiplicity=1,
                output_vector_multiplicity=0,
                output_inv_dimension=positional_encoding_dimension,
            ).to(self.device)

        if self.use_normals:
            # Hyena operators
            self.hyena_layer1 = SE3HyenaOperator(
                input_inv_dimension=self.positional_encoding_dimension,
                input_vector_multiplicity=2,
                hidden_inv_dimension=2 * input_dimension_1,
                hidden_vector_multiplicity=5,
                output_inv_dimension=self.input_dimension_1,  ### invariant dim can change
                output_vector_multiplicity=3,
                group=self.group,
                device=self.device,
            )
        else:
            self.hyena_layer1 = SE3HyenaOperator(
                input_inv_dimension=self.positional_encoding_dimension,
                input_vector_multiplicity=1,
                hidden_inv_dimension=2 * input_dimension_1,
                hidden_vector_multiplicity=5,
                output_inv_dimension=self.input_dimension_1,  ### invariant dim can change
                output_vector_multiplicity=3,
                group=self.group,
                device=self.device,
            )

        self.mlp1 = LinearProjection(
            input_inv_dimension=self.input_dimension_1,
            output_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=3,
            output_vector_multiplicity=3,
        ).to(self.device)

        self.norm1 = BatchNormLayer(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=3,
        ).to(self.device)
        self.relu1 = NormActivationLayer(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=3,
        ).to(self.device)

        # Second Hyena layer
        self.hyena_layer2 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_1,
            input_vector_multiplicity=3,
            hidden_inv_dimension=2 * self.input_dimension_2,
            hidden_vector_multiplicity=5,
            output_inv_dimension=self.input_dimension_2,
            output_vector_multiplicity=5,
        ).to(self.device)

        self.mlp2 = LinearProjection(
            input_inv_dimension=self.input_dimension_2,
            output_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=5,
            output_vector_multiplicity=5,
        ).to(self.device)

        self.norm1 = BatchNormLayer(
            input_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=5,
        ).to(self.device)
        self.relu1 = NormActivationLayer(
            input_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=5,
        ).to(self.device)

        # Third Hyena layer
        self.hyena_layer3 = SE3HyenaOperator(
            input_inv_dimension=self.input_dimension_2,
            input_vector_multiplicity=5,
            hidden_inv_dimension=2 * self.input_dimension_3,
            hidden_vector_multiplicity=7,
            output_vector_multiplicity=5,
            output_inv_dimension=self.input_dimension_3,
            group=self.group,
            device=self.device,
        ).to(self.device)

        ### compute prob. density on nodes ~ [b,N,1]
        self.mlp_density = LinearProjection(
            input_inv_dimension=self.input_dimension_3,
            output_inv_dimension=1,
            input_vector_multiplicity=5,
            output_vector_multiplicity=0,
        ).to(self.device)

        ### compute a rotation via axis-angle param: ~[b,3] ---convert--> [b,SO(3)] via axis-angle
        self.mlp_rot = LinearProjection(
            input_inv_dimension=self.input_dimension_3,
            output_inv_dimension=0,
            input_vector_multiplicity=5,
            output_vector_multiplicity=3,  ### i.e. 3 sets of 3 vectors --> 1 nine dim 1 rep
        ).to(self.device)

        ### compute the depth estimation ~[b,1]
        self.mlp_depth = LinearProjection(
            input_inv_dimension=self.input_dimension_3,
            output_inv_dimension=1,
            input_vector_multiplicity=5,
            output_vector_multiplicity=0,
        ).to(self.device)

        ### Initialize positional encodings
        self.register_buffer(
            "sinusoidal_pos_enc",
            positional_encoding(
                sequence_length, positional_encoding_dimension, self.device
            ),
        )

    def forward(self, x, n):
        # Get batch size and number of points
        batch_size, num_points, _ = x.shape

        if self.use_normals == True:
            x = torch.cat([x, n], dim=-1)

        ### Apply embedding to get initial features
        x = x.reshape(batch_size * num_points, x.shape[-1])

        if self.positional_encoding_type == "pos_only":
            x = x.reshape(batch_size, num_points, x.shape[-1])

            # Add positional encoding to node features
            if num_points <= self.sequence_length:
                pos_enc = (
                    self.sinusoidal_pos_enc[:num_points, :]
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
            else:
                pos_enc = positional_encoding(
                    num_points, self.positional_encoding_dimension, x.device
                )
                pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

            # Add positional encoding to features
            f = pos_enc

        else:
            f = self.embedding(
                x
            )  # [batch_size, max_points, positional_encoding_dimension]
            f = f.reshape(batch_size, num_points, self.positional_encoding_dimension)
            x = x.reshape(batch_size, num_points, x.shape[-1])

            # Add positional encoding to node features
            if num_points <= self.sequence_length:
                pos_enc = (
                    self.sinusoidal_pos_enc[:num_points, :]
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
            else:
                pos_enc = positional_encoding(
                    num_points, self.positional_encoding_dimension, x.device
                )
                pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

            # Add positional encoding to features
            f = pos_enc + f

        # First Hyena layer - now correctly passing both x and f
        x, f = self.hyena_layer1.forward(x, f)

        # Save for residual and apply MLP
        x_resid, f_resid = x, f
        x, f = self.mlp1(x, f)
        x, f = self.relu1(x, f)
        x = x.reshape(batch_size, num_points, x.shape[-1])
        f = f.reshape(batch_size, num_points, f.shape[-1])
        x, f = x_resid + x, f_resid + f  # Residual connection

        ### Second Hyena layer
        x, f = self.hyena_layer2.forward(x, f)

        # Save for residual and apply MLP
        x_resid, f_resid = x, f

        x, f = self.mlp2(x, f)
        x, f = self.relu2(x, f)
        x = x.reshape(batch_size, num_points, x.shape[-1])
        f = f.reshape(
            batch_size, num_points, f.shape[-1]
        )  # [batch_size, num_points, input_dimension_1]
        x, f = x_resid + x, f_resid + f  # Residual connection

        # Third Hyena layer
        x, f = self.hyena_layer3.forward(x, f)

        _, f_density = self.mlp_density(x, f)
        f_density = f_density.reshape(batch_size, num_points, 1)
        f_density = F.softmax(f_density, dim=1)  ### ~ [ b, N, 3]

        x = x.reshape(batch_size, num_points, x.shape[-1])
        x = torch.mean(x, dim=1)  ### [b, d]

        f = f.reshape(batch_size, num_points, f.shape[-1])
        f = torch.mean(f, dim=1)  ### [b, d]

        ### compute a rotation via axis-angle param: ~[b,3] ---convert--> [b,SO(3)] via axis-angle
        f_rot, _ = self.mlp_rot(x, f)

        ### convert [b,9]-->[b,3,3]
        f_rot = construct_so3_frame_from_flat(f_rot)

        ### compute the depth estimation ~[b,1]
        _, f_dist = self.mlp_depth(x, f)

        return f_density.squeeze(-1), f_rot, f_dist.squeeze(-1)


### DGCNN baseline
class NormalDGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5):
        super(NormalDGCNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Edge convolution layers
        self.edge_conv1 = EdgeConv(3, 64, k=k)
        self.edge_conv2 = EdgeConv(64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, k=k)

        # output MLP
        self.mlp = nn.Sequential(
            nn.Linear(512, emb_dims),
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
            nn.Linear(256, 3),
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        # Apply edge convolutions
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        # Concatenate and mlp
        x = torch.cat([x1, x2, x3, x4], dim=2)
        x = x.reshape(batch_size * num_tokens, x.shape[-1])
        x = self.mlp(x)
        x = x.reshape(batch_size, num_tokens, x.shape[-1])

        return x


class SE3HyperHyenaNormal(nn.Module):
    """SE(3) equivariant HyperHyena for ModelNet classification"""

    def __init__(
        self,
        sequence_length=1024,  # Max number of points
        positional_encoding_dimension=128,
        input_dimension_1=128,
        input_dimension_2=64,
        input_dimension_3=32,
        positional_encoding_type="pos_only",  ### pos_only, none
        kernel_size=3,
        scalar_attention_type="Standard",
    ):
        super(SE3HyperHyenaNormal, self).__init__()

        self.sequence_length = sequence_length
        self.positional_encoding_dimension = positional_encoding_dimension
        self.input_dimension_1 = input_dimension_1
        self.input_dimension_2 = input_dimension_2
        self.input_dimension_3 = input_dimension_3
        self.positional_encoding_type = positional_encoding_type

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the group
        self.group = no_base_space(SO3())

        if self.positional_encoding_type != "pos_only":
            # Embedding layer to convert 3D coordinates to higher dimension (of invariant type)
            self.embedding = InvariantPointCloudEmbedding(
                embedding_dimension=positional_encoding_dimension,
                group=self.group,
                device=self.device,
            ).to(self.device)

        # Hyena operators
        self.hyena_layer1 = SE3HyperHyenaOperator(
            input_multiplicities=[positional_encoding_dimension, 1],
            hidden_multiplicities=[2 * input_dimension_1, 3, 1],
            output_multiplicities=[input_dimension_1, 1],
            group=self.group,
            device=self.device,
        ).to(self.device)

        self.mlp1 = HybridProjection(
            feature_dim=input_dimension_1,
            output_dim=input_dimension_1,
            group=self.group,
            device=self.device,
        )
        self.norm1 = HybridLayerNorm(
            feature_dim=input_dimension_1, group=self.group, device=self.device
        )
        self.relu1 = HybridRelu(
            feature_dim=input_dimension_1, group=self.group, device=self.device
        )

        # Second Hyena layer
        self.hyena_layer2 = SE3HyperHyenaOperator(
            input_multiplicities=[input_dimension_1, 1],
            hidden_multiplicities=[2 * input_dimension_2, 5, 3],
            output_multiplicities=[input_dimension_2, 1],
            group=self.group,
            device=self.device,
        ).to(self.device)

        self.mlp2 = HybridProjection(
            feature_dim=input_dimension_2,
            output_dim=input_dimension_2,
            group=self.group,
            device=self.device,
        )
        self.norm2 = HybridLayerNorm(
            feature_dim=input_dimension_2,
            group=self.group,
            device=self.device,
        )
        self.relu2 = HybridRelu(
            feature_dim=input_dimension_2, group=self.group, device=self.device
        )

        # Third Hyena layer
        self.hyena_layer3 = SE3HyperHyenaOperator(
            input_multiplicities=[input_dimension_2, 1],
            hidden_multiplicities=[2 * input_dimension_3, 7, 5, 3],
            output_multiplicities=[input_dimension_3, 1],
            group=self.group,
            device=self.device,
        ).to(self.device)

        self.mlp3 = InvariantProjection(
            feature_dim=input_dimension_3,
            output_dim=input_dimension_3,
            group=self.group,
            device=self.device,
        )
        self.relu3 = nn.ReLU()

        ### Initialize positional encodings
        self.register_buffer(
            "sinusoidal_pos_enc",
            positional_encoding(
                sequence_length, positional_encoding_dimension, self.device
            ),
        )

    def forward(self, x):
        # Get batch size and number of points
        batch_size, num_points, _ = x.shape

        ### Apply embedding to get initial features
        x = x.reshape(batch_size * num_points, 3)

        if self.positional_encoding_type == "pos_only":
            x = x.reshape(batch_size, num_points, 3)

            # Add positional encoding to node features
            if num_points <= self.sequence_length:
                pos_enc = (
                    self.sinusoidal_pos_enc[:num_points, :]
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
            else:
                pos_enc = positional_encoding(
                    num_points, self.positional_encoding_dimension, x.device
                )
                pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

            # Add positional encoding to features
            f = pos_enc

        else:
            f = self.embedding(
                x
            )  # [batch_size, max_points, positional_encoding_dimension]
            f = f.reshape(batch_size, num_points, self.positional_encoding_dimension)
            x = x.reshape(batch_size, num_points, 3)

            # Add positional encoding to node features
            if num_points <= self.sequence_length:
                pos_enc = (
                    self.sinusoidal_pos_enc[:num_points, :]
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
            else:
                pos_enc = positional_encoding(
                    num_points, self.positional_encoding_dimension, x.device
                )
                pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

            # Add positional encoding to features
            f = pos_enc + f

        # First Hyena layer - now correctly passing both x and f
        xf = torch.cat([x, f], dim=-1)
        xf = self.hyena_layer1.forward(xf)
        f, x = xf[:, :, 0:-3], xf[:, :, -3:]

        # Save for residual and apply MLP
        x_resid, f_resid = x, f
        x, f = self.mlp1(x, f)
        x, f = self.relu1(x, f)
        x = x.reshape(batch_size, num_points, x.shape[-1])
        f = f.reshape(batch_size, num_points, f.shape[-1])
        x, f = x_resid + x, f_resid + f  # Residual connection

        xf = torch.cat([x, f], dim=-1)
        xf = self.hyena_layer2.forward(xf)
        f, x = xf[:, :, 0:-3], xf[:, :, -3:]

        # Save for residual and apply MLP
        x_resid, f_resid = x, f
        x, f = self.mlp2(x, f)
        x, f = self.relu2(x, f)
        x = x.reshape(batch_size, num_points, x.shape[-1])
        f = f.reshape(
            batch_size, num_points, f.shape[-1]
        )  # [batch_size, num_points, input_dimension_1]
        x, f = x_resid + x, f_resid + f  # Residual connection

        xf = torch.cat([x, f], dim=-1)
        xf = self.hyena_layer3.forward(xf)
        f, x = xf[:, :, 0:-3], xf[:, :, -3:]

        ## normilize to be unit vector: x ~ [b,N,3]
        x = F.normalize(x, p=2, dim=-1)  # L2 normalization along the last dimension

        return x


class GraspDGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, use_normals=True, dropout=0.5):
        super(GraspDGCNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_normals = use_normals

        # Edge convolution layers
        if self.use_normals:
            self.edge_conv1 = EdgeConv(6, 64, k=k)
        else:
            self.edge_conv1 = EdgeConv(3, 64, k=k)
        self.edge_conv2 = EdgeConv(64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, k=k)

        # feature output MLP
        self.mlp = nn.Sequential(
            nn.Linear(512, emb_dims),
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
            nn.Linear(256, 256),
        )

        ### compute prob. density on nodes ~ [b,N,1]
        self.mlp_density = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        ### compute a rotation via axis-angle param: ~[b,3] ---convert--> [b,SO(3)] via axis-angle
        self.mlp_rot = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
        )

        ### compute the depth estimation ~[b,1]
        self.mlp_depth = nn.Sequential(
            nn.Linear(256, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x, n):
        batch_size, num_tokens, _ = x.shape
        if self.use_normals:
            x = torch.cat([x, n], dim=-1)

        # Apply edge convolutions
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        # Concatenate and mlp
        x = torch.cat([x1, x2, x3, x4], dim=2)
        x = x.reshape(batch_size * num_tokens, x.shape[-1])
        x = self.mlp(x)

        x_density = self.mlp_density(x)
        x_density = x_density.reshape(batch_size, num_tokens, x_density.shape[-1])
        x_density = F.softmax(x_density, dim=1)  ### ~ [ b, N, 1]

        x = x.reshape(batch_size, num_tokens, x.shape[-1])
        x = torch.mean(x, dim=1)
        x_rot = self.mlp_rot(x)
        x_rot = axis_angle_to_matrix(x_rot)  ### [b,3,3]

        x_depth = self.mlp_depth(x)

        return x_density.squeeze(-1), x_rot, x_depth.squeeze(-1)
