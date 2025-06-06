import torch
from torch import nn
import math
from typing import List


from src.layers import (
    RegularTensorProductLayer,
    RegularInputConversion,
    RegularOutputConversion,
    RegularNormActivation,
    RegularLinearProjection,
    RegularBatchNorm,
)
from src.models import SE3HyenaOperator, StandardAttention, SE3HyperHyenaOperator
from src.models import HyenaAttention as Hyena_Operator
from src.projections import LinearProjection, NormActivationLayer, BatchNormLayer

from src.utils import positional_encoding


class GRDSE3HyperHyena(nn.Module):
    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension: int = 64,
        input_multiplicity_1: int = 64,
        input_harmonic_1: int = 1,
        hidden_multiplicity_1: int = 10,
        hidden_harmonic_1: int = 5,
        hidden_multiplicity_2: int = 10,
        hidden_harmonic_2: int = 5,
        hidden_multiplicity_3: int = 10,
        hidden_harmonic_3: int = 3,
    ):
        super(GRDSE3HyperHyena, self).__init__()

        self.positional_encoding_dimension = positional_encoding_dimension
        self.sequence_length = sequence_length

        # Set device here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### token embeddings
        self.token_type_embedding = nn.Embedding(
            3, self.positional_encoding_dimension
        ).to(self.device)

        ### positional encodings
        self.invariant_features = positional_encoding(
            self.sequence_length, self.positional_encoding_dimension, device=self.device
        )

        self.input_proj = RegularInputConversion(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_multiplicity=input_multiplicity_1,
            output_max_harmonic=1,
        ).to(self.device)

        self.hyena_layer1 = SE3HyperHyenaOperator(
            input_multiplicity=input_multiplicity_1,
            input_max_harmonic=1,
            hidden_multiplicity=hidden_multiplicity_1,
            hidden_max_harmonic=hidden_harmonic_1,
            output_multiplicity=input_multiplicity_1,
            output_max_harmonic=1,
        ).to(self.device)

        # self.tensor_product_1 = RegularTensorProductLayer(
        #     input_max_harmonic=1,
        #     input_multiplicity=input_multiplicity_1,
        #     output_multiplicity=input_multiplicity_1,
        #     output_max_harmonic=1,
        # ).to(self.device)

        self.mlp1 = RegularLinearProjection(
            input_max_harmonic=1,
            input_multiplicity=input_multiplicity_1,
            output_multiplicity=input_multiplicity_1,
            output_max_harmonic=1,
        ).to(self.device)

        self.batch_norm1 = RegularBatchNorm(
            input_multiplicity=input_multiplicity_1, input_max_harmonic=1
        ).to(self.device)

        self.norm_activation_1 = RegularNormActivation(
            input_max_harmonic=1,
            input_multiplicity=input_multiplicity_1,
        ).to(self.device)

        self.hyena_layer2 = SE3HyperHyenaOperator(
            input_multiplicity=input_multiplicity_1,
            input_max_harmonic=1,
            hidden_multiplicity=hidden_multiplicity_2,
            hidden_max_harmonic=hidden_harmonic_2,
            output_multiplicity=input_multiplicity_1,
            output_max_harmonic=1,
        ).to(self.device)

        # self.tensor_product_2 = RegularTensorProductLayer(
        #     input_max_harmonic=1,
        #     input_multiplicity=input_multiplicity_1,
        #     output_multiplicity=input_multiplicity_1,
        #     output_max_harmonic=1,
        # ).to(self.device)

        self.mlp2 = RegularLinearProjection(
            input_max_harmonic=1,
            input_multiplicity=input_multiplicity_1,
            output_multiplicity=input_multiplicity_1,
            output_max_harmonic=1,
        ).to(self.device)

        self.batch_norm2 = RegularBatchNorm(
            input_multiplicity=input_multiplicity_1, input_max_harmonic=1
        ).to(self.device)

        self.norm_activation_2 = RegularNormActivation(
            input_max_harmonic=input_harmonic_1,
            input_multiplicity=1,
        ).to(self.device)

        self.hyena_layer3 = SE3HyperHyenaOperator(
            input_multiplicity=1,
            input_max_harmonic=input_harmonic_1,
            hidden_multiplicity=hidden_multiplicity_3,
            hidden_max_harmonic=hidden_harmonic_3,
            output_multiplicity=input_multiplicity_1,
            output_max_harmonic=1,
        ).to(self.device)

        # self.tensor_product_3 = RegularTensorProductLayer(
        #     input_max_harmonic=1,
        #     input_multiplicity=input_multiplicity_1,
        #     output_multiplicity=input_multiplicity_1,
        #     output_max_harmonic=1,
        # ).to(self.device)

        self.mlp3 = RegularLinearProjection(
            input_max_harmonic=1,
            input_multiplicity=input_multiplicity_1,
            output_multiplicity=input_multiplicity_1,
            output_max_harmonic=1,
        ).to(self.device)

        self.batch_norm3 = RegularBatchNorm(
            input_multiplicity=input_multiplicity_1, input_max_harmonic=1
        ).to(self.device)

        self.norm_activation_3 = RegularNormActivation(
            input_max_harmonic=1,
            input_multiplicity=input_multiplicity_1,
        ).to(self.device)

        self.proj = RegularOutputConversion(
            input_multiplicity=input_multiplicity_1,
            input_max_harmonic=input_harmonic_1,
            output_inv_dimension=0,
            output_vector_multiplicity=1,
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape

        f = self.invariant_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Create token type indices: 0 for keys, 1 for values, 2 for question
        token_types = torch.zeros(
            (batch_size, num_tokens), dtype=torch.long, device=x.device
        )
        token_types[:, 0::2] = 0  # key
        token_types[:, 1::2] = 1  # value
        token_types[:, -1] = 2  # question (last token)

        # Get token-type embeddings
        token_type_embed = self.token_type_embedding(token_types)  # (B, N, D)

        # Combine with positional encodings
        f = f + token_type_embed.to(self.device)

        x = self.input_proj(x, f)

        ### now, apply the first SE(3)-Hyena Operator Layer
        x = x + self.hyena_layer1.forward(x)

        # Save the inputs for residual
        x_resid = x
        # x_out = self.tensor_product_1(x)
        x_out = self.mlp1(x)
        x_out = self.batch_norm1(x_out)
        x_out = self.norm_activation_1(x_out)
        x_out = x_out.reshape(batch_size, num_tokens, x_out.shape[-1])
        x = x_resid + x_out

        ### now, apply the second SE(3)-Hyena Operator Layer
        x = x + self.hyena_layer2.forward(x)

        # Save the inputs for residual
        x_resid = x
        # x_out = self.tensor_product_2(x)
        x_out = self.mlp2(x)
        x_out = self.batch_norm2(x_out)
        x_out = self.norm_activation_2(x_out)
        x_out = x_out.reshape(batch_size, num_tokens, x_out.shape[-1])
        x = x_resid + x_out

        ### now, apply the third SE(3)-Hyena Operator Layer
        x = x + self.hyena_layer3.forward(x)

        # Save the inputs for residual
        x_resid = x
        # x_out = self.tensor_product_3(x)
        x_out = self.mlp3(x)
        x_out = self.batch_norm3(x_out)
        x_out = self.norm_activation_3(x_out)
        x_out = x_out.reshape(batch_size, num_tokens, x_out.shape[-1])
        x = x_resid + x_out

        x = x.reshape(batch_size, num_tokens, x.shape[-1])
        x = x.mean(dim=1).unsqueeze(1)

        out, _ = self.proj(x)
        return out.squeeze(1)


class GRDSE3Hyena(nn.Module):
    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension=16,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
        vector_attention_type="FFT",
    ):
        """
        followed by a mean pooling of equivariant features and one output equivariant MLP
        In SE3-hyena operator We set the hidden dimension=16 for invariant and 128 for equivariant streams
        in the gating operator, we used dimension of 8 for both invariant and equivariant features
        this ends up with ~800k trainable parameters
        """
        super(GRDSE3Hyena, self).__init__()
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

        ### se3 hyena operators
        self.hyena_layer1 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=self.input_dimension_1,
            hidden_vector_multiplicity=64,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
            vector_attention_type=vector_attention_type,
        ).to(self.device)

        self.mlp1 = LinearProjection(
            input_inv_dimension=self.positional_encoding_dimension,
            output_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_vector_multiplicity=1,
        ).to(self.device)

        self.batch_norm_1 = BatchNormLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        )

        self.activation_1 = NormActivationLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        )

        self.hyena_layer2 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=input_dimension_2,
            hidden_vector_multiplicity=64,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
            vector_attention_type=vector_attention_type,
        )

        self.mlp2 = LinearProjection(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
        )
        self.batch_norm_2 = BatchNormLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        )

        self.activation_2 = NormActivationLayer(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
        )

        self.hyena_layer3 = SE3HyenaOperator(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            hidden_inv_dimension=input_dimension_3,
            hidden_vector_multiplicity=32,
            output_inv_dimension=self.positional_encoding_dimension,
            output_vector_multiplicity=1,
            vector_attention_type=vector_attention_type,
        )

        ### final output projection
        self.proj = LinearProjection(
            input_inv_dimension=self.positional_encoding_dimension,
            input_vector_multiplicity=1,
            output_inv_dimension=0,
            output_vector_multiplicity=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ### need to make sure input is in the form
        ### (q_1,a_1),(q_2,a_2),...,(q_n,a_n), q_(n+1), where q_{n+1} is the query matrix

        ### input sequence x~(b,N,3)
        batch_size, num_tokens, _ = x.shape
        ### Expand tensor to shape (b, N, d)
        f = self.invariant_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Create token type indices: 0 for keys, 1 for values, 2 for question
        token_types = torch.zeros(
            (batch_size, num_tokens), dtype=torch.long, device=x.device
        )
        token_types[:, 0::2] = 0  # key
        token_types[:, 1::2] = 1  # value
        token_types[:, -1] = 2  # question (last token)

        # Get token-type embeddings
        token_type_embed = self.token_type_embedding(token_types)  # (B, N, D)

        # Combine with positional encodings
        f = f  # + token_type_embed.to(self.device)

        x_start, f_start = x, f
        x, f = self.hyena_layer1.forward(x, f)
        x, f = x + x_start, f + f_start

        # Save the inputs for residual
        x_resid, f_resid = x, f

        # Apply MLP
        x_out, f_out = self.mlp1(x, f)
        x_out, f_out = self.batch_norm_1(x_out, f_out)
        x_out, f_out = self.activation_1(x_out, f_out)
        x_out = x_out.reshape(batch_size, num_tokens, x.shape[-1])
        f_out = f_out.reshape(batch_size, num_tokens, f.shape[-1])

        # Add residual
        x = x_resid + x_out
        f = f_resid + f_out

        x_start, f_start = x, f
        x, f = self.hyena_layer2.forward(x, f)
        x, f = x + x_start, f + f_start

        # Save the inputs for residual
        x_resid, f_resid = x, f

        x_out, f_out = self.mlp2(x, f)
        x_out, f_out = self.batch_norm_2(x_out, f_out)
        x_out, f_out = self.activation_1(x_out, f_out)

        x_out = x_out.reshape(batch_size, num_tokens, x_out.shape[-1])
        f_out = f_out.reshape(batch_size, num_tokens, f.shape[-1])

        # Add residual
        x = x_resid + x_out
        f = f_resid + f_out

        x_start, f_start = x, f
        x, f = self.hyena_layer1.forward(x, f)
        x, f = x + x_start, f + f_start

        ### Mean pooling over token dimension
        x = x.mean(dim=1).unsqueeze(1)
        f = f.mean(dim=1).unsqueeze(1)

        ### Apply the linear layer to the last dimension (b,3+d)-->(b,3)
        out, _ = self.proj(x, f)
        out = out.squeeze(1)
        return out


class GRDStandard(nn.Module):
    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension=512,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
    ):
        super(GRDStandard, self).__init__()
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

        ### token embeddings, 3 types
        self.token_type_embedding = nn.Embedding(3, self.token_encoding_dimension).to(
            self.device
        )

        self.positions = torch.arange(self.seq_len, device=self.device)
        self.learned_pos_enc = nn.Linear(
            self.sequence_length, self.token_encoding_dimension
        ).to(self.device)

        ### se3 hyena operators
        self.hyena_layer1 = StandardAttention(
            input_dimension=self.positional_encoding_dimension,
            hidden_dimension=4 * self.positional_encoding_dimension,
            output_dimension=self.positional_encoding_dimension,
        ).to(self.device)

        # MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(
                self.positional_encoding_dimension, self.positional_encoding_dimension
            ),
            nn.BatchNorm1d(self.positional_encoding_dimension),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hyena_layer2 = StandardAttention(
            input_dimension=self.positional_encoding_dimension,
            hidden_dimension=4 * self.input_dimension_1,
            output_dimension=self.positional_encoding_dimension,
        ).to(self.device)

        # MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(
                self.positional_encoding_dimension, self.positional_encoding_dimension
            ),
            nn.BatchNorm1d(self.positional_encoding_dimension),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hyena_layer3 = StandardAttention(
            input_dimension=self.positional_encoding_dimension,
            hidden_dimension=4 * self.input_dimension_2,
            output_dimension=self.positional_encoding_dimension,
        )

        # MLP
        self.mlp3 = nn.Sequential(
            nn.Linear(
                self.positional_encoding_dimension, self.positional_encoding_dimension
            ),
            nn.BatchNorm1d(self.positional_encoding_dimension),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        ### final output projection
        self.proj = nn.Linear(self.positional_encoding_dimension, 3).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        f = self.invariant_features.unsqueeze(0).expand(batch_size, -1, -1)

        token_types = torch.zeros(
            (batch_size, num_tokens), dtype=torch.long, device=x.device
        )
        token_types[:, 0::2] = 0  # key
        token_types[:, 1::2] = 1  # value
        token_types[:, -1] = 2  # question (last token)

        # Get token-type embeddings
        token_type_embed = self.token_type_embedding(token_types)  # (B, N, D)

        f = f + token_type_embed.to(self.device)  ### fixed positional encoding
        f = f + self.learned_pos_enc(self.positions)  ### learned positional encoding

        f = f + self.hyena_layer1.forward(f)

        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp1(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        f = f + self.hyena_layer2.forward(f)

        # f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])

        f = self.mlp2(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        ### now, apply the third SE(3)-Hyena Operator Layer
        f = f + self.hyena_layer3.forward(f)

        f_resid = f
        f = f.reshape(batch_size * num_tokens, f.shape[-1])
        f = self.mlp3(f)
        f = f.reshape(batch_size, num_tokens, f.shape[-1])
        f = f_resid + f

        f = f.mean(dim=1)
        out = self.proj(f)

        assert out.shape == (batch_size, 3)
        return out


class GRDHyena(nn.Module):
    def __init__(
        self,
        sequence_length=256,
        positional_encoding_dimension=512,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
    ):
        super(GRDHyena, self).__init__()
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
        self.hyena_layer1 = Hyena_Operator(
            input_dimension=self.positional_encoding_dimension,
            output_dimension=self.input_dimension_1,
            device=self.device,
        )

        self.linear1 = nn.Linear(self.input_dimension_1, self.input_dimension_1).to(
            self.device
        )

        self.hyena_layer2 = Hyena_Operator(
            input_dimension=self.input_dimension_1,
            output_dimension=self.input_dimension_2,
            device=self.device,
        )

        self.linear2 = nn.Linear(self.input_dimension_2, self.input_dimension_2).to(
            self.device
        )

        self.hyena_layer3 = Hyena_Operator(
            input_dimension=self.input_dimension_2,
            output_dimension=self.input_dimension_3,
            device=self.device,
        )

        self.linear3 = nn.Linear(self.input_dimension_3, self.input_dimension_3).to(
            self.device
        )

        ### final output projection
        self.proj = nn.Linear(self.input_dimension_3, 3).to(self.device)

        ### non-linearity
        self.activation = nn.GELU()  # or nn.ReLU(), etc.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ### need to make sure input is in the form
        ### (q_1,a_1),(q_2,a_2),...,(q_n,a_n), q_(n+1), where q_{n+1} is the query matrix

        ### input sequence x~(b,N,3)
        batch_size, num_tokens, _ = x.shape
        ### Expand tensor to shape (b, N, d)
        f = self.invariant_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Create token type indices: 0 for keys, 1 for values, 2 for question
        token_types = torch.zeros(
            (batch_size, num_tokens), dtype=torch.long, device=x.device
        )
        token_types[:, 0::2] = 0  # key
        token_types[:, 1::2] = 1  # value
        token_types[:, -1] = 2  # question (last token)

        # Get token-type embeddings
        token_type_embed = self.token_type_embedding(token_types)  # (B, N, D)

        # Combine with positional encodings
        f = f + token_type_embed.to(self.device)
        x_emb = self.emb_position(x)
        f = f + x_emb

        f = self.hyena_layer1.forward(f)

        # Save the inputs for residual
        f_resid = f
        f_out = self.linear1(f)
        f_out = self.activation(f_out)
        f = f_resid + f_out

        f = self.hyena_layer2.forward(f)

        # Save the inputs for residual
        f_resid = f

        f_out = self.linear2(f)

        f_out = self.activation(f_out)
        f = f_resid + f_out

        ### now, apply the third SE(3)-Hyena Operator Layer
        f = self.hyena_layer3.forward(f)

        f_resid = f
        f_out = self.linear3(f)
        f_out = self.activation(f_out)
        f = f_resid + f_out
        ### Mean pooling over token dimension
        f = f.mean(dim=1)

        assert f.shape == (batch_size, self.input_dimension_3), "Mean pool wrong size"

        ### Apply the linear layer to the last dimension (b,3+d)-->(b,3)
        out = self.proj(f)

        assert out.shape == (batch_size, 3)
        return out
