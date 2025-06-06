import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
from e3nn.nn import NormActivation, BatchNorm


class BatchNormLayer(nn.Module):
    def __init__(
        self,
        input_inv_dimension: int = 215,
        input_vector_multiplicity: int = 5,
    ):
        super().__init__()

        self.input_inv_dimension = input_inv_dimension
        self.input_vector_multiplicity = input_vector_multiplicity

        self.input_irreps = Irreps(f"{input_inv_dimension}x0e") + Irreps(
            f"{input_vector_multiplicity}x1o"
        )
        self.batch_norm = BatchNorm(self.input_irreps)

    def forward(self, vectors, scalars):
        """
        Args:
            x: shape [B, N, input_dim]

        Returns:
            x: shape [B, N, output_dim]
        """
        B, N, _ = vectors.shape

        # Concatenate and reshape to [B * N, total_input_dim]
        inputs = torch.cat([scalars, vectors], dim=-1).view(B * N, -1)
        outputs = self.batch_norm(inputs)
        outputs = outputs.view(B, N, -1)

        scalar_out = outputs[..., 0 : self.input_inv_dimension]
        vector_out = outputs[..., self.input_inv_dimension :]

        return vector_out, scalar_out


class TensorProductLayer(nn.Module):
    """
    Equivariant layer that mixes scalar and vector inputs using a fully connected tensor product.

    Assumes input is composed of:
        - `feature_dim` scalar channels (0e)
        - `vector_multiplicity` vector channels (1o)
    """

    def __init__(
        self,
        input_inv_dimension=512,
        input_vector_multiplicity=1,
        output_inv_dimension=512,
        output_vector_multiplicity=1,
    ):
        super().__init__()

        self.input_inv_dimension = input_inv_dimension
        self.input_vector_multiplicity = input_vector_multiplicity

        self.output_inv_dimension = output_inv_dimension
        self.output_vector_multiplicity = output_vector_multiplicity

        self.input_irreps = Irreps(f"{input_inv_dimension}x0e") + Irreps(
            f"{input_vector_multiplicity}x1o"
        )
        self.output_irreps = Irreps(f"{output_inv_dimension}x0e") + Irreps(
            f"{output_vector_multiplicity}x1o"
        )

        self.tp = FullyConnectedTensorProduct(
            self.input_irreps, self.input_irreps, self.output_irreps
        )

    def forward(self, vectors, scalars):
        """
        Args:
            scalars: Tensor of shape [batch, n_tokens, feature_dim]
            vectors: Tensor of shape [batch, n_tokens, 3*vector_multiplicity]
        Returns:
            Tensor of shape [batch, n_tokens, output_irreps.dim]
        """
        B = scalars.shape[0]
        N = scalars.shape[1]

        inputs = torch.cat([scalars, vectors], dim=-1)

        inputs = inputs.view(B * N, -1)
        outputs = self.tp(inputs, inputs)

        ### Reshape back
        outputs = outputs.view(B, N, -1)

        scalar_out = outputs[..., 0 : self.output_inv_dimension]
        vector_out = outputs[..., self.output_inv_dimension :]

        return vector_out, scalar_out


class LinearProjection(nn.Module):
    def __init__(
        self,
        input_inv_dimension: int = 512,
        output_inv_dimension: int = 512,
        input_vector_multiplicity: int = 1,
        output_vector_multiplicity: int = 1,
    ):
        super().__init__()

        self.input_inv_dimension = input_inv_dimension
        self.input_vector_multiplicity = input_vector_multiplicity

        self.output_inv_dimension = output_inv_dimension
        self.output_vector_multiplicity = output_vector_multiplicity

        self.input_irreps = Irreps(
            f"{input_inv_dimension}x0e + {input_vector_multiplicity}x1o"
        )
        self.output_irreps = Irreps(
            f"{output_inv_dimension}x0e + {output_vector_multiplicity}x1o"
        )

        self.linear = Linear(self.input_irreps, self.output_irreps)

    def forward(self, vectors: torch.Tensor, scalars: torch.Tensor):
        """
        Args:
            scalars: shape [B, N, input_inv_dim]
            vectors: shape [B, N, 3*input_vector_multiplicity]

        Returns:
            vector_out: [B, N, 3*output_vector_multiplicity]
            scalar_out: [B, N, output_inv_dim]
        """
        B, N, _ = scalars.shape

        # Concatenate and reshape to [B * N, total_input_dim]
        inputs = torch.cat([scalars, vectors], dim=-1).view(B * N, -1)

        # Apply equivariant linear layer
        outputs = self.linear(inputs)

        # Reshape to [B, N, output_irreps.dim]
        outputs = outputs.view(B, N, -1)

        # Split scalar and vector outputs
        scalar_out = outputs[..., : self.output_inv_dimension]
        vector_out = outputs[..., self.output_inv_dimension :]

        return vector_out, scalar_out


class NormActivationLayer(nn.Module):
    def __init__(
        self,
        input_inv_dimension: int = 512,
        input_vector_multiplicity: int = 1,
    ):
        super().__init__()

        self.input_inv_dimension = input_inv_dimension
        self.input_vector_multiplicity = input_vector_multiplicity

        self.input_irreps = Irreps(
            f"{input_inv_dimension}x0e + {input_vector_multiplicity}x1o"
        )

        self.norm = NormActivation(
            irreps_in=self.input_irreps, scalar_nonlinearity=torch.nn.functional.relu
        )

    def forward(self, vectors: torch.Tensor, scalars: torch.Tensor):
        """
        Args:
            scalars: shape [B, N, input_inv_dim]
            vectors: shape [B, N, 3*input_vector_multiplicity]

        Returns:
            vector_out: [B, N, 3*output_vector_multiplicity]
            scalar_out: [B, N, output_inv_dim]
        """
        B, N, _ = scalars.shape

        # Concatenate and reshape to [B * N, total_input_dim]
        inputs = torch.cat([scalars, vectors], dim=-1).view(B * N, -1)

        # Apply equivariant linear layer
        outputs = self.norm(inputs)

        # Reshape to [B, N, output_irreps.dim]
        outputs = outputs.view(B, N, -1)

        # Split scalar and vector outputs
        scalar_out = outputs[..., : self.input_inv_dimension]
        vector_out = outputs[..., self.input_inv_dimension :]

        return vector_out, scalar_out


class EquivariantGating(nn.Module):
    """
    Equivariant gamma gating module.

    Takes equivariant vector features (R^3) and invariant scalar features (R^d),
    and outputs two scalar gates, using an equivariant linear map under a
    specified symmetry group

    This module is typically used to modulate or gate the influence of equivariant
    and invariant components in an SE(3)-equivariant neural network. The gating
    operation is equivariant and designed to be independent of the global position
    (i.e., translation-invariant).

    Args:
        feature_dim (int): Number of scalar (invariant) input features per point.
        vector_multiplicity (int): Number of vector (equivariant) input channels per point.
        group (escnn.gspaces.GSpace): Symmetry group defining equivariance, e.g., SO(3).

    Inputs:
        u_eqv (Tensor): Equivariant input of shape (B, N, 3 * vector_multiplicity).
        u_inv (Tensor): Invariant input of shape (B, N, feature_dim).

    Returns:
        m_eqv (Tensor): Scalar gate for equivariant features, shape (B, N, 1).
        m_inv (Tensor): Scalar gate for invariant features, shape (B, N, 1).
    """

    def __init__(
        self,
        input_inv_dimension: int = 512,
        input_vector_multiplicity=1,
        hidden_inv_dimension: int = 512,
        hidden_vector_multiplicity=5,
        hidden_type_2_multiplicity=3,
        output_inv_dimension: int = 2,
    ):
        super().__init__()
        self.input_inv_dimension = input_inv_dimension
        self.input_vector_multiplicity = input_vector_multiplicity

        self.hidden_inv_dimension = hidden_inv_dimension
        self.hidden_vector_multiplicity = hidden_vector_multiplicity

        self.input_irreps = Irreps(
            f"{input_inv_dimension}x0e + {input_vector_multiplicity}x1o"
        )
        self.hidden_irreps = Irreps(
            f"{hidden_inv_dimension}x0e + {hidden_vector_multiplicity}x1o+{hidden_type_2_multiplicity}x1o"
        )
        self.output_irreps = Irreps(f"{output_inv_dimension}x0e")

        self.tp = FullyConnectedTensorProduct(
            self.input_irreps, self.input_irreps, self.hidden_irreps
        )
        self.linear = Linear(self.hidden_irreps, self.output_irreps)

    def forward(self, vectors, scalars):
        b, N, d = scalars.shape

        inputs = torch.cat([scalars, vectors], dim=-1).view(b * N, -1)
        outputs = self.tp(inputs, inputs)
        outputs = self.linear(outputs)

        ### unstack gamma_u in same way
        m_eqv = outputs[:, :1]  # (b N, 1)
        m_inv = outputs[:, 1:]  # (b N, 1)

        ### reshape back into (b,N,1), (b,N,1)
        m_eqv = m_eqv.reshape(b, N, 1)
        m_inv = m_inv.reshape(b, N, 1)

        return m_eqv, m_inv
