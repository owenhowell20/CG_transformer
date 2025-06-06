import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
from e3nn.nn import NormActivation, BatchNorm
from e3nn.o3 import FullyConnectedTensorProduct


def regular_rep_irreps(l_max: int, multiplicity: int) -> Irreps:
    return Irreps(
        [(multiplicity, f"{l}{'e' if l % 2 == 0 else 'o'}") for l in range(l_max + 1)]
    )


class RegularBatchNorm(nn.Module):
    def __init__(
        self,
        input_max_harmonic=3,
        input_multiplicity: int = 512,
    ):
        super().__init__()

        self.input_max_harmonic = input_max_harmonic
        self.input_multiplicity = input_multiplicity

        # Assuming `regular_rep_irreps` is a function that defines input irreps (same as in the original code)
        self.input_irreps = regular_rep_irreps(input_max_harmonic, input_multiplicity)

        # Using e3nn BatchNorm (it expects irreps)
        self.batch_norm = BatchNorm(self.input_irreps)

    def forward(self, x):
        """
        Args:
            x: shape [B, N, input_dim]

        Returns:
            x: shape [B, N, output_dim]
        """
        B, N, _ = x.shape

        # Apply batch normalization across the batch dimension
        # First reshape to [B * N, input_dim] to apply batch normalization
        inputs = x.view(B * N, -1)

        # Apply the batch normalization using e3nn's BatchNorm
        outputs = self.batch_norm(inputs)

        # Reshape back to [B, N, output_dim]
        outputs = outputs.view(B, N, -1)

        return outputs


class RegularTensorProductLayer(nn.Module):
    def __init__(
        self,
        input_max_harmonic=3,
        input_multiplicity: int = 512,
        output_max_harmonic=3,
        output_multiplicity: int = 512,
    ):
        super().__init__()

        self.input_max_harmonic = input_max_harmonic
        self.input_multiplicity = input_multiplicity

        self.output_max_harmonic = output_max_harmonic
        self.output_multiplicity = output_multiplicity

        self.input_irreps = regular_rep_irreps(input_max_harmonic, input_multiplicity)
        self.output_irreps = regular_rep_irreps(
            output_max_harmonic, output_multiplicity
        )

        self.tp = FullyConnectedTensorProduct(
            self.input_irreps, self.input_irreps, self.output_irreps
        )

    def forward(self, x):
        """
        Args:
            x: shape [B, N, input_dim]

        Returns:
            x: [B, N, output_dim]
        """

        B, N, _ = x.shape
        inputs = x.view(B * N, -1)

        outputs = self.tp(inputs, inputs)
        outputs = outputs.view(B, N, outputs.shape[-1])

        return outputs


class RegularNormActivation(nn.Module):
    def __init__(
        self,
        input_max_harmonic=3,
        input_multiplicity: int = 512,
    ):
        super().__init__()

        self.input_max_harmonic = input_max_harmonic
        self.input_multiplicity = input_multiplicity

        self.input_irreps = regular_rep_irreps(input_max_harmonic, input_multiplicity)

        self.norm = NormActivation(
            irreps_in=self.input_irreps, scalar_nonlinearity=torch.nn.functional.relu
        )

    def forward(self, x):
        """
        Args:
            x: shape [B, N, input__dim]

        Returns:
            x: shape [B, N, output_dim]
        """
        B, N, _ = x.shape

        # Concatenate and reshape to [B * N, total_input_dim]
        inputs = x.view(B * N, -1)
        outputs = self.norm(inputs)
        outputs = outputs.view(B, N, -1)

        return outputs


class RegularLinearProjection(nn.Module):
    def __init__(
        self,
        input_max_harmonic=3,
        input_multiplicity: int = 512,
        output_max_harmonic=3,
        output_multiplicity: int = 512,
    ):
        super().__init__()

        self.input_max_harmonic = input_max_harmonic
        self.input_multiplicity = input_multiplicity

        self.output_max_harmonic = output_max_harmonic
        self.output_multiplicity = output_multiplicity

        self.input_irreps = regular_rep_irreps(input_max_harmonic, input_multiplicity)
        self.output_irreps = regular_rep_irreps(
            output_max_harmonic, output_multiplicity
        )

        self.linear = Linear(self.input_irreps, self.output_irreps)

    def forward(self, x):
        """
        Args:
            x: shape [B, N, input_dim]

        Returns:
            x: [B, N, output_dim]
        """
        B, N, _ = x.shape

        # Concatenate and reshape to [B * N, total_input_dim]
        inputs = x.view(B * N, -1)

        outputs = self.linear(inputs)
        outputs = outputs.view(B, N, -1)

        return outputs


class RegularInputConversion(nn.Module):
    def __init__(
        self,
        input_inv_dimension=3,
        input_vector_multiplicity: int = 512,
        output_max_harmonic=3,
        output_multiplicity: int = 512,
    ):
        super().__init__()

        self.input_max_harmonic = input_inv_dimension
        self.input_multiplicity = input_vector_multiplicity

        self.output_max_harmonic = output_max_harmonic
        self.output_multiplicity = output_multiplicity

        self.input_irreps = Irreps(f"{input_inv_dimension}x0e") + Irreps(
            f"{input_vector_multiplicity}x1o"
        )
        self.output_irreps = regular_rep_irreps(
            output_max_harmonic, output_multiplicity
        )

        self.linear = Linear(self.input_irreps, self.output_irreps)

    def forward(self, vectors, scalars):
        """
        Args:
            x: shape [B, N, 3*vector]
            f: shape [b,N, d]

        Returns:
            x: [B, N, output_dim]
        """
        B, N, _ = vectors.shape
        inputs = torch.cat([scalars, vectors], dim=-1)
        inputs = inputs.view(B * N, -1)
        outputs = self.linear(inputs)
        outputs = outputs.view(B, N, -1)

        return outputs


class RegularOutputConversion(nn.Module):
    def __init__(
        self,
        input_max_harmonic=3,
        input_multiplicity: int = 3,
        output_inv_dimension: int = 512,
        output_vector_multiplicity: int = 1,
    ):
        super().__init__()

        self.input_max_harmonic = input_max_harmonic
        self.input_multiplicity = input_multiplicity

        self.output_inv_dimension = output_inv_dimension

        self.input_irreps = regular_rep_irreps(input_max_harmonic, input_multiplicity)

        self.output_irreps = Irreps(f"{output_inv_dimension}x0e") + Irreps(
            f"{output_vector_multiplicity}x1o"
        )

        self.linear = Linear(self.input_irreps, self.output_irreps)

    def forward(self, x):
        """
        Args:

        Returns:
            x: [B, N, output_dim]
        """
        B, N, _ = x.shape
        inputs = x.view(B * N, -1)
        outputs = self.linear(inputs)
        outputs = outputs.view(B, N, -1)

        ### split
        f = outputs[:, :, 0 : self.output_inv_dimension]
        x = outputs[:, :, self.output_inv_dimension :]

        return x, f
