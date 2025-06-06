import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from .utils import positional_encoding
from .equiFFT import equiFFT, ClebschGordonProjection


from .nonlinearities import normalize_features

from .projections import (
    LinearProjection,
    TensorProductLayer,
    NormActivationLayer,
    EquivariantGating,
    BatchNormLayer,
)
from .layers import (
    RegularLinearProjection,
    RegularTensorProductLayer,
    RegularNormActivation,
    RegularBatchNorm,
)
from .SE3Hyena import (
    VectorLongConv,
    VectorSelfAttention,
    VectorCrossProduct,
    VectorDotProduct,
)
from .attention import (
    ScalarLongConv,
    ScalarFFTConv1D,
    ScalarSelfAttention,
)

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import (
    embed_point,
    embed_scalar,
    extract_point,
    extract_scalar,
    embed_translation,
    extract_translation,
)

from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias, GConvSE3
from equivariant_attention.fibers import Fiber


class AttentionModule(nn.Module):
    """
    Generic Attention Module for equivariant transformer.

    Can be one of:
        - Standard (Vanilla transformer)
        - SE3Hyena
        - SE3HyperHyena
        - SE3Transformer
        - SE3RandomFeaturesEquivariant
        - SEGNN
        - Equiformerv2 module
        - Hyena Attention
    """

    def __init__(self, model_type: str, **kwargs):
        super().__init__()

        attention_classes = {
            "Standard": StandardAttention,
            "SE3Hyena": SE3HyenaOperator,
            "SE3HyperHyena": SE3HyperHyenaOperator,
            # "SE3Transformer": SE3Transformer,
            # "SE3RandomFeaturesEquivariant": SE3RandomFeaturesEquivariant,
            # "SEGNN": SEGNN
        }

        if model_type not in attention_classes:
            raise ValueError(
                f"Unknown model_type: {model_type}. Choose from {list(attention_classes.keys())}"
            )

        self.attention = attention_classes[model_type](**kwargs)

    def forward(self, *args, **kwargs):
        return self.attention(*args, **kwargs)


class StandardAttention(nn.Module):
    def __init__(
        self,
        input_dimension=512,
        hidden_dimension=128,
        output_dimension=32,
        num_heads=8,  ### default 8 heads
        dropout=0.1,
    ):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dimension = input_dimension
        self.hidden_dimension = input_dimension
        self.output_dimension = output_dimension

        self.num_heads = num_heads
        self.head_dim = self.input_dimension // self.num_heads
        assert (
            self.input_dimension % self.num_heads == 0
        ), "Embed dim must be divisible by num_heads"

        embed_dim = self.num_heads * self.head_dim

        self.norm = nn.LayerNorm(self.input_dimension)  # Pre-attn LayerNorm

        self.q_proj = nn.Linear(self.input_dimension, embed_dim).to(self.device)
        self.k_proj = nn.Linear(self.input_dimension, embed_dim).to(self.device)
        self.v_proj = nn.Linear(self.input_dimension, embed_dim).to(self.device)
        self.out_proj = nn.Linear(embed_dim, self.input_dimension).to(self.device)

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_dimension),
            nn.Linear(self.input_dimension, 4 * self.hidden_dimension),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(4 * self.hidden_dimension, self.input_dimension),
        )

    def forward(self, f: torch.tensor):
        ### get bach and token dimension
        b, N, d = f.shape

        assert f.shape[2] == self.input_dimension, "wrong input dimension"

        ### apply layer norm
        xf = self.norm(f)

        ### input projection
        q = self.q_proj(xf)
        k = self.k_proj(xf)
        v = self.v_proj(xf)

        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        attn = self.out_proj(attn_output)

        assert f.shape == attn.shape, "Attention output different shape"
        out = f + F.gelu(attn)
        out = out + self.mlp(out)

        out = out.view(b, N, self.output_dimension)
        return out


class HyenaAttention(nn.Module):
    def __init__(
        self,
        input_dimension: int = 512,
        hidden_dimension: int = 128,
        output_dimension: int = 32,
        num_heads: int = 8,
        order: int = 3,
        device: torch.device = "cpu",
        kernel_size: int = 7,
        stride: int = 1,
    ):
        super(HyenaAttention, self).__init__()
        self.order = order
        self.dimension = input_dimension
        self.output_dimension = output_dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.embedding_dimension = hidden_dimension
        self.device = device

        self.projection = nn.Linear(
            self.dimension, self.dimension * (self.order + 1), bias=True
        )
        self.short_conv1d = nn.Conv1d(
            in_channels=self.dimension * (self.order + 1),
            out_channels=self.dimension * (self.order + 1),
            kernel_size=3,
            padding=1,
        )
        self.output_proj = nn.Linear(self.dimension, self.output_dimension)
        self.FFN = nn.Linear(
            self.embedding_dimension, self.order * self.dimension, bias=True
        )

    def FFT_conv(self, h_filter, v):
        v = v.contiguous()
        h_filter = h_filter.contiguous()
        h_filter = h_filter.unsqueeze(0).repeat(v.shape[0], 1, 1)
        V_f = torch.fft.rfft(v, dim=1)
        H_f = torch.fft.rfft(h_filter, dim=1)
        Y_f = V_f * H_f
        y = torch.fft.irfft(Y_f, n=v.shape[1], dim=1)
        return y

    def forward(self, u):
        batch_size, seq_len, dim = u.shape

        # No caching — recompute t and window every forward pass
        t = positional_encoding(seq_len, self.embedding_dimension).to(u.device)
        window = torch.hann_window(seq_len, periodic=False, device=u.device).unsqueeze(
            1
        )

        hat_z = self.projection(u)
        z = self.short_conv1d(hat_z.permute(0, 2, 1)).permute(0, 2, 1)
        assert z.shape == hat_z.shape, "conv mismatch shapes"

        v = z[:, :, 0 : self.dimension]
        projs = z[:, :, self.dimension :]
        hat_h = self.FFN(t)

        assert projs.shape[1] == hat_h.shape[0], "dimensions"
        assert projs.shape[2] == hat_h.shape[1], "dimensions"
        h = hat_h * window.to(hat_h.device)

        for n in range(self.order):
            xn = projs[:, :, self.dimension * n : self.dimension * (n + 1)]
            hn = h[:, n * self.dimension : (n + 1) * self.dimension]
            v = xn * self.FFT_conv(hn, v)

        v = self.output_proj(v)


class SE3HyenaOperator(nn.Module):
    def __init__(
        self,
        input_inv_dimension=32,
        input_vector_multiplicity=1,
        output_inv_dimension=32,
        output_vector_multiplicity=1,
        hidden_vector_multiplicity=3,
        hidden_inv_dimension=32,
        kernel_size=7,
        vector_attention_type="FFT",
        scalar_attention_type="Standard",
        scalar_attention_kwargs=None,
        **kwargs,
    ):
        super().__init__()

        SCALAR_ATTENTION_TYPE = {
            "Long-Conv": lambda input_dim, output_dim, kernel_size=None, **kwargs: ScalarLongConv(
                input_dim,
                kernel_size or 7,  # default kernel size
                kwargs.get("stride", 1),  # default stride
            ),
            "Standard": lambda input_dim, output_dim, kernel_size=None, **kwargs: ScalarSelfAttention(),
            "FFT": lambda input_dim, output_dim, kernel_size=None, **kwargs: ScalarFFTConv1D(
                input_dim, kernel_size or 7
            ),
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_inv_dimension = input_inv_dimension
        self.input_vector_multiplicity = input_vector_multiplicity
        self.input_dimension = input_inv_dimension + 3 * input_vector_multiplicity

        self.hidden_inv_dimension = hidden_inv_dimension
        self.hidden_vector_multiplicity = hidden_vector_multiplicity

        self.output_inv_dimension = output_inv_dimension
        self.output_vector_multiplicity = output_vector_multiplicity
        self.output_dimension = (
            output_inv_dimension + 3 * self.output_vector_multiplicity
        )

        ### attention types
        self.scalar_attention_type = scalar_attention_type
        self.vector_attention_type = vector_attention_type

        self.batch_norm = BatchNormLayer(
            input_inv_dimension=self.input_inv_dimension,
            input_vector_multiplicity=self.input_vector_multiplicity,
        ).to(self.device)

        self.q_proj = LinearProjection(
            input_inv_dimension=self.input_inv_dimension,
            input_vector_multiplicity=self.input_vector_multiplicity,
            output_inv_dimension=self.hidden_inv_dimension,
            output_vector_multiplicity=self.hidden_vector_multiplicity,
        ).to(self.device)

        self.k_proj = LinearProjection(
            input_inv_dimension=self.input_inv_dimension,
            input_vector_multiplicity=self.input_vector_multiplicity,
            output_inv_dimension=self.hidden_inv_dimension,
            output_vector_multiplicity=self.hidden_vector_multiplicity,
        ).to(self.device)

        self.v_proj = LinearProjection(
            input_inv_dimension=self.input_inv_dimension,
            input_vector_multiplicity=self.input_vector_multiplicity,
            output_inv_dimension=self.hidden_inv_dimension,
            output_vector_multiplicity=self.hidden_vector_multiplicity,
        ).to(self.device)

        ### scalar attention
        scalar_attention_kwargs = scalar_attention_kwargs or {}
        if scalar_attention_type not in SCALAR_ATTENTION_TYPE:
            raise ValueError(f"Unknown attention type: {scalar_attention_type}")

        self.scalar_long_conv = SCALAR_ATTENTION_TYPE[scalar_attention_type](
            self.input_dimension,
            self.output_dimension,
            kernel_size,
            **scalar_attention_kwargs,
        )

        ### vector context aggregation
        self.vector_long_conv = VectorLongConv().to(self.device)
        self.vector_self_attention = VectorSelfAttention().to(self.device)

        if self.vector_attention_type == "FFT":
            self.gamma = EquivariantGating(
                input_inv_dimension=self.hidden_inv_dimension,
                input_vector_multiplicity=self.hidden_vector_multiplicity,
                hidden_inv_dimension=64,
                hidden_vector_multiplicity=8,
            ).to(self.device)

        self.cross_product = VectorCrossProduct().to(self.device)
        self.dot_product = VectorDotProduct().to(self.device)

        self.projection_1 = LinearProjection(
            input_inv_dimension=self.hidden_inv_dimension,
            input_vector_multiplicity=self.hidden_vector_multiplicity,
            output_inv_dimension=self.input_inv_dimension,
            output_vector_multiplicity=self.input_vector_multiplicity,
        ).to(self.device)

        self.batch_norm_1 = BatchNormLayer(
            input_inv_dimension=self.input_inv_dimension,
            input_vector_multiplicity=self.input_vector_multiplicity,
        ).to(self.device)

        self.nonlinear_1 = NormActivationLayer(
            input_inv_dimension=self.input_inv_dimension,
            input_vector_multiplicity=self.input_vector_multiplicity,
        ).to(self.device)

        self.final_proj = LinearProjection(
            input_inv_dimension=self.input_inv_dimension,
            input_vector_multiplicity=self.input_vector_multiplicity,
            output_inv_dimension=self.output_inv_dimension,
            output_vector_multiplicity=self.output_vector_multiplicity,
        ).to(self.device)
        self.batch_norm_2 = BatchNormLayer(
            input_inv_dimension=self.output_inv_dimension,
            input_vector_multiplicity=self.output_vector_multiplicity,
        ).to(self.device)
        self.nonlinear_2 = NormActivationLayer(
            input_inv_dimension=self.output_inv_dimension,
            input_vector_multiplicity=self.output_vector_multiplicity,
        ).to(self.device)

    def forward(self, x: torch.tensor, f: torch.tensor):
        ### get batch and token dimension
        b = x.shape[0]
        N = x.shape[1]

        x_res, f_res = x, f

        x, f = self.batch_norm(x, f)
        x = x.view(b, N, 3 * self.input_vector_multiplicity)
        f = f.view(b, N, f.shape[-1])

        ### input projection
        q_eqv, q_inv = self.q_proj(x, f)
        k_eqv, k_inv = self.k_proj(x, f)
        v_eqv, v_inv = self.v_proj(x, f)

        ### reshape into (b,N,3*v1_hidden)
        q_eqv = q_eqv.view(b, N, 3 * self.hidden_vector_multiplicity)
        k_eqv = k_eqv.view(b, N, 3 * self.hidden_vector_multiplicity)
        v_eqv = v_eqv.view(b, N, 3 * self.hidden_vector_multiplicity)

        ### reshape into (b,N, hidden_dim)
        q_inv = q_inv.view(b, N, self.hidden_inv_dimension)
        k_inv = k_inv.view(b, N, self.hidden_inv_dimension)
        v_inv = v_inv.view(b, N, self.hidden_inv_dimension)

        if self.scalar_attention_type == "Standard":
            u_inv = self.scalar_long_conv(q_inv, k_inv, v_inv)
        else:
            u_inv = self.scalar_long_conv(q_inv, k_inv)

        if self.vector_attention_type == "Standard":
            outputs = []
            for m1 in range(self.hidden_vector_multiplicity):
                q_slice = q_eqv[:, :, 3 * m1 : 3 * m1 + 3]
                k_slice = k_eqv[:, :, 3 * m1 : 3 * m1 + 3]
                v_slice = v_eqv[:, :, 3 * m1 : 3 * m1 + 3]
                u_eqv = self.vector_self_attention(q_slice, k_slice, v_slice)
                outputs.append(u_eqv)

            ### concat vector long convolutions
            u_eqv = torch.concat(outputs, dim=-1)

        elif self.vector_attention_type == "FFT_inter":
            outputs = []
            for m1 in range(self.hidden_vector_multiplicity):
                q_slice = q_eqv[:, :, 3 * m1 : 3 * m1 + 3]
                for m2 in range(self.hidden_vector_multiplicity):
                    k_slice = k_eqv[:, :, 3 * m2 : 3 * m2 + 3]

                    u_eqv = self.vector_long_conv(q_slice, k_slice)
                    outputs.append(u_eqv)
            u_eqv = torch.concat(outputs, dim=-1)

        elif self.vector_attention_type == "FFT":  ### FFT intra attention
            outputs = []
            for m1 in range(self.hidden_vector_multiplicity):
                q_slice = q_eqv[:, :, 3 * m1 : 3 * m1 + 3]
                k_slice = k_eqv[:, :, 3 * m1 : 3 * m1 + 3]

                u_eqv = self.vector_long_conv(q_slice, k_slice)
                outputs.append(u_eqv)

            u_eqv = torch.concat(outputs, dim=-1)
        else:
            raise ValueError

        ### apply gamma gating
        m_eqv, m_inv = self.gamma(u_eqv, u_inv)

        if self.scalar_attention_type == "Standard":
            u_inv = u_inv
        else:
            u_inv = F.softmax(m_inv, dim=-1) * u_inv
            u_inv = self.dot_product(f, u_inv, v_inv)

        if self.vector_attention_type == "Standard":
            u_inv = u_inv
        else:
            outputs = []
            for m1 in range(self.hidden_vector_multiplicity):
                q_slice = u_eqv[:, :, 3 * m1 : 3 * m1 + 3]
                v_slice = v_eqv[:, :, 3 * m1 : 3 * m1 + 3]

                u_eqv_temp = self.cross_product(q_slice, v_slice)
                outputs.append(u_eqv_temp)
            u_eqv = F.softmax(m_eqv, dim=-1) * torch.concat(outputs, dim=-1)

        u_eqv, u_inv = self.projection_1(u_eqv, u_inv)

        u_eqv = u_eqv.view(b, N, 3 * self.input_vector_multiplicity)
        u_inv = u_inv.view(b, N, self.input_inv_dimension)

        x = x_res + u_eqv
        f = f_res + u_inv

        x = x.view(b * N, 3 * self.input_vector_multiplicity)
        f = f.view(b * N, self.input_inv_dimension)

        x = x.view(b, N, 3 * self.input_vector_multiplicity)
        f = f.view(b, N, self.input_inv_dimension)
        x, f = self.batch_norm_1(x, f)
        x, f = self.nonlinear_1(x, f)  ### non-linearity

        x, f = self.final_proj(x, f)
        x, f = self.batch_norm_2(x, f)
        x, f = self.nonlinear_2(x, f)
        x = x.view(b, N, 3 * self.output_vector_multiplicity)
        f = f.view(b, N, self.output_inv_dimension)

        return x, f


### SWE task: intra or inter attention type should be passed as arg
### other key args to be passed: CG type computation, intermediate weightings, dropout ect
class SE3HyperHyenaOperator(nn.Module):
    def __init__(
        self,
        input_multiplicity: int = 3,
        input_max_harmonic: int = 3,
        hidden_multiplicity: int = 2,
        hidden_max_harmonic: int = 3,
        output_multiplicity: int = 3,
        output_max_harmonic: int = 3,
        num_head: int = 3,
        Clebsch_Gordon_Sparse: bool = True,
        projection_type="Intra",  ### intra or inter
        **kwargs,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_multiplicity = input_multiplicity
        self.input_max_harmonic = input_max_harmonic

        self.hidden_multiplicity = hidden_multiplicity
        self.hidden_max_harmonic = hidden_max_harmonic

        self.total_dimension = (
            self.hidden_multiplicity * (1 + self.hidden_max_harmonic) ** 2
        )

        self.projection_type = projection_type
        self.Clebsch_Gordon_Sparse = Clebsch_Gordon_Sparse

        self.input_batch_norm = RegularBatchNorm(
            input_multiplicity=input_multiplicity, input_max_harmonic=input_max_harmonic
        )

        self.q_proj = RegularLinearProjection(
            input_multiplicity=input_multiplicity,
            input_max_harmonic=input_max_harmonic,
            output_multiplicity=hidden_multiplicity,
            output_max_harmonic=hidden_max_harmonic,
        ).to(self.device)

        self.k_proj = RegularLinearProjection(
            input_multiplicity=input_multiplicity,
            input_max_harmonic=input_max_harmonic,
            output_multiplicity=hidden_multiplicity,
            output_max_harmonic=hidden_max_harmonic,
        ).to(self.device)

        self.v_proj = RegularLinearProjection(
            input_multiplicity=input_multiplicity,
            input_max_harmonic=input_max_harmonic,
            output_multiplicity=hidden_multiplicity,
            output_max_harmonic=hidden_max_harmonic,
        ).to(self.device)

        self.output_proj = RegularLinearProjection(
            input_multiplicity=hidden_multiplicity,
            input_max_harmonic=hidden_max_harmonic,
            output_multiplicity=output_multiplicity,
            output_max_harmonic=output_max_harmonic,
        ).to(self.device)

    def forward(self, x):
        b, N, _ = x.shape

        x = self.input_batch_norm(x)
        x = x.view(b, N, x.shape[-1])

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        dim_cntr = 0
        output = []
        for ell in range(self.hidden_max_harmonic + 1):
            new_dim = (2 * ell + 1) * self.hidden_multiplicity

            q_l = q[:, :, dim_cntr : dim_cntr + new_dim]
            k_l = k[:, :, dim_cntr : dim_cntr + new_dim]
            v_l = v[:, :, dim_cntr : dim_cntr + new_dim]

            q_l = q_l.reshape(b, N, self.hidden_multiplicity, 2 * ell + 1).permute(
                0, 1, 3, 2
            )
            k_l = k_l.reshape(b, N, self.hidden_multiplicity, 2 * ell + 1).permute(
                0, 1, 3, 2
            )
            v_l = v_l.reshape(b, N, self.hidden_multiplicity, 2 * ell + 1).permute(
                0, 1, 3, 2
            )

            q_l = normalize_features(q_l)
            k_l = normalize_features(k_l)
            v_l = normalize_features(v_l)

            dim_cntr += new_dim

            attn = equiFFT(
                q_l,
                k_l,
                ell_out=ell,
                projection_type=self.projection_type,
                use_sparse=self.Clebsch_Gordon_Sparse,
            )

            attn_l = ClebschGordonProjection(
                attn,
                v_l,
                ell_out=ell,
                projection_type=self.projection_type,
                use_sparse=self.Clebsch_Gordon_Sparse,
            )
            attn_l = attn_l.permute(0, 1, 3, 2).reshape(
                b, N, (2 * ell + 1) * self.hidden_multiplicity
            )

            output.append(attn_l)

        #########################################################################################################################
        x = torch.cat(output, dim=2).squeeze(-1)
        assert x.shape[2] == self.total_dimension

        x = self.output_proj(x)
        x = x.reshape(b, N, x.shape[-1])
        return x


### inputs: (input_invariant_dim*scalar, input_vector_multiplicty*vector) -->  (output_invariant_dim*scalar, output_multiplicty*vector)
class GATr_model(nn.Module):
    def __init__(
        self,
        input_inv_dimension: int = 32,
        input_vector_multiplicity: int = 1,
        hidden_inv_dimension: int = 32,
        hidden_vector_multiplicity: int = 1,
        output_inv_dimension: int = 32,
        output_vector_multiplicity: int = 1,
        blocks=3,
        attention_kwargs=None,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_inv_dimension = input_inv_dimension
        self.input_vector_multiplicity = input_vector_multiplicity
        self.in_c = max(input_inv_dimension, input_vector_multiplicity)
        self.output_inv_dimension = output_inv_dimension
        self.output_vector_multiplicity = output_vector_multiplicity
        self.out_c = max(output_inv_dimension, output_vector_multiplicity)

        self.blocks = blocks

        # self.in_c = 1

        hidden_mv_channels = 5
        hidden_s_channels = 5

        self.gatr = GATr(
            in_mv_channels=self.in_c,
            out_mv_channels=self.out_c,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        ).to(self.device)

    def forward(self, x):
        ### x~ [b, N, f_in]
        ### get batch and token dimension
        b = x.shape[0]
        N = x.shape[1]

        # Build one multivector holding masses, points, and velocities for each object
        ### invariant features
        x_invariant = x[:, :, : self.input_inv_dimension, None].to(
            self.device
        )  # (batchsize, objects, channel, 1)

        ### vector features
        x_vector = x[:, :, self.input_inv_dimension :].to(
            self.device
        )  # (batchsize, objects, channel, 3)
        x_vector = rearrange(x_vector, "b n (c d) -> b n c d", d=3)

        inv = embed_scalar(x_invariant)  # (batchsize, objects, c_inv, 16)
        vec = embed_translation(x_vector)  # (batchsize, objects, c_vec, 16)
        multivector = torch.zeros(b, N, self.in_c, 16).to(
            self.device
        )  # (batchsize, objects, channel, 16)
        multivector[..., : self.input_inv_dimension, :] = inv
        multivector[..., : self.input_vector_multiplicity, :] = vec

        # Pass data through GATr
        embedded_outputs, _ = self.gatr(multivector)  # (..., num_points, 1, 16)

        # Get output scalars of shape [B, N, output_inv_dimension, 1]
        s_out = extract_scalar(embedded_outputs)[..., : self.output_inv_dimension, :]
        s_out = s_out.reshape(b, N, -1)

        # Get output vectors of shape [B, N, output_vector_multiplicity, 3]
        v_out = extract_translation(embedded_outputs)[
            ..., : self.output_vector_multiplicity, :
        ]
        v_out = v_out.reshape(b, N, -1)

        x = torch.concat([s_out, v_out], dim=-1)

        return x


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(
        self,
        input_invariant_multplicity: int,
        input_vector_multiplicity: int,
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

        ### input
        self.input_invariant_multplicity = input_invariant_multplicity
        self.input_vector_multiplicity = input_vector_multiplicity

        self.fibers = {
            "in": Fiber(
                dictionary={
                    0: self.input_invariant_multplicity,
                    1: self.input_invariant_multplicity,
                }
            ),
            "mid": Fiber(self.num_degrees, self.num_channels),
            "out": Fiber(dictionary={1: 2}),
        }

        self.Gblock = self._build_gcn(self.fibers)

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


class TFN(nn.Module):
    """Tensorfield Network"""

    def __init__(
        self, num_layers: int, num_channels: int, num_degrees: int = 4, **kwargs
    ):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = 1

        self.fibers = {
            "in": Fiber(dictionary={0: 1, 1: 1}),
            "mid": Fiber(self.num_degrees, self.num_channels),
            "out": Fiber(dictionary={1: 2}),
        }

        blocks = self._build_gcn(self.fibers)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)
        # purely for counting paramters in utils_logging.py
        self.enc, self.dec = self.Gblock, self.FCblock

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers["in"]

        for i in range(self.num_layers - 1):
            Gblock.append(
                GConvSE3(
                    fin,
                    fibers["mid"],
                    self_interaction=True,
                    flavor="TFN",
                    edge_dim=self.edge_dim,
                )
            )
            Gblock.append(GNormBias(fibers["mid"]))
            fin = fibers["mid"]
        Gblock.append(
            GConvSE3(
                fibers["mid"],
                fibers["out"],
                self_interaction=True,
                flavor="TFN",
                edge_dim=self.edge_dim,
            )
        )

        return nn.ModuleList(Gblock), nn.ModuleList([])

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h_enc = {"0": G.ndata["c"], "1": G.ndata["v"]}
        for layer in self.Gblock:
            h_enc = layer(h_enc, G=G, r=r, basis=basis)

        return h_enc["1"]
