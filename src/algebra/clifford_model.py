import torch
import torch.nn as nn

from clifford_projection import clifford_projection
from gating import gamma_gate, sigma_gate
from residual import VectorCrossProduct, VectorDotProduct, zeta_layer
from SE3Hyena import VectorLongConv, VectorSelfAttention


class clifford_SE3Hyena(nn.Module):
    def __init__(
        self,
    ):
        self.proj = clifford_projection()

        self.agg_eqv = VectorLongConv()
        self.agg_inv = VectorSelfAttention()

        self.gamma = gamma_gate()
        self.sigma = sigma_gate()

        self.cross_product = VectorCrossProduct()
        self.dot_product = VectorDotProduct()
        self.zeta = zeta_layer()

    def forward(self, x: torch.tensor, f: torch.tensor):
        ### input projection
        q_eqv, k_eqv, v_eqv, q_inv, k_inv, v_inv = self.proj(x, f)

        ### global context aggregation
        u_eqv = self.agg_eqv(q_eqv, k_eqv)
        u_inv = self.agg_inv(q_inv, k_inv)

        ### gating
        m_eqv, m_inv = self.gamma(u_eqv, u_inv)
        u_eqv = self.sigma(m_eqv) * u_eqv
        u_inv = self.sigma(m_inv) * u_inv

        ### resdual and output projection
        x = self.zeta(self.cross_product(x, u_eqv, v_eqv))
        f = self.zeta(self.dot_product(f, u_inv, v_inv))

        assert x.shape[1] == f.shape[1], "Dimension mismatch"
        assert x.shape[2] == 3, "Spatial dimension must be three"

        return x, f
