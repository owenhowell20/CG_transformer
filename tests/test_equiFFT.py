import pytest
import sys
import os
import torch
import numpy as np
import escnn
from fixtures import so3_group

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.equiFFT import equiFFT, ClebschGordonProjection
from src.tensor_products import project_tensor_product
from src.sparse_tensor_products import project_tensor_product_torch_sparse


def test_CGP():
    batch_size = 4
    N = 16

    l1 = 3
    l2 = 4

    m1 = 4
    m2 = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u = torch.randn(batch_size, N, 2 * l1 + 1, m1, device=device)
    v = torch.randn(batch_size, N, 2 * l2 + 1, m2, device=device)

    y = ClebschGordonProjection(
        u, v, ell_out=3, projection_type="inter", use_sparse=False
    )
    y_sparse = ClebschGordonProjection(
        u, v, ell_out=3, projection_type="inter", use_sparse=True
    )

    assert y.shape == y_sparse.shape
    assert torch.allclose(y, y_sparse, atol=1e-3), "Sparse and Dense not agree"


def test_inter_equiFFT_equivariant(so3_group):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_vals = [1, 2, 3, 4, 5]
    J_vals = [0, 1, 2, 3, 4, 5]
    ell_out_vals = [0, 1, 2, 3]
    ell_in_vals = [0, 2, 3]

    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_out_vals:
                if np.abs(ell_in_2 - ell_in_1) <= J and J <= ell_in_1 + ell_in_2:
                    for m1 in m_vals:
                        for m2 in m_vals:
                            ell_out = J

                            batch_size = 4
                            num_tokens = 20

                            q = torch.randn(
                                batch_size,
                                num_tokens,
                                2 * ell_in_1 + 1,
                                m1,
                                device=device,
                            )
                            k = torch.randn(
                                batch_size,
                                num_tokens,
                                2 * ell_in_2 + 1,
                                m2,
                                device=device,
                            )

                            ### output tensor
                            v_o = equiFFT(q, k, ell_out, projection_type="inter")

                            rep_ell_in_1 = so3_group.irrep(ell_in_1)
                            rep_ell_in_2 = so3_group.irrep(ell_in_2)
                            rep_out = so3_group.irrep(ell_out)

                            ### NOTE: MUST PERMUTE FIRST, ( i.e. (b,N, m_l,d_l) format ) then reshape !!!!!!
                            q = q.permute(0, 1, 3, 2).reshape(
                                batch_size * num_tokens, (2 * ell_in_1 + 1) * m1
                            )
                            k = k.permute(0, 1, 3, 2).reshape(
                                batch_size * num_tokens, (2 * ell_in_2 + 1) * m2
                            )
                            v_o = v_o.permute(0, 1, 3, 2).reshape(
                                batch_size * num_tokens, (2 * ell_out + 1) * m1 * m2
                            )

                            # Field types
                            type_ell_in_1 = escnn.nn.FieldType(
                                so3_group, m1 * [rep_ell_in_1]
                            )
                            type_ell_in_2 = escnn.nn.FieldType(
                                so3_group, m2 * [rep_ell_in_2]
                            )
                            type_out = escnn.nn.FieldType(
                                so3_group, m1 * m2 * [rep_out]
                            )

                            # Wrap tensors
                            q = escnn.nn.GeometricTensor(q, type_ell_in_1)
                            k = escnn.nn.GeometricTensor(k, type_ell_in_2)
                            v_o = escnn.nn.GeometricTensor(v_o, type_out)

                            ### apply G transformation
                            g = so3_group.fibergroup.sample()

                            # Apply the transformation to the vector features (x)
                            q_g = q.transform(g).tensor
                            k_g = k.transform(g).tensor
                            v_o_g = v_o.transform(g).tensor

                            ### now, convert back
                            q_g = q_g.reshape(
                                batch_size, num_tokens, m1, 2 * ell_in_1 + 1
                            ).permute(0, 1, 3, 2)
                            k_g = k_g.reshape(
                                batch_size, num_tokens, m2, 2 * ell_in_2 + 1
                            ).permute(0, 1, 3, 2)
                            v_o_g = v_o_g.reshape(
                                batch_size, num_tokens, m1 * m2, 2 * ell_out + 1
                            ).permute(0, 1, 3, 2)

                            ### output tensor
                            v_g_o = equiFFT(q_g, k_g, ell_out, projection_type="inter")

                            assert (
                                v_o_g.shape == v_g_o.shape
                            ), "shape outputs do not match"
                            assert torch.allclose(
                                v_o_g, v_g_o, atol=1e-4
                            ), "features not transforming correctly"


def test_intra_equiFFT_equivariant(so3_group):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_vals = [1, 2, 3, 4, 5]

    J_vals = [0, 1, 2, 3, 4, 5]
    ell_out_vals = [0, 1, 2, 3]
    ell_in_vals = [0, 2, 3]

    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_out_vals:
                if np.abs(ell_in_2 - ell_in_1) <= J and J <= ell_in_1 + ell_in_2:
                    for m1 in m_vals:
                        ell_out = J

                        batch_size = 4
                        num_tokens = 20

                        q = torch.randn(
                            batch_size,
                            num_tokens,
                            2 * ell_in_1 + 1,
                            m1,
                            device=device,
                        )
                        k = torch.randn(
                            batch_size,
                            num_tokens,
                            2 * ell_in_2 + 1,
                            m1,
                            device=device,
                        )

                        ### output tensor
                        v_o = equiFFT(q, k, ell_out, projection_type="intra")

                        rep_ell_in_1 = so3_group.irrep(ell_in_1)
                        rep_ell_in_2 = so3_group.irrep(ell_in_2)
                        rep_out = so3_group.irrep(ell_out)

                        ### NOTE: MUST PERMUTE FIRST, ( i.e. (b,N, m_l,d_l) format ) then reshape !!!!!!
                        q = q.permute(0, 1, 3, 2).reshape(
                            batch_size * num_tokens, (2 * ell_in_1 + 1) * m1
                        )
                        k = k.permute(0, 1, 3, 2).reshape(
                            batch_size * num_tokens, (2 * ell_in_2 + 1) * m1
                        )
                        v_o = v_o.permute(0, 1, 3, 2).reshape(
                            batch_size * num_tokens, (2 * ell_out + 1) * m1
                        )

                        # Field types
                        type_ell_in_1 = escnn.nn.FieldType(
                            so3_group, m1 * [rep_ell_in_1]
                        )
                        type_ell_in_2 = escnn.nn.FieldType(
                            so3_group, m1 * [rep_ell_in_2]
                        )
                        type_out = escnn.nn.FieldType(so3_group, m1 * [rep_out])

                        # Wrap tensors
                        q = escnn.nn.GeometricTensor(q, type_ell_in_1)
                        k = escnn.nn.GeometricTensor(k, type_ell_in_2)
                        v_o = escnn.nn.GeometricTensor(v_o, type_out)

                        ### apply G transformation
                        g = so3_group.fibergroup.sample()

                        # Apply the transformation to the vector features (x)
                        q_g = q.transform(g).tensor
                        k_g = k.transform(g).tensor
                        v_o_g = v_o.transform(g).tensor

                        ### now, convert back
                        q_g = q_g.reshape(
                            batch_size, num_tokens, m1, 2 * ell_in_1 + 1
                        ).permute(0, 1, 3, 2)
                        k_g = k_g.reshape(
                            batch_size, num_tokens, m1, 2 * ell_in_2 + 1
                        ).permute(0, 1, 3, 2)
                        v_o_g = v_o_g.reshape(
                            batch_size, num_tokens, m1, 2 * ell_out + 1
                        ).permute(0, 1, 3, 2)

                        ### output tensor
                        v_g_o = equiFFT(q_g, k_g, ell_out, projection_type="intra")

                        assert v_o_g.shape == v_g_o.shape, "shape outputs do not match"
                        assert torch.allclose(
                            v_o_g, v_g_o, atol=1e-4
                        ), "features not transforming correctly"
