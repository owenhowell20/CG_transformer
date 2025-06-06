import pytest
import sys
import os
import torch
import numpy as np
from fixtures import mock_data
import escnn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.equiFFT import equiFFT

from src.tensor_products import (
    get_clebsch_gordon,
    get_sparse_clebsch_gordon,
)

if torch.cuda.is_available():
    import triton
    import triton.language as tl

    from src.tensor_products import wrap_triton_project_tensor_product

    TRITON_AVAILABLE = True
else:
    TRITON_AVAILABLE = False


from fixtures import so3_group

import torch
import numpy as np


def test_dense_vs_sparse_contraction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    J_vals = [0, 1, 2, 3, 4, 5]
    ell_out_vals = [0, 1, 2, 3]
    ell_in_vals = [
        0,
        2,
        3,
    ]

    for J in J_vals:
        for ell_in in ell_in_vals:
            for ell_out in ell_out_vals:
                if np.abs(ell_out - ell_in) <= J and J <= ell_out + ell_in:
                    # Assert validity of the Clebsch-Gordon condition
                    assert (
                        np.abs(ell_out - ell_in) <= J <= ell_out + ell_in
                    ), f"Invalid parameters: |ell_out - ell_in| <= J <= ell_out + ell_in must hold"

                    # Get dense Clebsch-Gordon matrix (C)
                    C = get_clebsch_gordon(
                        J=J, l_in=ell_in, l_out=ell_out, device=device
                    ).to(
                        dtype=torch.float64
                    )  # [2l_out+1, 2l_in+1, 2J+1]

                    # Get sparse Clebsch-Gordon values (list of tuples)
                    sparse_values = get_sparse_clebsch_gordon(
                        J=J, l_in=ell_in, l_out=ell_out, device=device
                    )

                    # Random tensor for testing contraction
                    tensor = torch.randn(
                        2 * ell_out + 1,
                        2 * ell_in + 1,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense contraction using einsum
                    contracted_dense = torch.einsum("mnp,mn->p", C, tensor)  # [2J+1]

                    # Sparse contraction using (value, M, m1, m2) tuples
                    contracted_sparse = torch.zeros(
                        2 * J + 1, device=device, dtype=torch.float64
                    )

                    # Iterate over sparse Clebsch-Gordon values and compute contraction
                    for value, M, m1, m2 in sparse_values:
                        # Debug: print the current sparse value and indices for clarity
                        print(f"Value: {value}, M: {M}, m1: {m1}, m2: {m2}")
                        contracted_sparse[M] += value * tensor[m2, m1]

                    # Debug: print the results of both dense and sparse contractions
                    print(f"Contracted Dense: {contracted_dense}")
                    print(f"Contracted Sparse: {contracted_sparse}")

                    # Compare the dense and sparse contractions
                    assert torch.allclose(
                        contracted_dense, contracted_sparse, atol=1e-6
                    ), f"Mismatch! Max difference: {torch.max(torch.abs(contracted_dense - contracted_sparse))}"

                    print("Test passed successfully.")


if TRITON_AVAILABLE:

    def test_triton_projection():
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

                                batch_size = 2
                                num_tokens = 20

                                q = torch.randn(
                                    batch_size, num_tokens, 2 * ell_in_1 + 1, m1
                                )
                                k = torch.randn(
                                    batch_size, num_tokens, 2 * ell_in_2 + 1, m2
                                )

                                v = wrap_triton_project_tensor_product(
                                    q, k, ell_in_1, ell_in_2, ell_out
                                )
                                v_sparse = equiFFT(q, k, ell_out)

                                assert (
                                    v.shape == v_sparse.shape
                                ), "Triton shape outputs do not match"

                                # Use stricter tolerance for comparison
                                assert torch.allclose(
                                    v, v_sparse, atol=1e-6
                                ), f"Triton Sparse projection is not correct. Max error: {torch.max(torch.abs(v - v_sparse))}"

                                print(
                                    "Triton Sparse projection test passed successfully!"
                                )
