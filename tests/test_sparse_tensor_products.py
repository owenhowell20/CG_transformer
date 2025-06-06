import pytest
import sys
import os
import torch
import numpy as np

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from src.tensor_products import (
    project_tensor_product_inter_channel,
    project_tensor_product_intra_channel,
)
from src.sparse_tensor_products import (
    project_tensor_product_torch_sparse,
    project_tensor_product_torch_sparse_v2,
    # project_tensor_product_torch_sparse_v3, # Temporarily commented out v3
    project_tensor_product_torch_sparse_v4,
    project_tensor_product_torch_sparse_v1_einsum,
    project_tensor_product_torch_sparse_v4_einsum,
)

# It's good practice to import fixtures if you have them, e.g., for so3_group
# from fixtures import so3_group # Assuming you might need this later


def test_sparse_vs_dense_inter_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_sparse_vs_dense_inter_projection")
    print("============================================")

    m_val_pairs = [(1, 1), (2, 3), (4, 2)]  # (m1, m2)
    J_vals = [0, 1, 2, 3, 4, 5]  # Extended to include higher J values
    ell_in_vals_q = [0, 1, 2, 3]  # Extended to include higher ell values
    ell_in_vals_k = [0, 1, 2, 3]  # Extended to include higher ell values
    # ell_out is J

    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell1 values: {ell_in_vals_q}")
    print(f"Testing with ell2 values: {ell_in_vals_k}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    print(
                        f"Testing case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 10

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,  # Using float64 for precision
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,  # Using float64 for precision
                    )

                    # Dense computation
                    v_dense = project_tensor_product_inter_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Sparse computation
                    v_sparse = project_tensor_product_torch_sparse(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert (
                        v_dense.shape == v_sparse.shape
                    ), f"Shape mismatch for inter projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Dense: {v_dense.shape}, Sparse: {v_sparse.shape}"

                    assert torch.allclose(
                        v_dense, v_sparse, atol=1e-5
                    ), f"Value mismatch for inter projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_dense - v_sparse))}"
    print(f"Total test cases executed: {test_count}")
    print("test_sparse_vs_dense_inter_projection passed.")


def test_sparse_vs_dense_intra_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_vals = [1, 2, 3, 4]  # Extended to include m=4
    J_vals = [0, 1, 2, 3, 4, 5]  # Extended to include higher J values
    ell_in_vals_q = [0, 1, 2, 3]  # Extended to include higher ell values
    ell_in_vals_k = [0, 1, 2, 3]  # Extended to include higher ell values
    # ell_out is J

    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m in m_vals:
                    batch_size = 2
                    num_tokens = 10

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m,  # m1
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m,  # m2 (must be same as m1 for intra)
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_intra_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Sparse computation
                    v_sparse = project_tensor_product_torch_sparse(
                        q.clone(), k.clone(), ell_out, projection_type="intra"
                    )

                    assert (
                        v_dense.shape == v_sparse.shape
                    ), f"Shape mismatch for intra projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Dense: {v_dense.shape}, Sparse: {v_sparse.shape}"

                    assert torch.allclose(
                        v_dense, v_sparse, atol=1e-5
                    ), f"Value mismatch for intra projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Max diff: {torch.max(torch.abs(v_dense - v_sparse))}"
    print("test_sparse_vs_dense_intra_projection passed.")


def test_sparse_v2_vs_dense_inter_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_val_pairs = [(1, 1), (2, 3), (4, 2), (3, 5)]  # Added one more pair
    J_vals = [0, 1, 2, 3, 4, 5]  # Extended to include higher J values
    ell_in_vals_q = [0, 1, 2, 3]  # Extended to include higher ell values
    ell_in_vals_k = [0, 1, 2, 3]  # Extended to include higher ell values
    # ell_out is J

    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    batch_size = 2
                    num_tokens = 10

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,  # Using float64 for precision
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,  # Using float64 for precision
                    )

                    # Dense computation
                    v_dense = project_tensor_product_inter_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Sparse v2 computation (chunked)
                    v_sparse_v2 = project_tensor_product_torch_sparse_v2(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert (
                        v_dense.shape == v_sparse_v2.shape
                    ), f"Shape mismatch for v2 inter projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Dense: {v_dense.shape}, Sparse v2: {v_sparse_v2.shape}"

                    assert torch.allclose(
                        v_dense, v_sparse_v2, atol=1e-5
                    ), f"Value mismatch for v2 inter projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_dense - v_sparse_v2))}"
    print("test_sparse_v2_vs_dense_inter_projection passed.")


def test_sparse_v2_vs_dense_intra_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_vals = [1, 2, 3, 4]  # Extended to include m=4
    J_vals = [0, 1, 2, 3, 4, 5]  # Extended to include higher J values
    ell_in_vals_q = [0, 1, 2, 3]  # Extended to include higher ell values
    ell_in_vals_k = [0, 1, 2, 3]  # Extended to include higher ell values
    # ell_out is J

    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m in m_vals:
                    batch_size = 2
                    num_tokens = 10

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m,  # m1
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m,  # m2 (must be same as m1 for intra)
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_intra_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Sparse v2 computation (chunked)
                    v_sparse_v2 = project_tensor_product_torch_sparse_v2(
                        q.clone(), k.clone(), ell_out, projection_type="intra"
                    )

                    assert (
                        v_dense.shape == v_sparse_v2.shape
                    ), f"Shape mismatch for v2 intra projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Dense: {v_dense.shape}, Sparse v2: {v_sparse_v2.shape}"

                    assert torch.allclose(
                        v_dense, v_sparse_v2, atol=1e-5
                    ), f"Value mismatch for v2 intra projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Max diff: {torch.max(torch.abs(v_dense - v_sparse_v2))}"
    print("test_sparse_v2_vs_dense_intra_projection passed.")


def test_all_sparse_implementations_agree():
    """Test that v1 and v2 sparse implementations produce the same results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_all_sparse_implementations_agree")
    print("============================================")

    # Test with larger ell values to better test sparse implementations
    m_val_pairs = [(1, 1), (2, 3), (4, 2), (5, 3)]  # Added one more pair
    J_vals = [1, 2, 3, 4, 5]  # Extended to include higher J values
    ell_in_vals_q = [1, 2, 3, 4]  # Extended to include higher ell values
    ell_in_vals_k = [1, 2, 3, 4]  # Extended to include higher ell values

    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell1 values: {ell_in_vals_q}")
    print(f"Testing with ell2 values: {ell_in_vals_k}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    print(
                        f"Testing case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 10

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Original sparse implementation
                    v_sparse_v1 = project_tensor_product_torch_sparse(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    # Chunked sparse implementation (v2)
                    v_sparse_v2 = project_tensor_product_torch_sparse_v2(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    # Check that v1 and v2 implementations agree with each other
                    assert torch.allclose(
                        v_sparse_v1, v_sparse_v2, atol=1e-5
                    ), f"v1 and v2 disagree: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_sparse_v1 - v_sparse_v2))}"

    print(f"Total test cases executed: {test_count}")
    print("test_all_sparse_implementations_agree passed.")


def test_sparse_v4_vs_dense_inter_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_sparse_v4_vs_dense_inter_projection")
    print("============================================")

    m_val_pairs = [(1, 1), (2, 3), (4, 2), (3, 5)]  # Same pairs as v3
    J_vals = [0, 1, 2, 3, 4, 5]  # Same J values as v3
    ell_in_vals_q = [0, 1, 2, 3]  # Same ell values as v3
    ell_in_vals_k = [0, 1, 2, 3]  # Same ell values as v3
    # ell_out is J

    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell1 values: {ell_in_vals_q}")
    print(f"Testing with ell2 values: {ell_in_vals_k}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    print(
                        f"Testing case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 10

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,  # Using float64 for precision
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,  # Using float64 for precision
                    )

                    # Dense computation
                    v_dense = project_tensor_product_inter_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Sparse v4 computation (optimized for integer angular momenta)
                    v_sparse_v4 = project_tensor_product_torch_sparse_v4(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert (
                        v_dense.shape == v_sparse_v4.shape
                    ), f"Shape mismatch for v4 inter projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Dense: {v_dense.shape}, Sparse v4: {v_sparse_v4.shape}"

                    assert torch.allclose(
                        v_dense, v_sparse_v4, atol=1e-5
                    ), f"Value mismatch for v4 inter projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_dense - v_sparse_v4))}"
    print(f"Total test cases executed: {test_count}")
    print("test_sparse_v4_vs_dense_inter_projection passed.")


def test_sparse_v4_vs_dense_intra_projection():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_sparse_v4_vs_dense_intra_projection")
    print("============================================")

    m_vals = [1, 2, 3, 4]  # Same m values as v3
    J_vals = [0, 1, 2, 3, 4, 5]  # Same J values as v3
    ell_in_vals_q = [0, 1, 2, 3]  # Same ell values as v3
    ell_in_vals_k = [0, 1, 2, 3]  # Same ell values as v3
    # ell_out is J

    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell1 values: {ell_in_vals_q}")
    print(f"Testing with ell2 values: {ell_in_vals_k}")
    print(f"Testing with multiplicity values: {m_vals}")
    print("--------------------------------------------")

    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m in m_vals:
                    print(
                        f"Testing case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 10

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m,  # m1
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m,  # m2 (must be same as m1 for intra)
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_intra_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Sparse v4 computation (optimized for integer angular momenta)
                    v_sparse_v4 = project_tensor_product_torch_sparse_v4(
                        q.clone(), k.clone(), ell_out, projection_type="intra"
                    )

                    assert (
                        v_dense.shape == v_sparse_v4.shape
                    ), f"Shape mismatch for v4 intra projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Dense: {v_dense.shape}, Sparse v4: {v_sparse_v4.shape}"

                    assert torch.allclose(
                        v_dense, v_sparse_v4, atol=1e-5
                    ), f"Value mismatch for v4 intra projection: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Max diff: {torch.max(torch.abs(v_dense - v_sparse_v4))}"
    print(f"Total test cases executed: {test_count}")
    print("test_sparse_v4_vs_dense_intra_projection passed.")


def test_higher_L_v4_implementation():
    """Test v4 implementation with very high angular momenta values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_higher_L_v4_implementation")
    print("============================================")

    # Test with even larger ell values than the v3 high-L test
    m_val_pairs = [(1, 1), (2, 2)]  # Simple multiplicities for high-L tests
    J_vals = [6, 8, 10]  # Higher J values than v3 test
    ell_in_vals_q = [4, 5, 6]  # Higher ell values than v3 test
    ell_in_vals_k = [4, 5, 6]  # Higher ell values than v3 test

    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell1 values: {ell_in_vals_q}")
    print(f"Testing with ell2 values: {ell_in_vals_k}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    print(
                        f"Testing case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 8  # Smaller size for higher L tests

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_inter_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # V4 sparse implementation
                    v_sparse_v4 = project_tensor_product_torch_sparse_v4(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert (
                        v_dense.shape == v_sparse_v4.shape
                    ), f"Shape mismatch for high L v4: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"

                    assert torch.allclose(
                        v_dense, v_sparse_v4, atol=1e-5
                    ), f"Value mismatch for high L v4: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_dense - v_sparse_v4))}"

    print(f"Total test cases executed: {test_count}")
    print("test_higher_L_v4_implementation passed.")


def test_v1_einsum_vs_dense_projections():
    """Test v1_einsum implementation against dense projections for both inter and intra modes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_v1_einsum_vs_dense_projections")
    print("============================================")

    m_val_pairs = [(1, 1), (2, 3), (4, 2), (3, 5)]
    J_vals = [0, 1, 2, 3]  # Smaller range to keep test fast
    ell_in_vals = [0, 1, 2, 3]

    print("Testing inter projection:")
    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell values: {ell_in_vals}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    # Inter projection tests
    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_in_vals:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    print(
                        f"Testing inter case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 8

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_inter_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Einsum sparse computation
                    v_einsum = project_tensor_product_torch_sparse_v1_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert (
                        v_dense.shape == v_einsum.shape
                    ), f"Shape mismatch for v1_einsum inter: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"

                    assert torch.allclose(
                        v_dense, v_einsum, atol=1e-5
                    ), f"Value mismatch for v1_einsum inter: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_dense - v_einsum))}"

    print(f"Total inter test cases: {test_count}")

    # Intra projection tests
    print("\nTesting intra projection:")
    m_vals = [1, 2, 4]
    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_in_vals:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m in m_vals:
                    print(
                        f"Testing intra case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 8

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_intra_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Einsum sparse computation
                    v_einsum = project_tensor_product_torch_sparse_v1_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="intra"
                    )

                    assert (
                        v_dense.shape == v_einsum.shape
                    ), f"Shape mismatch for v1_einsum intra: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}"

                    assert torch.allclose(
                        v_dense, v_einsum, atol=1e-5
                    ), f"Value mismatch for v1_einsum intra: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Max diff: {torch.max(torch.abs(v_dense - v_einsum))}"

    print(f"Total intra test cases: {test_count}")
    print("test_v1_einsum_vs_dense_projections passed.")


def test_v4_einsum_vs_dense_projections():
    """Test v4_einsum implementation against dense projections for both inter and intra modes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_v4_einsum_vs_dense_projections")
    print("============================================")

    m_val_pairs = [(1, 1), (2, 3), (4, 2)]
    J_vals = [0, 1, 2, 3]  # Smaller range to keep test fast
    ell_in_vals = [0, 1, 2, 3]

    print("Testing inter projection:")
    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell values: {ell_in_vals}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    # Inter projection tests
    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_in_vals:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    print(
                        f"Testing inter case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 8

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_inter_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Einsum sparse computation (v4)
                    v_einsum = project_tensor_product_torch_sparse_v4_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert (
                        v_dense.shape == v_einsum.shape
                    ), f"Shape mismatch for v4_einsum inter: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"

                    assert torch.allclose(
                        v_dense, v_einsum, atol=1e-5
                    ), f"Value mismatch for v4_einsum inter: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_dense - v_einsum))}"

    print(f"Total inter test cases: {test_count}")

    # Intra projection tests
    print("\nTesting intra projection:")
    m_vals = [1, 2, 4]
    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_in_vals:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m in m_vals:
                    print(
                        f"Testing intra case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = 8

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_intra_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # Einsum sparse computation (v4)
                    v_einsum = project_tensor_product_torch_sparse_v4_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="intra"
                    )

                    assert (
                        v_dense.shape == v_einsum.shape
                    ), f"Shape mismatch for v4_einsum intra: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}"

                    assert torch.allclose(
                        v_dense, v_einsum, atol=1e-5
                    ), f"Value mismatch for v4_einsum intra: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Max diff: {torch.max(torch.abs(v_dense - v_einsum))}"

    print(f"Total intra test cases: {test_count}")
    print("test_v4_einsum_vs_dense_projections passed.")


def test_high_L_v4_einsum():
    """Test v4_einsum implementation with high angular momentum values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_high_L_v4_einsum")
    print("============================================")

    # Test with higher angular momentum values
    m_val_pairs = [(1, 1), (2, 2)]  # Simple multiplicities for high-L tests
    J_vals = [6, 8, 10]  # Higher J values
    ell_in_vals_q = [4, 5, 6]  # Higher ell values
    ell_in_vals_k = [4, 5, 6]  # Higher ell values

    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell1 values: {ell_in_vals_q}")
    print(f"Testing with ell2 values: {ell_in_vals_k}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    test_count = 0
    for J in J_vals:
        for ell_in_1 in ell_in_vals_q:
            for ell_in_2 in ell_in_vals_k:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    print(
                        f"Testing case: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"
                    )
                    test_count += 1
                    batch_size = 2
                    num_tokens = (
                        6  # Use fewer tokens for high-L tests to reduce memory usage
                    )

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Dense computation
                    v_dense = project_tensor_product_inter_channel(
                        q.clone(), k.clone(), ell_out
                    )

                    # V4 einsum implementation
                    v_einsum = project_tensor_product_torch_sparse_v4_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert (
                        v_dense.shape == v_einsum.shape
                    ), f"Shape mismatch for high L v4_einsum: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}"

                    assert torch.allclose(
                        v_dense, v_einsum, atol=1e-5
                    ), f"Value mismatch for high L v4_einsum: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v_dense - v_einsum))}"

    print(f"Total test cases executed: {test_count}")
    print("test_high_L_v4_einsum passed.")


def test_einsum_vs_standard_implementations():
    """Compare einsum versions against their standard counterparts to ensure equivalence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nRunning test_einsum_vs_standard_implementations")
    print("================================================")

    # Test parameters
    m_val_pairs = [(1, 1), (2, 3), (4, 2)]
    J_vals = [0, 1, 2]  # Smaller range to keep test fast
    ell_in_vals = [0, 1, 2]

    print("Testing implementation equivalence:")
    print(f"Testing with J values: {J_vals}")
    print(f"Testing with ell values: {ell_in_vals}")
    print(f"Testing with multiplicity pairs: {m_val_pairs}")
    print("--------------------------------------------")

    # Test v1 vs v1_einsum
    print("Comparing v1 vs v1_einsum")
    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_in_vals:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    batch_size = 2
                    num_tokens = 8

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Standard v1 implementation
                    v1_standard = project_tensor_product_torch_sparse(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    # Einsum v1 implementation
                    v1_einsum = project_tensor_product_torch_sparse_v1_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert torch.allclose(
                        v1_standard, v1_einsum, atol=1e-5
                    ), f"v1 vs v1_einsum mismatch: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v1_standard - v1_einsum))}"

    # Test v4 vs v4_einsum
    print("\nComparing v4 vs v4_einsum")
    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_in_vals:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                for m1, m2 in m_val_pairs:
                    batch_size = 2
                    num_tokens = 8

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m1,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m2,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Standard v4 implementation
                    v4_standard = project_tensor_product_torch_sparse_v4(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    # Einsum v4 implementation
                    v4_einsum = project_tensor_product_torch_sparse_v4_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="inter"
                    )

                    assert torch.allclose(
                        v4_standard, v4_einsum, atol=1e-5
                    ), f"v4 vs v4_einsum mismatch: J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m1={m1}, m2={m2}. Max diff: {torch.max(torch.abs(v4_standard - v4_einsum))}"

    # Also test intra projections for v4
    print("\nComparing v4 vs v4_einsum (intra projection)")
    for J in J_vals:
        for ell_in_1 in ell_in_vals:
            for ell_in_2 in ell_in_vals:
                if not (np.abs(ell_in_1 - ell_in_2) <= J <= ell_in_1 + ell_in_2):
                    continue

                ell_out = J
                m_vals = [1, 2, 4]
                for m in m_vals:
                    batch_size = 2
                    num_tokens = 8

                    q = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_1 + 1,
                        m,
                        device=device,
                        dtype=torch.float64,
                    )
                    k = torch.randn(
                        batch_size,
                        num_tokens,
                        2 * ell_in_2 + 1,
                        m,
                        device=device,
                        dtype=torch.float64,
                    )

                    # Standard v4 implementation
                    v4_standard = project_tensor_product_torch_sparse_v4(
                        q.clone(), k.clone(), ell_out, projection_type="intra"
                    )

                    # Einsum v4 implementation
                    v4_einsum = project_tensor_product_torch_sparse_v4_einsum(
                        q.clone(), k.clone(), ell_out, projection_type="intra"
                    )

                    assert torch.allclose(
                        v4_standard, v4_einsum, atol=1e-5
                    ), f"v4 vs v4_einsum mismatch (intra): J={J}, ell1={ell_in_1}, ell2={ell_in_2}, m={m}. Max diff: {torch.max(torch.abs(v4_standard - v4_einsum))}"

    print("test_einsum_vs_standard_implementations passed.")


# To run these tests, you would typically use pytest from your terminal:
# pytest tests/test_sparse_tensor_products.py

# You can also run individual tests with:
# pytest tests/test_sparse_tensor_products.py::test_function_name
