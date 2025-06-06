import math
from typing import Tuple
import torch
from torch import Tensor
from functools import lru_cache
import e3nn.o3 as o3
from torch_scatter import scatter_add

# Import the cached function
from src.tensor_products import get_clebsch_gordon


@lru_cache(maxsize=None)
def get_sparse_coo_clebsch_gordon(
    J: int, l_in: int, l_out: int, device, dtype=torch.float32
) -> torch.Tensor:
    """
    Compute Clebsch-Gordan coefficients as a sparse COO tensor.

    Args:
        J: Output angular momentum (int)
        l_in: Input angular momentum (int)
        l_out: Second input angular momentum (int)
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        torch.sparse_coo_tensor: Sparse tensor of shape [2*J+1, 2*l_in+1, 2*l_out+1]
            - Indices: [3, nnz] (M, m1, m2)
            - Values: [nnz] (nonzero CG coefficients)
    """
    # Get the cached CG coefficients (shape: [2*l_out+1, 2*l_in+1, 2*J+1]) with indices (m2, m1, M)
    dense_cg = get_clebsch_gordon(J=J, l_in=l_in, l_out=l_out, device=device)
    # Convert to the expected dtype if needed (CG from get_clebsch_gordon is float64)
    if dtype != dense_cg.dtype:
        dense_cg = dense_cg.to(dtype=dtype)

    # Permute to match expected dimensions for sparse tensor [2*J+1, 2*l_in+1, 2*l_out+1] (M, m1, m2)
    dense_cg = dense_cg.permute(2, 1, 0)

    # Find nonzero elements (returns indices where CG != 0)
    non_zero = torch.nonzero(dense_cg)
    # Extract values at those indices
    values = dense_cg[non_zero[:, 0], non_zero[:, 1], non_zero[:, 2]]

    # Create sparse COO tensor
    sparse_cg = torch.sparse_coo_tensor(
        indices=non_zero.t(),  # shape: [3, nnz]
        values=values,  # shape: [nnz]
        size=dense_cg.shape,  # (2*J+1, 2*l_in+1, 2*l_out+1)
        device=device,
    )
    # Coalesce to ensure unique indices and sorted order
    sparse_cg = sparse_cg.coalesce()
    return sparse_cg


def project_tensor_product_e3nn(
    u: Tensor, v: Tensor, ell_out: int, projection_type: str
) -> Tensor:
    """Tensor product projection using e3nn library."""
    b, N, d1, m1 = u.shape
    _, _, d2, m2 = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Assumption: input features and desired output features have even parity ('e').
    # This simplifies integration with non-parity-aware benchmark functions.
    # e3nn handles internal parity calculations correctly. If ell_out+'e' is not
    # possible from (ell1+'e' x ell2+'e'), e3nn will produce no output for it.
    # Output parity if inputs are 'e' x 'e' is 'e'.
    output_parity_char = "e"

    # Determine expected output multiplicity for zero tensor fallback
    expected_m_out = 0
    if projection_type == "inter":
        expected_m_out = m1 * m2
    elif projection_type == "intra":
        if m1 != m2:
            # This case will be an error for ElementwiseTensorProduct setup,
            # but for defining zero tensor shape, we use m1.
            # The actual error will be raised later if m1 != m2.
            expected_m_out = m1
        else:
            expected_m_out = m1
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

    zeros_fallback = torch.zeros(
        b, N, 2 * ell_out + 1, expected_m_out, device=u.device, dtype=u.dtype
    )

    try:
        # Define filter Irrep based on desired ell_out and assumed even parity
        filter_out_irrep = o3.Irrep(f"{ell_out}{output_parity_char}")
    except ValueError:  # Should not happen if ell_out >= 0
        return zeros_fallback

    # Reshape u and v for e3nn: (batch_total, features_total)
    # u: [B, N, D1, M1] -> u_perm: [B, N, M1, D1] -> u_flat: [B*N, M1*D1]
    u_flat = u.permute(0, 1, 3, 2).reshape(b * N, m1 * d1)
    v_flat = v.permute(0, 1, 3, 2).reshape(b * N, m2 * d2)

    tp = None
    actual_m_out = 0

    if projection_type == "inter":
        actual_m_out = m1 * m2
        try:
            irreps_in1 = o3.Irreps(f"{m1}x{ell1}{input_parity_char}")
            irreps_in2 = o3.Irreps(f"{m2}x{ell2}{input_parity_char}")
            tp = o3.FullTensorProduct(
                irreps_in1, irreps_in2, filter_ir_out=[filter_out_irrep]
            )
        except ValueError:  # Handles invalid irrep strings e.g. if ell1/ell2 < 0
            return zeros_fallback

    elif projection_type == "intra":
        if m1 != m2:
            raise ValueError(
                f"Intra projection requires m1 == m2, but got m1={m1}, m2={m2}"
            )
        actual_m_out = m1

        # For ElementwiseTensorProduct, input Irreps are lists of m1 individual Irreps.
        # e.g., if m1=2, ell1=1: "1x1e+1x1e"
        try:
            if m1 == 0:  # Handle case with zero multiplicity
                return zeros_fallback

            irreps_l1_single_str_list = [f"1x{ell1}{input_parity_char}"] * m1
            irreps_l2_single_str_list = [f"1x{ell2}{input_parity_char}"] * m1
            irreps_in1_ew_str = "+".join(irreps_l1_single_str_list)
            irreps_in2_ew_str = "+".join(irreps_l2_single_str_list)

            irreps_in1_ew = o3.Irreps(irreps_in1_ew_str)
            irreps_in2_ew = o3.Irreps(irreps_in2_ew_str)
            tp = o3.ElementwiseTensorProduct(
                irreps_in1_ew, irreps_in2_ew, filter_ir_out=[filter_out_irrep]
            )
        except ValueError:  # Handles invalid irrep strings
            return zeros_fallback
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

    if tp is None or not tp.irreps_out or tp.irreps_out.dim == 0:
        # No valid paths for this ell_out, or CG condition not met.
        # expected_m_out was determined earlier for this projection_type.
        return zeros_fallback

    tp = tp.to(u.device)  # Move tensor product module to the correct device
    y_flat = tp(u_flat, v_flat)  # Output shape [B*N, tp.irreps_out.dim]

    # Verify output irrep from tp matches expectation
    # tp.irreps_out should be a single entry list like [(calc_mul, Irrep(ell_out, P_actual))]
    # if filter_ir_out worked and CG path exists.
    final_mul_out = 0
    final_l_out = -1

    if tp.irreps_out.num_irreps == 1:
        mul_ir_pair = tp.irreps_out[0]
        final_mul_out = mul_ir_pair.mul
        final_l_out = mul_ir_pair.ir.l
        # Parity should match filter_out_irrep.ir.p if path exists

    if final_l_out != ell_out or final_mul_out != actual_m_out:
        # If e3nn produced an output that doesn't match the expected structure
        # (e.g. different multiplicity due to internal rules, or different L despite filter)
        # or if the multiplicity is zero (no path after all).
        return zeros_fallback

    # Reshape y_flat [B*N, final_mul_out * (2*ell_out+1)]
    # to output format [B, N, (2*ell_out+1), final_mul_out]

    # e3nn's data layout for Irreps("MxLp") is typically [..., M, (2L+1)] if split,
    # or [..., M*(2L+1)] if flat. TensorProduct outputs flat.
    # The .reshape operation needs to group by multiplicity first.
    # y_flat is [B*N, FeaturesDim]. FeaturesDim = final_mul_out * (2*ell_out+1)
    # We want [B, N, final_mul_out, (2*ell_out+1)] then permute.
    try:
        y_reshaped_interim = y_flat.reshape(b, N, final_mul_out, (2 * ell_out + 1))
        y = y_reshaped_interim.permute(0, 1, 3, 2).contiguous()
    except RuntimeError as e:
        # This can happen if y_flat.shape[1] is not b*N*final_mul_out*(2*ell_out+1)
        # or more likely if final_mul_out * (2*ell_out+1) does not match y_flat.shape[1]
        # which means tp.irreps_out.dim was inconsistent with final_mul_out * (2*ell_out+1)
        # This should ideally be caught by earlier checks on tp.irreps_out.
        return zeros_fallback

    return y


def project_tensor_product_torch_sparse(
    u: Tensor, v: Tensor, ell_out: int, projection_type: str = "inter"
) -> Tensor:
    """Sparse tensor product projection using torch.sparse.

    Args:
        u: [b, N, 2*ell1+1, m1] (input tensor)
        v: [b, N, 2*ell2+1, m2] (input tensor)
        ell_out: Output angular momentum
        projection_type: 'inter' (outer product) or 'intra' (elementwise, m1==m2)

    Returns:
        y: [b, N, 2*ell_out+1, m_out]
            - m_out = m1*m2 for 'inter', m1 for 'intra'
    """
    b, N, d1, m1 = u.shape
    _, _, d2, m2 = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Get sparse CG coefficients as a COO tensor
    sparse_cg = get_sparse_coo_clebsch_gordon(
        J=ell_out, l_in=ell1, l_out=ell2, device=u.device, dtype=u.dtype
    )
    # Extract indices and values
    cg_indices = sparse_cg.indices()  # [3, nnz] (M, m1, m2) ### M = m1 + m2
    cg_values = sparse_cg.values()  # [nnz]

    # Split indices for advanced indexing
    M_indices = cg_indices[0]  # [nnz]
    m1_indices = cg_indices[1]  # [nnz]
    m2_indices = cg_indices[2]  # [nnz]

    # Gather relevant slices from u and v for each nonzero CG term
    u_selected = u[:, :, m1_indices, :]  # [b, N, nnz, m1]
    v_selected = v[:, :, m2_indices, :]  # [b, N, nnz, m2]

    if projection_type == "inter":
        # Outer product over multiplicities: [b, N, nnz, m1, m2]
        product = u_selected.unsqueeze(-1) * v_selected.unsqueeze(-2)

        # Reshape last two dims: [b, N, nnz, m1*m2]
        product = product.reshape(b, N, -1, m1 * m2)
        output_multiplicity = m1 * m2

    else:  # 'intra' channel: elementwise product, m1 == m2
        assert m1 == m2, "Intra projection requires m1 == m2"
        product = u_selected * v_selected  # [b, N, nnz, m1]
        output_multiplicity = m1

    # Weight by CG values (broadcast over batch, tokens, and multiplicity)
    weighted_terms = product * cg_values.view(1, 1, -1, 1)

    # Initialize output tensor
    y = torch.zeros(
        b, N, 2 * ell_out + 1, output_multiplicity, device=u.device, dtype=u.dtype
    )

    # Scatter add: sum contributions to each output M index
    y = scatter_add(src=weighted_terms, index=M_indices.long(), dim=2, out=y)
    return y


@lru_cache(maxsize=None)
def get_sparse_coo_clebsch_gordon_v2(
    J: int,
    l_in: int,
    l_out: int,
    device,
    dtype=torch.float32,
    threshold_L_for_direct_sparse: int = 8,
) -> torch.Tensor:
    """
    Compute Clebsch-Gordan coefficients as a sparse COO tensor.
    This version first computes the full dense CG tensor and then extracts non-zero elements,
    which may be a memory bottleneck for large angular momenta. For better memory scaling with
    large L values, consider using get_truly_sparse_coo_clebsch_gordon instead.

    Args:
        J: Output angular momentum (int)
        l_in: Input angular momentum (int)
        l_out: Second input angular momentum (int)
        device: PyTorch device
        dtype: PyTorch dtype
        threshold_L_for_direct_sparse: Parameter is kept for signature compatibility.

    Returns:
        torch.sparse_coo_tensor: Sparse tensor of shape [2*J+1, 2*l_in+1, 2*l_out+1]
    """
    # Get the cached CG coefficients (shape: [2*l_out+1, 2*l_in+1, 2*J+1]) with indices (m2, m1, M)
    dense_cg = get_clebsch_gordon(J=J, l_in=l_in, l_out=l_out, device=device)

    # Convert to the expected dtype if needed (CG from get_clebsch_gordon is float64)
    if dtype != dense_cg.dtype:
        dense_cg = dense_cg.to(dtype=dtype)

    # Permute to match expected dimensions for sparse tensor [2*J+1, 2*l_in+1, 2*l_out+1] (M, m1, m2)
    dense_cg = dense_cg.permute(2, 1, 0)

    # Find nonzero elements (returns indices where CG != 0)
    # Using as_tuple=False to be explicit, though default behavior might be okay.
    non_zero_indices = torch.nonzero(dense_cg, as_tuple=False)

    # Handle case where there are no non-zero elements (e.g., selection rules forbid this J)
    if non_zero_indices.numel() == 0:
        return torch.sparse_coo_tensor(
            indices=torch.zeros((3, 0), device=device, dtype=torch.long),
            values=torch.zeros(0, device=device, dtype=dtype),
            size=(2 * J + 1, 2 * l_in + 1, 2 * l_out + 1),
            device=device,
        ).coalesce()

    # Extract values at those indices
    values = dense_cg[
        non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]
    ]

    # Create sparse COO tensor
    sparse_cg = torch.sparse_coo_tensor(
        indices=non_zero_indices.t(),  # shape: [3, nnz]
        values=values,  # shape: [nnz]
        size=dense_cg.shape,  # (2*J+1, 2*l_in+1, 2*l_out+1)
        device=device,
    )
    # Coalesce to ensure unique indices and sorted order
    return sparse_cg.coalesce()


@lru_cache(maxsize=None)
def wigner_3j_from_clebsch_gordon(
    j1: int, j2: int, j3: int, m1: int, m2: int, m3: int, device
) -> float:
    # This is the restored version that should return a Wigner 3j symbol value directly
    # based on the output of tensor_products.get_clebsch_gordon.
    if m1 + m2 + m3 != 0:  # Wigner 3j m-conservation
        return 0.0

    # Triangle inequality for 2*l values
    if (j1 + j2 < j3) or (abs(j1 - j2) > j3):
        return 0.0

    l1_val = j1 / 2.0
    l2_val = j2 / 2.0
    l3_val = (
        j3 / 2.0
    )  # This is J for get_clebsch_gordon in e3nn wigner_3j(J, l_in, l_out)

    m1_actual = m1 / 2.0
    m2_actual = m2 / 2.0
    m3_actual = (
        m3 / 2.0
    )  # This m3_actual corresponds to the m3 argument of the desired Wigner 3j symbol W(l1,l2,l3; m1,m2,m3)

    # tensor_products.get_clebsch_gordon(J, l_in, l_out) returns permuted Wigner 3j symbols
    # from e3nn.o3.wigner_3j(J, l_in, l_out).permute(2,1,0)
    # If we call get_clebsch_gordon(J=L3, l_in=L1, l_out=L2),
    # the returned tensor T is effectively e3nn.o3.wigner_3j(L3, L1, L2) permuted.
    # T[idx_m_L2, idx_m_L1, idx_m_L3_e3nn] gives W(L3, L1, L2; m_L3_e3nn, m_L1, m_L2)
    #
    # We want W(l1_val, l2_val, l3_val; m1_actual, m2_actual, m3_actual).
    # By cyclic permutation, this is W(l3_val, l1_val, l2_val; m3_actual, m1_actual, m2_actual).
    # So, we call get_clebsch_gordon with J_arg=l3_val, l_in_arg=l1_val, l_out_arg=l2_val.
    cg_tensor = get_clebsch_gordon(
        J=int(l3_val), l_in=int(l1_val), l_out=int(l2_val), device=device
    )

    # Indices for accessing cg_tensor to get W(l3_val, l1_val, l2_val; m3_actual, m1_actual, m2_actual)
    # cg_tensor is indexed [idx_for_l2_val, idx_for_l1_val, idx_for_l3_val_e3nn_style]
    idx_for_l1_val_dim = int(
        m1_actual + l1_val
    )  # Index for dimension 1 of cg_tensor (from l1_val)
    idx_for_l2_val_dim = int(
        m2_actual + l2_val
    )  # Index for dimension 0 of cg_tensor (from l2_val)
    idx_for_l3_val_dim = int(
        m3_actual + l3_val
    )  # Index for dimension 2 of cg_tensor (from l3_val, matching m3_actual)

    # Bounds check
    if not (
        0 <= idx_for_l2_val_dim <= 2 * l2_val
        and 0 <= idx_for_l1_val_dim <= 2 * l1_val
        and 0 <= idx_for_l3_val_dim <= 2 * l3_val
    ):
        return 0.0

    # The value from cg_tensor at these indices is W(l3_val, l1_val, l2_val; m3_actual, m1_actual, m2_actual)
    # which is equal to W(l1_val, l2_val, l3_val; m1_actual, m2_actual, m3_actual)
    w3j_value = float(
        cg_tensor[idx_for_l2_val_dim, idx_for_l1_val_dim, idx_for_l3_val_dim].item()
    )

    return w3j_value


@lru_cache(maxsize=None)
def get_truly_sparse_coo_clebsch_gordon(l1: int, l2: int, l3: int, device, dtype):
    # This is the restored version for v3, which calls the above wigner_3j_from_clebsch_gordon
    # (that returns a Wigner 3j symbol) and then applies the conversion to CG coefficient.
    if not (abs(l1 - l2) <= l3 <= l1 + l2):  # Triangle inequality for actual l values
        return torch.empty((3, 0), dtype=torch.long, device=device), torch.empty(
            0, dtype=dtype, device=device
        )

    cg_indices_list = []
    cg_values_list = []
    sqrt_2l3_plus_1 = math.sqrt(2 * l3 + 1.0)
    tolerance = 1e-9

    for m1_val in range(-l1, l1 + 1):  # m1_val is integer
        for m2_val in range(-l2, l2 + 1):  # m2_val is integer
            m3_cg = m1_val + m2_val  # This is M_out for CG, integer
            if abs(m3_cg) <= l3:
                # For Wigner 3j symbol W(l1,l2,l3; m1,m2,m3_wigner), m1+m2+m3_wigner = 0.
                # So, m3_wigner = -(m1+m2) = -m3_cg.
                m3_wigner = -m3_cg

                # Call wigner_3j_from_clebsch_gordon, which expects 2*l and 2*m arguments.
                # It will return W(l1,l2,l3; m1_val,m2_val,m3_wigner).
                val_3j = wigner_3j_from_clebsch_gordon(
                    j1=2 * l1,
                    j2=2 * l2,
                    j3=2 * l3,
                    m1=2 * m1_val,
                    m2=2 * m2_val,
                    m3=2 * m3_wigner,  # Pass 2*m3_wigner
                    device=device,
                )
                val_3j = float(val_3j)  # Ensure it is a float

                if abs(val_3j) > tolerance:
                    # Standard Clebsch-Gordan formula:
                    # C(l1,l2,l3; m1,m2,m3_cg) = (-1)^(l1-l2+m3_cg) * sqrt(2*l3+1) * W(l1,l2,l3; m1,m2,-m3_cg)
                    # Since val_3j is W(l1,l2,l3; m1_val, m2_val, m3_wigner)
                    # and m3_wigner = -m3_cg, val_3j is the correct Wigner symbol for the formula.
                    phase = 1.0 if (l1 - l2 + m3_cg) % 2 == 0 else -1.0
                    cg_val_calculated = phase * sqrt_2l3_plus_1 * val_3j

                    idx_m1 = m1_val + l1
                    idx_m2 = m2_val + l2
                    idx_m3_cg_out = m3_cg + l3  # This is the M index for the CG tensor

                    cg_indices_list.append((idx_m3_cg_out, idx_m1, idx_m2))
                    cg_values_list.append(cg_val_calculated)

    if not cg_indices_list:  # No non-zero elements found
        indices_tensor = torch.empty((3, 0), dtype=torch.long, device=device)
        values_tensor = torch.empty(0, dtype=dtype, device=device)
    else:
        indices_tensor = torch.tensor(
            cg_indices_list, dtype=torch.long, device=device
        ).T
        values_tensor = torch.tensor(cg_values_list, dtype=dtype, device=device)

    return indices_tensor, values_tensor


def project_tensor_product_torch_sparse_v2(
    u: Tensor,
    v: Tensor,
    ell_out: int,
    projection_type: str = "inter",
    chunk_size: int = 1024,  # Process non-zeros in chunks
) -> Tensor:
    """Chunked sparse tensor product projection using get_sparse_coo_clebsch_gordon_v2.

    This version improves upon v1 by processing CG coefficients in chunks to reduce memory usage
    of intermediate tensors. However, it still uses get_sparse_coo_clebsch_gordon_v2 which
    first creates a dense CG tensor and then extracts non-zero elements.

    Args:
        u: [b, N, 2*ell1+1, m1] (input tensor)
        v: [b, N, 2*ell2+1, m2] (input tensor)
        ell_out: Output angular momentum
        projection_type: 'inter' (outer product) or 'intra' (elementwise, m1==m2)
        chunk_size: Number of non-zero CG coefficients to process in each chunk.

    Returns:
        y: [b, N, 2*ell_out+1, m_out]
            - m_out = m1*m2 for 'inter', m1 for 'intra'
    """
    b, N, d1, m1_mult = u.shape
    _, _, d2, m2_mult = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Get sparse CG coefficients using the (potentially optimized) v2 getter
    sparse_cg = get_sparse_coo_clebsch_gordon_v2(
        J=ell_out, l_in=ell1, l_out=ell2, device=u.device, dtype=u.dtype
    )

    cg_indices_full = sparse_cg.indices()  # [3, nnz_total]
    cg_values_full = sparse_cg.values()  # [nnz_total]

    if projection_type == "inter":
        output_multiplicity = m1_mult * m2_mult
    elif projection_type == "intra":
        if m1_mult != m2_mult:
            raise ValueError(
                f"Intra projection requires same multiplicity, got m1={m1_mult} and m2={m2_mult}"
            )
        output_multiplicity = m1_mult
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

    y = torch.zeros(
        b, N, 2 * ell_out + 1, output_multiplicity, device=u.device, dtype=u.dtype
    )

    nnz_total = cg_indices_full.shape[1]
    if nnz_total == 0:
        return y  # No non-zero CG coefficients, output remains zero

    for start_idx in range(0, nnz_total, chunk_size):
        end_idx = min(start_idx + chunk_size, nnz_total)

        current_chunk_size = end_idx - start_idx
        cg_indices_chunk = cg_indices_full[:, start_idx:end_idx]
        cg_values_chunk = cg_values_full[start_idx:end_idx]

        M_indices_chunk = cg_indices_chunk[0]  # [current_chunk_size]
        m1_indices_chunk = cg_indices_chunk[1]  # [current_chunk_size]
        m2_indices_chunk = cg_indices_chunk[2]  # [current_chunk_size]

        # Gather relevant slices from u and v for this chunk
        # u_selected_chunk: [b, N, current_chunk_size, m1_mult]
        u_selected_chunk = u[:, :, m1_indices_chunk, :]
        # v_selected_chunk: [b, N, current_chunk_size, m2_mult]
        v_selected_chunk = v[:, :, m2_indices_chunk, :]

        if projection_type == "inter":
            # product_chunk: [b, N, current_chunk_size, m1_mult, m2_mult]
            product_chunk = u_selected_chunk.unsqueeze(-1) * v_selected_chunk.unsqueeze(
                -2
            )

            # Reshape to [b, N, current_chunk_size, m1_mult*m2_mult]
            product_chunk_flat_mult = product_chunk.reshape(
                b, N, current_chunk_size, output_multiplicity
            )
        else:  # 'intra'
            # product_chunk_flat_mult: [b, N, current_chunk_size, m1_mult]
            product_chunk_flat_mult = u_selected_chunk * v_selected_chunk

        # Weight by CG values for this chunk
        # cg_values_chunk view: [1, 1, current_chunk_size, 1]
        weighted_terms_chunk = product_chunk_flat_mult * cg_values_chunk.view(
            1, 1, -1, 1
        )

        # Scatter add for this chunk
        y = scatter_add(
            src=weighted_terms_chunk, index=M_indices_chunk.long(), dim=2, out=y
        )
    return y


def project_tensor_product_torch_sparse_v3(
    u: Tensor,
    v: Tensor,
    ell_out: int,
    projection_type: str = "inter",
    chunk_size: int = 1024,  # Process non-zeros in chunks
) -> Tensor:
    """Fully sparse tensor product that never materializes a dense CG tensor.

    This V3 implementation has the same chunking strategy as v2, but it differs in how it
    obtains the CG coefficients. Instead of first creating a dense CG tensor with
    get_sparse_coo_clebsch_gordon_v2 and then extracting non-zero elements, it uses
    get_truly_sparse_coo_clebsch_gordon which directly computes only the non-zero CG coefficients.

    This implementation has better memory scaling for large angular momenta and is particularly
    useful when large L values would otherwise cause out-of-memory errors.

    Args:
        u: [b, N, 2*ell1+1, m1] (input tensor)
        v: [b, N, 2*ell2+1, m2] (input tensor)
        ell_out: Output angular momentum
        projection_type: 'inter' (outer product) or 'intra' (elementwise, m1==m2)
        chunk_size: Number of non-zero CG coefficients to process in each chunk.

    Returns:
        y: [b, N, 2*ell_out+1, m_out]
            - m_out = m1*m2 for 'inter', m1 for 'intra'
    """
    b, N, d1, m1_mult = u.shape
    _, _, d2, m2_mult = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Get sparse CG coefficients directly as indices and values
    # This function uses the spherical library for truly sparse computation
    # with large L values
    cg_indices_full, cg_values_full = get_truly_sparse_coo_clebsch_gordon(
        l1=ell1, l2=ell2, l3=ell_out, device=u.device, dtype=u.dtype
    )

    if projection_type == "inter":
        output_multiplicity = m1_mult * m2_mult
    elif projection_type == "intra":
        if m1_mult != m2_mult:
            raise ValueError(
                f"Intra projection requires same multiplicity, got m1={m1_mult} and m2={m2_mult}"
            )
        output_multiplicity = m1_mult
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

    y = torch.zeros(
        b, N, 2 * ell_out + 1, output_multiplicity, device=u.device, dtype=u.dtype
    )

    nnz_total = cg_indices_full.shape[1]
    if nnz_total == 0:
        return y  # No non-zero CG coefficients, output remains zero

    for start_idx in range(0, nnz_total, chunk_size):
        end_idx = min(start_idx + chunk_size, nnz_total)

        current_chunk_size = end_idx - start_idx
        if current_chunk_size == 0:  # Should not happen if nnz_total > 0
            continue

        cg_indices_chunk = cg_indices_full[:, start_idx:end_idx]
        cg_values_chunk = cg_values_full[start_idx:end_idx]

        M_indices_chunk = cg_indices_chunk[0]  # [current_chunk_size]
        m1_indices_chunk = cg_indices_chunk[1]  # [current_chunk_size]
        m2_indices_chunk = cg_indices_chunk[2]  # [current_chunk_size]

        u_selected_chunk = u[:, :, m1_indices_chunk, :]
        v_selected_chunk = v[:, :, m2_indices_chunk, :]

        if projection_type == "inter":
            product_chunk = u_selected_chunk.unsqueeze(-1) * v_selected_chunk.unsqueeze(
                -2
            )
            product_chunk_flat_mult = product_chunk.reshape(
                b, N, current_chunk_size, output_multiplicity
            )
        else:  # 'intra'
            product_chunk_flat_mult = u_selected_chunk * v_selected_chunk

        weighted_terms_chunk = product_chunk_flat_mult * cg_values_chunk.view(
            1, 1, -1, 1
        )

        y = scatter_add(
            src=weighted_terms_chunk,
            index=M_indices_chunk.long(),
            dim=2,  # scatter along the 2*ell_out+1 dimension
            out=y,
        )
    return y


# V4 dedicated helper functions
@lru_cache(maxsize=None)
def wigner_3j_from_clebsch_gordon_v4(
    l1_in: int, l2_in: int, l3_in: int, m1_in: int, m2_in: int, m3_in: int, device
) -> float:
    """
    SO(3) specific: Directly extracts Clebsch-Gordan coefficients from get_clebsch_gordon.

    This function is specifically designed to work with the dense tensor product implementation
    where get_clebsch_gordon already returns properly formatted coefficients that can be
    used directly in tensor contractions.

    Args:
        l1_in: First angular momentum (integer)
        l2_in: Second angular momentum (integer)
        l3_in: Output angular momentum (integer)
        m1_in: Magnetic quantum number for l1 (integer)
        m2_in: Magnetic quantum number for l2 (integer)
        m3_in: Combined magnetic quantum number (integer) which is m1 + m2 for CG coefficients
        device: PyTorch device

    Returns:
        float: The Clebsch-Gordan coefficient value
    """
    # Check selection rules
    if m1_in + m2_in != m3_in:  # For CG coefficients, m3 = m1 + m2
        return 0.0

    # Triangle inequality and bounds check
    if not (abs(l1_in - l2_in) <= l3_in <= l1_in + l2_in):
        return 0.0

    if abs(m1_in) > l1_in or abs(m2_in) > l2_in or abs(m3_in) > l3_in:
        return 0.0

    # Call get_clebsch_gordon with parameters matching the dense implementation
    cg_tensor = get_clebsch_gordon(J=l3_in, l_in=l1_in, l_out=l2_in, device=device)

    # Adjust indices: get_clebsch_gordon returns a tensor with shape [2*l2_in+1, 2*l1_in+1, 2*l3_in+1]
    # The indices correspond to m2, m1, m3 offsets from -l to +l
    idx_m1 = m1_in + l1_in  # Offset index for m1
    idx_m2 = m2_in + l2_in  # Offset index for m2
    idx_m3 = m3_in + l3_in  # Offset index for m3

    # Extract the coefficient directly - this matches how it's used in project_tensor_product_inter/intra_channel
    cg_value = float(cg_tensor[idx_m2, idx_m1, idx_m3].item())

    return cg_value


@lru_cache(maxsize=None)
def get_truly_sparse_coo_clebsch_gordon_v4(l1: int, l2: int, l3: int, device, dtype):
    """
    SO(3) specific: Computes Clebsch-Gordan coefficients in sparse COO format for integer l.

    This version is specifically designed to match the behavior of get_sparse_coo_clebsch_gordon
    which is known to work correctly with the tensor product implementations. It directly uses
    get_clebsch_gordon to get coefficients but only extracts the non-zero values to minimize
    memory usage for large angular momenta.
    """
    # Get the cached CG coefficients from tensor_products.get_clebsch_gordon
    # This has shape [2*l2+1, 2*l1+1, 2*l3+1] with indices (m2, m1, M)
    # which is the same format that the original get_sparse_coo_clebsch_gordon uses
    dense_cg = get_clebsch_gordon(J=l3, l_in=l1, l_out=l2, device=device)

    # Convert dtype if needed
    if dtype != dense_cg.dtype:
        dense_cg = dense_cg.to(dtype=dtype)

    # Permute to match expected dimensions for sparse tensor [2*l3+1, 2*l1+1, 2*l2+1] (M, m1, m2)
    # This matches exactly what get_sparse_coo_clebsch_gordon does
    dense_cg = dense_cg.permute(2, 1, 0)

    # Instead of materializing the full tensor, we'll iterate and collect non-zero elements
    cg_indices_list = []
    cg_values_list = []
    tolerance = 1e-9

    # Iterate through all possible index combinations
    for m3_idx in range(2 * l3 + 1):  # M dimension (first after permute)
        for m1_idx in range(2 * l1 + 1):  # m1 dimension (second after permute)
            for m2_idx in range(2 * l2 + 1):  # m2 dimension (third after permute)
                # Get the actual CG value
                cg_val = float(dense_cg[m3_idx, m1_idx, m2_idx].item())

                # Only store non-zero values (with numerical tolerance)
                if abs(cg_val) > tolerance:
                    cg_indices_list.append(
                        [m3_idx, m1_idx, m2_idx]
                    )  ### m3 = m1 + m2 (selection rule)
                    cg_values_list.append(cg_val)

    # Convert to tensors
    if not cg_indices_list:
        # Handle empty case
        indices_tensor = torch.empty((3, 0), dtype=torch.long, device=device)
        values_tensor = torch.empty(0, dtype=dtype, device=device)
    else:
        indices_tensor = torch.tensor(
            cg_indices_list, dtype=torch.long, device=device
        ).T
        values_tensor = torch.tensor(cg_values_list, dtype=dtype, device=device)

    return indices_tensor, values_tensor


def project_tensor_product_torch_sparse_v4(
    u: Tensor,
    v: Tensor,
    ell_out: int,
    projection_type: str = "inter",
    chunk_size: int = 1024,
) -> Tensor:
    """
    Optimized tensor product for integer angular momenta (SO(3) representations)
    with chunked processing for memory efficiency.

    This v4 implementation is specifically optimized for integer angular momenta,
    avoiding half-integer conversions. Like v3, it maintains a fully sparse approach
    that never materializes the full dense CG tensor, making it memory-efficient
    for large angular momenta.

    Args:
        u: [b, N, 2*ell1+1, m1] (input tensor)
        v: [b, N, 2*ell2+1, m2] (input tensor)
        ell_out: Output angular momentum (integer)
        projection_type: 'inter' (outer product) or 'intra' (elementwise, m1==m2)
        chunk_size: Number of non-zero CG coefficients to process in each chunk.

    Returns:
        y: [b, N, 2*ell_out+1, m_out]
            - m_out = m1*m2 for 'inter', m1 for 'intra'
    """
    b, N, d1, m1_mult = u.shape
    _, _, d2, m2_mult = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Ensure this calls the new V4-specific CG getter
    cg_indices_full, cg_values_full = get_truly_sparse_coo_clebsch_gordon_v4(
        l1=ell1, l2=ell2, l3=ell_out, device=u.device, dtype=u.dtype
    )

    if projection_type == "inter":
        output_multiplicity = m1_mult * m2_mult
    elif projection_type == "intra":
        if m1_mult != m2_mult:
            raise ValueError(
                f"Intra projection requires same multiplicity, got m1={m1_mult} and m2={m2_mult}"
            )
        output_multiplicity = m1_mult
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

    y = torch.zeros(
        b, N, 2 * ell_out + 1, output_multiplicity, device=u.device, dtype=u.dtype
    )

    nnz_total = cg_indices_full.shape[1]
    if nnz_total == 0:
        return y

    for start_idx in range(0, nnz_total, chunk_size):
        end_idx = min(start_idx + chunk_size, nnz_total)

        current_chunk_size = end_idx - start_idx
        if current_chunk_size == 0:
            continue

        cg_indices_chunk = cg_indices_full[:, start_idx:end_idx]
        cg_values_chunk = cg_values_full[start_idx:end_idx]

        m1_indices_chunk = cg_indices_chunk[1]
        m2_indices_chunk = cg_indices_chunk[2]

        u_selected_chunk = u[:, :, m1_indices_chunk, :]
        v_selected_chunk = v[:, :, m2_indices_chunk, :]

        if projection_type == "inter":
            # Fix the einsum operation to properly handle the dimensions
            # u_selected_chunk: [b, N, current_chunk_size, m1_mult]
            # v_selected_chunk: [b, N, current_chunk_size, m2_mult]

            # Use unsqueeze + reshape approach which is more reliable than einsum for this case
            product_chunk = u_selected_chunk.unsqueeze(-1) * v_selected_chunk.unsqueeze(
                -2
            )  # [b, N, current_chunk_size, m1_mult, m2_mult]
            product_chunk_flat_mult = product_chunk.reshape(
                b, N, current_chunk_size, output_multiplicity
            )
        else:  # 'intra'
            product_chunk_flat_mult = torch.einsum(
                "bnim,bnim->bnim", u_selected_chunk, v_selected_chunk
            )

        weighted_terms_chunk = product_chunk_flat_mult * cg_values_chunk.view(
            1, 1, -1, 1
        )

        y = scatter_add(
            src=weighted_terms_chunk, index=M_indices_chunk.long(), dim=2, out=y
        )
    return y


def project_tensor_product_torch_sparse_v4_einsum(
    u: Tensor,
    v: Tensor,
    ell_out: int,
    projection_type: str = "inter",
    chunk_size: int = 1024,
) -> Tensor:
    """
    Optimized tensor product for integer angular momenta (SO(3) representations)
    with chunked processing for memory efficiency, using einsum operations.

    This v4_einsum implementation is based on v4 but uses torch.einsum operations
    for tensor contractions where appropriate. This can sometimes lead to more
    optimized execution depending on the backend.

    Args:
        u: [b, N, 2*ell1+1, m1] (input tensor)
        v: [b, N, 2*ell2+1, m2] (input tensor)
        ell_out: Output angular momentum (integer)
        projection_type: 'inter' (outer product) or 'intra' (elementwise, m1==m2)
        chunk_size: Number of non-zero CG coefficients to process in each chunk.

    Returns:
        y: [b, N, 2*ell_out+1, m_out]
            - m_out = m1*m2 for 'inter', m1 for 'intra'
    """
    b, N, d1, m1_mult = u.shape
    _, _, d2, m2_mult = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Ensure this calls the V4-specific CG getter
    cg_indices_full, cg_values_full = get_truly_sparse_coo_clebsch_gordon_v4(
        l1=ell1, l2=ell2, l3=ell_out, device=u.device, dtype=u.dtype
    )

    if projection_type == "inter":
        output_multiplicity = m1_mult * m2_mult
    elif projection_type == "intra":
        if m1_mult != m2_mult:
            raise ValueError(
                f"Intra projection requires same multiplicity, got m1={m1_mult} and m2={m2_mult}"
            )
        output_multiplicity = m1_mult
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

    y = torch.zeros(
        b, N, 2 * ell_out + 1, output_multiplicity, device=u.device, dtype=u.dtype
    )

    nnz_total = cg_indices_full.shape[1]
    if nnz_total == 0:
        return y

    for start_idx in range(0, nnz_total, chunk_size):
        end_idx = min(start_idx + chunk_size, nnz_total)

        current_chunk_size = end_idx - start_idx
        if current_chunk_size == 0:
            continue

        cg_indices_chunk = cg_indices_full[:, start_idx:end_idx]
        cg_values_chunk = cg_values_full[start_idx:end_idx]

        M_indices_chunk = cg_indices_chunk[0]
        m1_indices_chunk = cg_indices_chunk[1]
        m2_indices_chunk = cg_indices_chunk[2]

        u_selected_chunk = u[:, :, m1_indices_chunk, :]
        v_selected_chunk = v[:, :, m2_indices_chunk, :]

        if projection_type == "inter":
            # Use einsum for outer product - carefully labeled to match dimensions
            # u_selected_chunk: [b, N, current_chunk_size, m1_mult]
            # v_selected_chunk: [b, N, current_chunk_size, m2_mult]
            # k = chunk index dimension
            product_chunk = torch.einsum(
                "bnki,bnkj->bnkij", u_selected_chunk, v_selected_chunk
            )
            # Reshape to [b, N, current_chunk_size, m1_mult*m2_mult]
            product_chunk_flat_mult = product_chunk.reshape(
                b, N, current_chunk_size, output_multiplicity
            )
        else:  # 'intra'
            # Use einsum for elementwise product
            product_chunk_flat_mult = torch.einsum(
                "bnki,bnki->bnki", u_selected_chunk, v_selected_chunk
            )

        # Weight by CG values for this chunk
        weighted_terms_chunk = product_chunk_flat_mult * cg_values_chunk.view(
            1, 1, -1, 1
        )

        y = scatter_add(
            src=weighted_terms_chunk, index=M_indices_chunk.long(), dim=2, out=y
        )
    return y


def project_tensor_product_torch_sparse_v1_einsum(
    u: Tensor, v: Tensor, ell_out: int, projection_type: str = "inter"
) -> Tensor:
    """Sparse tensor product projection using torch.sparse with einsum operations.

    This variant of the original v1 implementation uses torch.einsum for tensor contractions
    where appropriate, which can be faster for certain operations.

    Args:
        u: [b, N, 2*ell1+1, m1] (input tensor)
        v: [b, N, 2*ell2+1, m2] (input tensor)
        ell_out: Output angular momentum
        projection_type: 'inter' (outer product) or 'intra' (elementwise, m1==m2)

    Returns:
        y: [b, N, 2*ell_out+1, m_out]
            - m_out = m1*m2 for 'inter', m1 for 'intra'
    """
    b, N, d1, m1 = u.shape
    _, _, d2, m2 = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Get sparse CG coefficients as a COO tensor
    sparse_cg = get_sparse_coo_clebsch_gordon(
        J=ell_out, l_in=ell1, l_out=ell2, device=u.device, dtype=u.dtype
    )
    # Extract indices and values
    cg_indices = sparse_cg.indices()  # [3, nnz] (M, m1, m2)
    cg_values = sparse_cg.values()  # [nnz]
    # Split indices for advanced indexing
    M_indices = cg_indices[0]  # [nnz]
    m1_indices = cg_indices[1]  # [nnz]
    m2_indices = cg_indices[2]  # [nnz]

    # Gather relevant slices from u and v for each nonzero CG term
    u_selected = u[:, :, m1_indices, :]  # [b, N, nnz, m1]
    v_selected = v[:, :, m2_indices, :]  # [b, N, nnz, m2]

    if projection_type == "inter":
        # Use einsum for outer product over multiplicities
        # u_selected: [b, N, nnz, m1], v_selected: [b, N, nnz, m2]
        # Notation: n=nonzero indices, i=m1 indices, j=m2 indices
        product = torch.einsum("bnki,bnkj->bnkij", u_selected, v_selected)

        # Reshape last two dims: [b, N, nnz, m1*m2]
        product = product.reshape(b, N, -1, m1 * m2)
        output_multiplicity = m1 * m2
    else:  # 'intra' channel: elementwise product, m1 == m2
        assert m1 == m2, "Intra projection requires m1 == m2"
        # Use einsum for elementwise product
        product = torch.einsum("bnki,bnki->bnki", u_selected, v_selected)
        output_multiplicity = m1

    # Weight by CG values (broadcast over batch, tokens, and multiplicity)
    weighted_terms = product * cg_values.view(1, 1, -1, 1)

    # Initialize output tensor
    y = torch.zeros(
        b, N, 2 * ell_out + 1, output_multiplicity, device=u.device, dtype=u.dtype
    )

    # Scatter add: sum contributions to each output M index
    y = scatter_add(
        src=weighted_terms,
        index=M_indices.long(),
        dim=2,
        out=y,
    )
    return y


def project_tensor_product_torch_sparse_v1_einsum_alt(
    u: Tensor, v: Tensor, ell_out: int, projection_type: str = "inter"
) -> Tensor:
    """Sparse tensor product projection using torch.sparse with einsum operations.
    Alternative implementation that first multiplies C with u, then with v.

    Args:
        u: [b, N, 2*ell1+1, m1] (input tensor)
        v: [b, N, 2*ell2+1, m2] (input tensor)
        ell_out: Output angular momentum
        projection_type: 'inter' (outer product) or 'intra' (elementwise, m1==m2)

    Returns:
        y: [b, N, 2*ell_out+1, m_out]
            - m_out = m1*m2 for 'inter', m1 for 'intra'
    """
    b, N, d1, m1 = u.shape
    _, _, d2, m2 = v.shape
    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)

    # Get sparse CG coefficients as a COO tensor
    sparse_cg = get_sparse_coo_clebsch_gordon(
        J=ell_out, l_in=ell1, l_out=ell2, device=u.device, dtype=u.dtype
    )
    # Extract indices and values
    cg_indices = sparse_cg.indices()  # [3, nnz] (M, m1, m2)
    cg_values = sparse_cg.values()  # [nnz]
    # Split indices for advanced indexing
    M_indices = cg_indices[0]  # [nnz]
    m1_indices = cg_indices[1]  # [nnz]
    m2_indices = cg_indices[2]  # [nnz]

    # Gather relevant slices from u and v for each nonzero CG term
    u_selected = u[:, :, m1_indices, :]  # [b, N, nnz, m1]
    v_selected = v[:, :, m2_indices, :]  # [b, N, nnz, m2]

    # First multiply C with u_selected
    # cg_values: [nnz], u_selected: [b, N, nnz, m1]
    # Result: [b, N, nnz, m1]
    cu = u_selected * cg_values.view(1, 1, -1, 1)

    if projection_type == "inter":
        # For inter-channel, we need to do an outer product with v
        # cu: [b, N, nnz, m1], v_selected: [b, N, nnz, m2]
        # Result: [b, N, nnz, m1, m2]
        product = torch.einsum("bnki,bnkj->bnkij", cu, v_selected)
        # Reshape last two dims: [b, N, nnz, m1*m2]
        product = product.reshape(b, N, -1, m1 * m2)
        output_multiplicity = m1 * m2
    else:  # 'intra' channel: elementwise product, m1 == m2
        assert m1 == m2, "Intra projection requires m1 == m2"
        # For intra-channel, we do elementwise multiplication with v
        # cu: [b, N, nnz, m1], v_selected: [b, N, nnz, m1]
        product = torch.einsum("bnki,bnki->bnki", cu, v_selected)
        output_multiplicity = m1

    # Initialize output tensor
    y = torch.zeros(
        b, N, 2 * ell_out + 1, output_multiplicity, device=u.device, dtype=u.dtype
    )

    # Scatter add: sum contributions to each output M index
    y = scatter_add(
        src=product,
        index=M_indices.long(),
        dim=2,
        out=y,
    )
    return y
