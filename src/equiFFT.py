import torch
from src.tensor_products import (
    project_tensor_product_intra_channel,
    project_tensor_product_inter_channel,
    project_tensor_product,
)
from src.sparse_tensor_products import project_tensor_product_torch_sparse


def equiFFT(
    u: torch.Tensor,
    v: torch.Tensor,
    ell_out: int,
    projection_type: str = "intra",
    use_sparse: bool = True,
):
    """
    Computes y_i = sum_k C [ u_{i - k} ⊗ v_k ] using FFT
    Projections are either inter or intra type
    Inputs:
        u: [b, N, 2l1+1, m1]
        v: [b, N, 2l2+1, m2]
    Returns:
     inter type   y: [b, N, 2lout+1, m1*m2]
     intra type   y: [b, N, 2lout+1, m1]
    """

    m1 = u.shape[3]
    m2 = v.shape[3]

    if type == "intra":
        assert (
            m1 == m2
        ), "For intra channel convolution multiplicity dimension must be the same!"

    u_fft = torch.fft.fft(u, dim=1)
    v_fft = torch.fft.fft(v, dim=1)

    y_fft = ClebschGordonProjection(
        u_fft, v_fft, ell_out, projection_type=projection_type, use_sparse=use_sparse
    )

    y = torch.fft.ifft(y_fft, dim=1)
    return y.real


def ClebschGordonProjection(
    u: torch.Tensor,
    v: torch.Tensor,
    ell_out: int,
    projection_type: str = "inter",
    use_sparse: bool = True,
):
    """
    Computes y_i = sum_k C [ u_{i - k} ⊗ v_k ] using FFT
    Projections are either inter or intra type
    Inputs:
        u: [b, N, 2l1+1, m1]
        v: [b, N, 2l2+1, m2]
    Returns:
     inter type   y: [b, N, 2lout+1, m1*m2]
     intra type   y: [b, N, 2lout+1, m1]
    """

    m1 = u.shape[3]
    m2 = v.shape[3]

    if projection_type == "intra":
        assert (
            m1 == m2
        ), "For intra channel convolution multiplicity dimension must be the same!"

    if use_sparse == True:
        y = project_tensor_product_torch_sparse(
            u, v, ell_out, projection_type=projection_type
        )

    else:
        y = project_tensor_product(u, v, ell_out, type=projection_type)

    return y


def generic_equiFFT(
    u, v, ell_out, tensor_product_fn, projection_type="intra", **tp_kwargs
):
    """
    Generic equivariant convolution using FFT with any tensor product function.
    This function accepts any tensor product implementation and handles the FFT/IFFT operations.

    Args:
        u (Tensor): Input tensor 1 of shape [b, N, 2*ell1+1, m1]
        v (Tensor): Input tensor 2 of shape [b, N, 2*ell2+1, m2]
        ell_out (int): Output angular momentum
        tensor_product_fn (callable): The tensor product function to use
        projection_type (str): Type of projection, "inter" or "intra"
        **tp_kwargs: Additional keyword arguments for the tensor product function

    Returns:
        Tensor: Result of convolution, shape depends on tensor product function and inputs
    """
    # Check dimensions for intra projection
    if projection_type == "intra" and u.shape[3] != v.shape[3]:
        raise ValueError(
            f"For intra projection, u and v must have same multiplicity, got {u.shape[3]} and {v.shape[3]}"
        )

    # Compute FFT along the sequence dimension
    u_fft = torch.fft.fft(u, dim=1)
    v_fft = torch.fft.fft(v, dim=1)

    # Get function name for special handling
    fn_name = tensor_product_fn.__name__

    # Special handling for e3nn which can't handle complex inputs directly
    if fn_name == "project_tensor_product_e3nn":
        # For e3nn, handle real and imaginary parts separately
        # 1. Process real parts
        u_fft_real = u_fft.real.contiguous()
        v_fft_real = v_fft.real.contiguous()
        y_fft_real = tensor_product_fn(
            u_fft_real,
            v_fft_real,
            ell_out,
            projection_type=projection_type,
            **tp_kwargs,
        )

        # 2. Process imaginary parts
        u_fft_imag = u_fft.imag.contiguous()
        v_fft_imag = v_fft.imag.contiguous()

        # Complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        # We need the imaginary part: ad + bc
        y_fft_imag_part1 = tensor_product_fn(
            u_fft_real,
            v_fft_imag,
            ell_out,
            projection_type=projection_type,
            **tp_kwargs,
        )
        y_fft_imag_part2 = tensor_product_fn(
            u_fft_imag,
            v_fft_real,
            ell_out,
            projection_type=projection_type,
            **tp_kwargs,
        )
        y_fft_imag = y_fft_imag_part1 + y_fft_imag_part2

        # 3. Reconstruct complex tensor
        y_fft = torch.complex(y_fft_real, y_fft_imag)
    elif fn_name == "sparse_project_tensor_product":
        # This function needs ell1 and ell2 explicitly
        ell1 = (u.shape[2] - 1) // 2
        ell2 = (v.shape[2] - 1) // 2
        y_fft = tensor_product_fn(u_fft, v_fft, ell1, ell2, ell_out, **tp_kwargs)
    elif fn_name == "project_tensor_product":
        # This function takes projection_type as 'type' parameter
        y_fft = tensor_product_fn(
            u_fft, v_fft, ell_out, type=projection_type, **tp_kwargs
        )
    elif fn_name == "project_tensor_product_inter_channel":
        # This function doesn't take projection_type, always inter
        y_fft = tensor_product_fn(u_fft, v_fft, ell_out, **tp_kwargs)
    elif fn_name == "project_tensor_product_intra_channel":
        # This function doesn't take projection_type, always intra
        y_fft = tensor_product_fn(u_fft, v_fft, ell_out, **tp_kwargs)
    else:
        # For all other tensor product functions
        if "projection_type" in tp_kwargs:
            # If already in kwargs, don't add it again
            y_fft = tensor_product_fn(u_fft, v_fft, ell_out, **tp_kwargs)
        else:
            # Standard case for most functions
            y_fft = tensor_product_fn(
                u_fft, v_fft, ell_out, projection_type=projection_type, **tp_kwargs
            )

    # Compute inverse FFT and return real part
    y = torch.fft.ifft(y_fft, dim=1)

    return y.real
