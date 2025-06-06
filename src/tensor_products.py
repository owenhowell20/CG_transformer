from itertools import tee
from typing import List, Tuple

from sympy import cancel
import e3nn.o3 as o3
from torch import Tensor
from functools import lru_cache
import torch


@lru_cache(maxsize=None)
def get_clebsch_gordon(J: int, l_in: int, l_out: int, device) -> Tensor:
    """Get the Q^{d_out,d_in}_J matrices from equation (8)

    returns: shape ~ [ 2*l_out + 1, 2*l_in + 1, 2*J+1]

    Args:
        J: Output angular momentum (int)
        l_in: Input angular momentum (int)
        l_out: Second input angular momentum (int)
        device: PyTorch device

    Returns:
        Tensor of shape [2*l_out+1, 2*l_in+1, 2*J+1] (real, double)
        - Indices: (m2, m1, M)
    """
    # Compute CG coefficients using e3nn (permute to [m2, m1, M])
    vals = o3.wigner_3j(J, l_in, l_out, dtype=torch.float64, device=device).permute(
        2, 1, 0
    )
    assert vals.shape[0] == 2 * l_out + 1
    assert vals.shape[1] == 2 * l_in + 1
    assert vals.shape[2] == 2 * J + 1

    return vals


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int, device) -> List[List[Tensor]]:
    """
    Compute all CG coefficient tensors for all combinations up to max_degree.
    Returns a nested list of tensors for each (d_in, d_out, J).
    """
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out, device))
            all_cb.append(K_Js)
    return all_cb


def project_tensor_product(u: Tensor, v: Tensor, ell_out: int, type="intra") -> Tensor:
    """
    Compute the projected tensor product y_i = C [ u_i ⊗ v_i ]
    Supports both inter and intra channel projections.

    Args:
        u: [b, N, 2l1+1, m1]
        v: [b, N, 2l2+1, m2]
        ell_out: output angular momentum
        type: 'inter' or 'intra'
    Returns:
        y: [b, N, 2lout+1, m1*m2] (inter) or [b, N, 2lout+1, m1] (intra)
    """
    m1 = u.shape[3]
    m2 = v.shape[3]

    if type == "intra":
        assert m1 == m2, "Intra projection must have same number of channels!"
        return project_tensor_product_intra_channel(u=u, v=v, ell_out=ell_out)
    else:
        return project_tensor_product_inter_channel(u=u, v=v, ell_out=ell_out)


def project_tensor_product_inter_channel(u: Tensor, v: Tensor, ell_out: int) -> Tensor:
    """
    Computes the inter-channel tensor product (i.e. channels talk with each other)
    The output multiplicity is the product of the two input multiplicities

    Args:
        u: [b, N, 2l1+1, m_l1]
        v: [b, N, 2l2+1, m_l2]
        ell_out: output angular momentum
    Returns:
        y: [b, N, 2l_out+1, m_l1 * m_l2]
    """
    b, N, d1, m1 = u.shape
    _, _, d2, m2 = v.shape

    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)
    # Get CG coefficients: [2l1+1, 2l2+1, 2J+1]
    C = get_clebsch_gordon(J=ell_out, l_in=ell1, l_out=ell2, device=u.device)
    C = C.to(dtype=u.dtype)
    # Compute outer product and contract with CG
    uv = torch.einsum("bnim,bnjg->bnijmg", u, v)
    y = torch.einsum("jio,bnijmg->bnomg", C, uv)

    return y.flatten(-2)


def project_tensor_product_intra_channel(u: Tensor, v: Tensor, ell_out: int) -> Tensor:
    """
    Computes tee intra channel (i.e. multiplicity dimensions do not talk with each other) projected tensor product
    inputs must have the same multiplcity dimension

    Args:
        u: [b, N, 2l1+1, m]
        v: [b, N, 2l2+1, m]
        ell_out: output angular momentum
    Returns:
        y: [b, N, 2l_out+1, m]
    """
    b, N, d1, m = u.shape
    _, _, d2, m2 = v.shape

    ell1 = int((d1 - 1) / 2)
    ell2 = int((d2 - 1) / 2)
    assert (
        m == m2
    ), "Multiplicity Dimensions must be the same for intra channel projection"
    # Get CG coefficients: [2l1+1, 2l2+1, 2J+1]
    C = get_clebsch_gordon(J=ell_out, l_in=ell1, l_out=ell2, device=u.device)
    C = C.to(dtype=u.dtype)
    # Compute elementwise product and contract with CG
    uv = torch.einsum("bnim,bnjm->bnijm", u, v)
    y = torch.einsum("jio,bnijm->bnom", C, uv)
    return y
