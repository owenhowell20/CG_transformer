import numpy as np
import torch
from scipy.special import sph_harm
from typing import Dict, Union, Optional, Tuple

# import ssht
import s2fft
import healpy as hp

ArrayLike = Union[np.ndarray, torch.Tensor]


def sht_tensor_product(
    u: torch.Tensor,
    v: torch.Tensor,
    L_max_in: int,
    L_max_out: Optional[int] = None,
    method: str = "HEALPix",
) -> torch.Tensor:
    """
    Compute the tensor product of two spherical harmonic transforms.

    Args:
        u: First input tensor of shape (batch_size, num_points, L_max_in + 1, 2*L_max_in + 1)
        v: Second input tensor of shape (batch_size, num_points, L_max_in + 1, 2*L_max_in + 1)
        L_max_in: Maximum degree of input spherical harmonics
        L_max_out: Maximum degree of output spherical harmonics. If None, defaults to L_max_in.
        method: Method to use for spherical harmonic transform ("HEALPix" or "GL")

    Returns:
        Tensor of shape (batch_size, num_points, L_max_out + 1, 2*L_max_out + 1)
    """
    if L_max_out is None:
        L_max_out = L_max_in

    b, N, d1, m1 = u.shape
    _, _, d2, m2 = v.shape

    assert d1 == d2, "u and v must have the same number of channels"
    assert m1 == m2, "u and v must have the same number of channels"

    # Convert tensors to dictionary format
    u_dict = split_tensor_to_dict(u, L_max_in)
    v_dict = split_tensor_to_dict(v, L_max_in)

    if method == "s2fft":
        #### compute SHT of u and v
        u = s2fft_sht(u, L_max_in)
        v = s2fft_sht(v, L_max_in)

        uv = u * v  # [b, N, (L+1)^2, m]

        #### compute iSHT of uv
        uv_sht = s2fft_isht(uv, L_max_out)

    elif method == "HEALPix":
        nside = 16  # HEALPix resolution parameter (gives 3,072 points)
        vecs = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))  # (3, num_points)
        points = np.stack(vecs, axis=-1)  # (num_points, 3)
        grid_points = points.shape[0]
        # HEALPix weights: all equal, sum to 4*pi
        weights = np.full(grid_points, 4 * np.pi / grid_points)

        # Convert to real space
        u = basic_isht(fourier=u_dict, directions=points)  # [num_points, b, M, m]
        v = basic_isht(fourier=v_dict, directions=points)  # [num_points, b, M, m]

        uv = u * v

        # Convert back to Fourier space
        uv_sht = basic_sht(
            samples=uv, directions=points, L_max=L_max_out, weights=weights
        )

        for l in range(L_max_out + 1):
            assert torch.allclose(
                uv_sht[l].imag, torch.zeros_like(uv_sht[l].imag), atol=1e-2
            ), f"Output for l={l} should be real"
            uv_sht[l] = uv_sht[l].real  # Convert to real tensor

    uv_sht_tensor = dict_to_tensor(uv_sht, L_max_out)
    return uv_sht_tensor


def check_format(fourier: Dict[int, ArrayLike]) -> bool:
    """
    Check if the Fourier coefficients dictionary has the correct format.

    Args:
        fourier: Dictionary mapping degree l to (B, M, 2l+1, C) array of coefficients
               where B is batch size, M is N dimension, and C is channels

    Returns:
        bool: True if format is correct, False otherwise
    """
    if not isinstance(fourier, dict):
        print("Input is not a dictionary.")
        return False

    keys = sorted(fourier.keys())

    # Check that keys are 0, 1, ..., L with no gaps
    if keys != list(range(len(keys))):
        print(f"Keys are not consecutive integers from 0 to {max(keys)}. Got: {keys}")
        return False

    for l, mat in fourier.items():
        if not isinstance(l, int):
            print(f"Key {l} is not an integer.")
            return False

        if isinstance(mat, np.ndarray):
            shape = mat.shape
        elif isinstance(mat, torch.Tensor):
            shape = mat.shape
        else:
            print(f"Value for l={l} is not a NumPy array or PyTorch tensor.")
            return False

        if len(shape) != 4:
            print(f"Value for l={l} is not 4D. Shape: {shape}")
            return False

        expected_dim = 2 * l + 1
        if shape[2] != expected_dim:
            print(f"Third dimension for l={l} is {shape[2]}, expected {expected_dim}")
            return False

    return True


def check_equal(
    fourier_1: Dict[int, torch.Tensor],
    fourier_2: Dict[int, torch.Tensor],
    eps: float = 1e-6,
) -> bool:
    """
    Check if two Fourier coefficient dictionaries are approximately equal.

    Args:
        fourier_1, fourier_2: Dictionaries mapping degree l to (2l+1, C) tensors
        eps: Tolerance for numerical comparison

    Returns:
        bool: True if dictionaries are equal within tolerance
    """
    if set(fourier_1.keys()) != set(fourier_2.keys()):
        print("Keys differ between the two dictionaries.")
        return False

    for l in fourier_1.keys():
        val1 = fourier_1[l]
        val2 = fourier_2[l]

        if not torch.allclose(val1, val2, atol=eps, rtol=eps):
            print(f"Values for l={l} differ more than tolerance {eps}.")
            return False

    return True


def get_spherical_coords(directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        directions: (N, 3) array of unit vectors

    Returns:
        Tuple of (theta, phi) arrays where:
        - theta is the polar angle θ ∈ [0, π]
        - phi is the azimuthal angle φ ∈ [-π, π]
    """
    # Ensure unit vectors
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))  # polar angle θ ∈ [0, π]
    phi = np.arctan2(y, x)  # azimuthal angle φ ∈ [-π, π]

    return theta, phi


def basic_sht(
    samples: ArrayLike,
    directions: ArrayLike,
    L_max: int,
    device: torch.device = torch.device("cpu"),
    weights: Optional[ArrayLike] = None,
) -> Dict[int, torch.Tensor]:
    """
    Compute spherical harmonic transform (SHT) of samples on the sphere.

    Args:
        samples: (N, B, M, C) array of samples at N directions, where B is batch size, M is N dimension, and C is channels
        directions: (N, 3) array of unit vectors
        L_max: maximum degree
        device: torch device for output tensors
        weights: (N,) array of quadrature weights (optional)

    Returns:
        Dict[int, torch.Tensor]: Fourier coefficients for each l with shape (B, M, 2l+1, C)
    """
    # Convert inputs to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(directions, torch.Tensor):
        directions = directions.detach().cpu().numpy()
    if weights is not None and isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    N, B, M, C = samples.shape
    theta, phi = get_spherical_coords(directions)

    # Default quadrature weights if none provided
    if weights is None:
        weights = np.sin(theta)
        weights *= (4 * np.pi) / weights.sum()

    fourier: Dict[int, torch.Tensor] = {}

    for l in range(L_max + 1):
        coeffs_l = np.zeros((B, M, 2 * l + 1, C), dtype=np.complex128)
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)[
                :, None, None, None
            ]  # (N, 1, 1, 1), complex
            weighted_samples = (
                samples * np.conj(Y_lm) * weights[:, None, None, None]
            )  # (N, B, M, C)
            coeffs_l[:, :, m + l] = weighted_samples.sum(
                axis=0
            )  # Sum over N directions
        fourier[l] = torch.from_numpy(coeffs_l.astype(np.complex64)).to(device)

    return fourier


def basic_isht(
    fourier: Dict[int, ArrayLike],
    directions: ArrayLike,
) -> np.ndarray:
    """
    Compute inverse spherical harmonic transform (iSHT).

    Args:
        fourier: Dictionary mapping degree l to (B, M, 2l+1, C) array of coefficients, where B is batch size, M is N dimension, and C is channels
        directions: (num_points, 3) array of unit vectors

    Returns:
        (num_points, B, M, C) array of reconstructed samples
    """
    # Convert directions to numpy
    if isinstance(directions, torch.Tensor):
        directions = directions.detach().cpu().numpy()
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    N, _ = directions.shape
    B, M, _, C = next(iter(fourier.values())).shape
    result = np.zeros((N, B, M, C), dtype=np.complex128)

    theta, phi = get_spherical_coords(directions)

    for l, coeffs in fourier.items():
        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.detach().cpu().numpy()
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)[
                :, None, None, None
            ]  # (N, 1, 1, 1), complex
            coeffs_reshaped = coeffs[:, :, m + l, :]  # (B, M, C)
            result += Y_lm * coeffs_reshaped[None, :, :, :]  # Broadcast over N

    return result.astype(np.complex64)


def fast_sht(samples, directions, L_max, device=None, weights=None):
    """
    Compute the spherical harmonic transform using ssht package.

    Args:
        samples: Complex tensor of shape [N, C] containing samples
        directions: Tensor of shape [N, 3] containing unit vectors
        L_max: Maximum harmonic degree
        device: Device to use for computation
        weights: Optional quadrature weights of shape [N]

    Returns:
        Dictionary mapping l to complex tensor of shape [C, 2l+1]
    """
    # Convert to numpy if needed
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    if torch.is_tensor(directions):
        directions = directions.detach().cpu().numpy()

    # Convert directions to spherical coordinates
    theta = np.arccos(directions[:, 2])  # [N]
    phi = np.arctan2(directions[:, 1], directions[:, 0])  # [N]

    # Default weights if not provided
    if weights is None:
        weights = np.sin(theta)  # [N]

    # Process each channel
    num_channels = samples.shape[1]
    fourier = {}

    for c in range(num_channels):
        # Apply weights
        weighted_samples = samples[:, c] * weights  # [N]

        # Compute SHT using ssht
        flm = ssht.forward(weighted_samples, L_max, Method="MW")  # [L_max+1, 2*L_max+1]

        # Store in dictionary format
        for l in range(L_max + 1):
            if l not in fourier:
                fourier[l] = np.zeros((num_channels, 2 * l + 1), dtype=np.complex128)
            fourier[l][c] = flm[l, L_max - l : L_max + l + 1]

    # Convert to torch tensors if needed
    if device is not None:
        fourier = {l: torch.from_numpy(f).to(device) for l, f in fourier.items()}
    else:
        fourier = {l: torch.from_numpy(f) for l, f in fourier.items()}

    return fourier


def s2fft_sht(
    samples: ArrayLike,
    directions: ArrayLike,
    L_max: int,
    device: torch.device = torch.device("cpu"),
    weights: Optional[ArrayLike] = None,
) -> Dict[int, torch.Tensor]:
    """
    Compute spherical harmonic transform (SHT) using the s2fft package.

    Args:
        samples: (N, C) array of samples at N directions
        directions: (N, 3) array of unit vectors
        L_max: maximum degree
        device: torch device for output tensors
        weights: (N,) array of quadrature weights (optional)

    Returns:
        Dict[int, torch.Tensor]: Fourier coefficients for each l
    """
    # Convert inputs to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(directions, torch.Tensor):
        directions = directions.detach().cpu().numpy()
    if weights is not None and isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    N, C = samples.shape
    print(
        f"[DEBUG] Input shapes - samples: {samples.shape}, directions: {directions.shape}"
    )

    theta, phi = get_spherical_coords(directions)

    # Default quadrature weights if none provided
    if weights is None:
        weights = np.sin(theta)
        weights *= (4 * np.pi) / weights.sum()

    # Initialize output dictionary
    fourier: Dict[int, torch.Tensor] = {}

    # Process each channel separately
    for c in range(C):
        # Get samples for this channel
        f = samples[:, c]

        # Apply weights
        f = f * weights

        # Reshape for s2fft (it expects a 2D array with theta and phi dimensions)
        ntheta = 2 * L_max + 1
        nphi = 2 * L_max + 1
        f_reshaped = np.zeros((ntheta, nphi), dtype=np.complex128)

        # Map samples to grid points
        theta_idx = np.round((theta / np.pi) * (ntheta - 1)).astype(int)
        phi_idx = np.round(((phi + np.pi) / (2 * np.pi)) * (nphi - 1)).astype(int)
        f_reshaped[theta_idx, phi_idx] = f

        print(f"[DEBUG] Channel {c} - f_reshaped shape: {f_reshaped.shape}")

        # Compute SHT using s2fft
        flm = s2fft.forward(f_reshaped, L_max)
        print(f"[DEBUG] Channel {c} - flm shape: {flm.shape}")

        # Store coefficients in dictionary format
        for l in range(L_max + 1):
            if l not in fourier:
                fourier[l] = np.zeros((2 * l + 1, C), dtype=np.complex128)
            # Extract coefficients for this l
            start_idx = l * l
            end_idx = (l + 1) * (l + 1)
            coeffs = flm[start_idx:end_idx]
            print(
                f"[DEBUG] Channel {c}, l={l} - coeffs shape: {coeffs.shape}, fourier[l] shape: {fourier[l].shape}"
            )
            # Store coefficients in our format (2l+1, C)
            fourier[l][:, c] = coeffs[: 2 * l + 1]

    # Convert to torch tensors
    for l in fourier:
        fourier[l] = torch.from_numpy(fourier[l].astype(np.complex64)).to(device)

    return fourier


def s2fft_isht(
    fourier: Dict[int, ArrayLike],
    directions: ArrayLike,
) -> np.ndarray:
    """
    Compute inverse spherical harmonic transform (iSHT) using the s2fft package.

    Args:
        fourier: Dictionary mapping degree l to (2l+1, C) array of coefficients
        directions: (N, 3) array of unit vectors

    Returns:
        (N, C) array of reconstructed samples
    """
    # Convert directions to numpy
    if isinstance(directions, torch.Tensor):
        directions = directions.detach().cpu().numpy()
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    N, _ = directions.shape
    _, C = next(iter(fourier.values())).shape
    L_max = max(fourier.keys())

    print(f"[DEBUG] ISHT Input shapes - directions: {directions.shape}, L_max: {L_max}")

    # Initialize output array
    result = np.zeros((N, C), dtype=np.complex128)

    # Process each channel separately
    for c in range(C):
        # Convert dictionary format to s2fft format
        flm = np.zeros((L_max + 1) ** 2, dtype=np.complex128)
        for l in range(L_max + 1):
            if l in fourier:
                coeffs = fourier[l][:, c]
                if isinstance(coeffs, torch.Tensor):
                    coeffs = coeffs.detach().cpu().numpy()
                print(f"[DEBUG] Channel {c}, l={l} - coeffs shape: {coeffs.shape}")
                # Store coefficients in s2fft format
                start_idx = l * l
                end_idx = (l + 1) * (l + 1)
                print(
                    f"[DEBUG] Channel {c}, l={l} - start_idx: {start_idx}, end_idx: {end_idx}"
                )
                # Store coefficients in s2fft format
                flm[start_idx : start_idx + 2 * l + 1] = coeffs

        print(f"[DEBUG] Channel {c} - flm shape before inverse: {flm.shape}")

        # Compute inverse transform using s2fft
        f = s2fft.inverse(flm, L_max)
        print(f"[DEBUG] Channel {c} - f shape after inverse: {f.shape}")

        # Map from grid points back to sample points
        theta, phi = get_spherical_coords(directions)
        ntheta, nphi = f.shape
        theta_idx = np.round((theta / np.pi) * (ntheta - 1)).astype(int)
        phi_idx = np.round(((phi + np.pi) / (2 * np.pi)) * (nphi - 1)).astype(int)
        result[:, c] = f[theta_idx, phi_idx]

    return result.astype(np.complex64)


def split_tensor_to_dict(tensor: torch.Tensor, L_max: int) -> Dict[int, torch.Tensor]:
    """
    Split a tensor of shape [b, N, (L+1)^2, m] into a dictionary of tensors with shape [b, N, 2l+1, m]
    for each degree l from 0 to L_max.

    Args:
        tensor: Input tensor of shape [b, N, (L+1)^2, m]
        L_max: Maximum harmonic degree

    Returns:
        Dictionary mapping degree l to tensor of shape [b, N, 2l+1, m]
    """
    b, N, _, m = tensor.shape
    result = {}

    for l in range(L_max + 1):
        start_idx = l * l
        end_idx = (l + 1) * (l + 1)
        # Extract coefficients for this l
        coeffs = tensor[:, :, start_idx:end_idx, :]
        # Reshape to [b, N, 2l+1, m]
        result[l] = coeffs.reshape(b, N, 2 * l + 1, m)

    return result


def dict_to_tensor(fourier_dict: Dict[int, torch.Tensor], L_max: int) -> torch.Tensor:
    """
    Convert a dictionary of tensors with shape [b, N, 2l+1, m] back to a single tensor of shape [b, N, (L+1)^2, m].

    Args:
        fourier_dict: Dictionary mapping degree l to tensor of shape [b, N, 2l+1, m]
        L_max: Maximum harmonic degree

    Returns:
        Tensor of shape [b, N, (L+1)^2, m]
    """
    # Get shape information from first entry
    b, N, _, m = next(iter(fourier_dict.values())).shape

    # Create output tensor
    total_coeffs = (L_max + 1) ** 2
    result = torch.zeros((b, N, total_coeffs, m))

    # Fill in the tensor
    for l in range(L_max + 1):
        start_idx = l * l
        end_idx = start_idx + 2 * l + 1
        result[:, :, start_idx:end_idx, :] = fourier_dict[l]

    return result
