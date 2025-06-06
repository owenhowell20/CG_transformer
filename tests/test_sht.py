import torch
from src.sht import (
    check_format,
    basic_sht,
    basic_isht,
    check_equal,
    sht_tensor_product,
)
import torch.nn.functional as F
import healpy as hp
import numpy as np
import escnn
from src.utils import regular_rep_irreps
from fixtures import so3_group

from src.tensor_products import project_tensor_product


def test_FFT_matrix():
    max_harmonic_in = 3
    max_harmonic_out = 3

    J_out = 1

    batch_size = 3

    N = 3

    ### set mult to be 1
    m1 = 1
    m2 = m1

    # Generate complex random tensors
    u = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m1)
    v = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m2)

    ### need u and v to decay in max harmonic
    decay = torch.tensor([1 / ((i + 1) ** 4) for i in range(max_harmonic_in + 1)])
    decay = torch.repeat_interleave(
        decay, torch.tensor([2 * i + 1 for i in range(max_harmonic_in + 1)])
    )
    decay = decay.view(1, 1, -1, 1)  # Reshape for broadcasting
    u = u * decay
    v = v * decay

    ### comput the tensor prouct
    uv = sht_tensor_product(u, v, max_harmonic_in, max_harmonic_out)

    ### get the J_out part
    uv_Jout = uv[:, :, J_out**2 : (J_out + 1) ** 2, :]

    tensors = []
    for i in range(max_harmonic_in + 1):
        for j in range(max_harmonic_out):
            if np.abs(i - j) <= J_out <= i + j:
                # Get the coefficients for the current harmonic degrees
                q = u[:, :, i**2 : (i + 1) ** 2, :]
                k = v[:, :, j**2 : (j + 1) ** 2, :]

                output = project_tensor_product(q, k, J_out, type="inter")
                tensors.append(output)

    ### sum over all tensor products
    output = torch.stack(tensors, dim=0).sum(dim=0)

    assert output.shape == uv_Jout.shape, "Shape mismatch"
    assert torch.allclose(
        output, uv_Jout, atol=1e-2
    ), "Tensor product computation mismatch"


# def test_sht_tensor_product_equivariant(so3_group):
#     """Test that SHT tensor product is equivariant."""
#     max_harmonic_in = 2  # Use same harmonic degree for input and output
#     max_harmonic_out = 2

#     batch_size = 3
#     N = 3
#     m1 = 1
#     m2 = m1


#     u = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m1)
#     v = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m2)


#     uv = sht_tensor_product(u, v, max_harmonic_in, max_harmonic_out)

#     rep_a = regular_rep_irreps(
#         l_max=max_harmonic_in + 1, multiplicity=m1, so3_group=so3_group
#     )
#     rep_b = regular_rep_irreps(
#         l_max=max_harmonic_in + 1, multiplicity=m2, so3_group=so3_group
#     )
#     rep_out = regular_rep_irreps(
#         l_max=max_harmonic_out + 1, multiplicity=m1, so3_group=so3_group
#     )

#     type_a = escnn.nn.FieldType(so3_group, rep_a)
#     type_b = escnn.nn.FieldType(so3_group, rep_b)
#     type_out = escnn.nn.FieldType(so3_group, rep_out)

#     ### apply G transformation
#     g = so3_group.fibergroup.sample()

#     u = u.permute(0, 1, 3, 2).reshape(batch_size * N, ((max_harmonic_in + 1) ** 2) * m1)
#     v = v.permute(0, 1, 3, 2).reshape(batch_size * N, ((max_harmonic_in + 1) ** 2) * m2)
#     uv = uv.permute(0, 1, 3, 2).reshape(
#         batch_size * N, ((max_harmonic_out + 1) ** 2) * (m1)
#     )

#     # Wrap tensors
#     u = escnn.nn.GeometricTensor(u, type_a)
#     v = escnn.nn.GeometricTensor(v, type_b)
#     uv = escnn.nn.GeometricTensor(uv, type_out)

#     u_g = u.transform(g).tensor
#     v_g = v.transform(g).tensor
#     uv_g = uv.transform(g).tensor

#     ### now, convert back
#     u_g = u_g.reshape(batch_size, N, m1, (max_harmonic_in + 1) ** 2).permute(0, 1, 3, 2)
#     v_g = v_g.reshape(batch_size, N, m2, (max_harmonic_in + 1) ** 2).permute(0, 1, 3, 2)
#     uv_g = uv_g.reshape(batch_size, N, m1, (max_harmonic_out + 1) ** 2).permute(
#         0, 1, 3, 2
#     )

#     g_uv = sht_tensor_product(u_g, v_g, max_harmonic_in, max_harmonic_out)

#     assert uv_g.shape == g_uv.shape, "Shape mismatch"
#     assert torch.allclose(uv_g, g_uv, atol=1e-2)


# ### Try withouth the permute and reshape
# def test_sht_tensor_product_equivariant_II(so3_group):
#     """Test that SHT tensor product is equivariant."""
#     max_harmonic_in = 1
#     max_harmonic_out = 2

#     batch_size = 3
#     N = 3
#     m1 = 1
#     m2 = m1

#     u = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m1)
#     v = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m2)

#     uv = sht_tensor_product(u, v, max_harmonic_in, max_harmonic_out)

#     rep_a = regular_rep_irreps(
#         l_max=max_harmonic_in + 1, multiplicity=m1, so3_group=so3_group
#     )
#     rep_b = regular_rep_irreps(
#         l_max=max_harmonic_in + 1, multiplicity=m2, so3_group=so3_group
#     )
#     rep_out = regular_rep_irreps(
#         l_max=max_harmonic_out + 1, multiplicity=m1, so3_group=so3_group
#     )

#     type_a = escnn.nn.FieldType(so3_group, rep_a)
#     type_b = escnn.nn.FieldType(so3_group, rep_b)
#     type_out = escnn.nn.FieldType(so3_group, rep_out)

#     ### apply G transformation
#     g = so3_group.fibergroup.sample()

#     u = u.reshape(batch_size * N, ((max_harmonic_in + 1) ** 2) * m1)
#     v = v.reshape(batch_size * N, ((max_harmonic_in + 1) ** 2) * m2)
#     uv = uv.reshape(batch_size * N, ((max_harmonic_out + 1) ** 2) * (m1))

#     # Wrap tensors
#     u = escnn.nn.GeometricTensor(u, type_a)
#     v = escnn.nn.GeometricTensor(v, type_b)
#     uv = escnn.nn.GeometricTensor(uv, type_out)

#     u_g = u.transform(g).tensor
#     v_g = v.transform(g).tensor
#     uv_g = uv.transform(g).tensor

#     ### now, convert back
#     u_g = u_g.reshape(batch_size, N, m1, (max_harmonic_in + 1) ** 2)
#     v_g = v_g.reshape(batch_size, N, m2, (max_harmonic_in + 1) ** 2)
#     uv_g = uv_g.reshape(batch_size, N, m1, (max_harmonic_out + 1) ** 2)

#     g_uv = sht_tensor_product(u_g, v_g, max_harmonic_in, max_harmonic_out)

#     assert uv_g.shape == g_uv.shape, "Shape mismatch"
#     assert torch.allclose(uv_g, g_uv, atol=1e-2)


# def test_sht_healpix_grid():
#     """Test SHT/iSHT with HEALPix grid points."""
#     max_harmonic_in = 5
#     max_harmonic_out = 5

#     batch_size = 2
#     N = 3
#     m = 1

#     fourier = {}
#     for l in range(max_harmonic_in + 1):
#         # Generate complex coefficients: real and imaginary parts with batch and N dimensions
#         real = torch.randn(batch_size, N, 2 * l + 1, m )
#         imag = torch.randn(batch_size, N, 2 * l + 1, m )
#         fourier[l] = torch.complex(real, imag)

#     assert check_format(fourier)

#     nside = 16  # HEALPix resolution parameter (gives 3,072 points)
#     vecs = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))  # (3, N)
#     points = np.stack(vecs, axis=-1)  # (N, 3)
#     grid_points = points.shape[0]
#     # HEALPix weights: all equal, sum to 4*pi
#     weights = np.full(grid_points, 4 * np.pi / grid_points)

#     f = basic_isht(fourier=fourier, directions=points)
#     assert f.shape[0] == grid_points, "iSHT wrong dimension"
#     assert f.shape[1] == batch_size, "iSHT wrong batch dimension"
#     assert f.shape[2] == N, "iSHT wrong N dimension"
#     output_fourier = basic_sht(
#         samples=f, directions=points, L_max=max_harmonic_out, weights=weights
#     )

#     # Print differences for each l
#     print("\nHEALPix Grid - Differences between input and output coefficients:")
#     for l in range( min(max_harmonic_in, max_harmonic_out) + 1):
#         diff = torch.abs(fourier[l] - output_fourier[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Input:  {fourier[l].flatten()}")
#         print(f"Output: {output_fourier[l].flatten()}\n")

#     # For HEALPix grid, we expect much better accuracy
#     assert check_equal(fourier, output_fourier, eps=1e-2)


# def test_fast_sht():
#     """Test fast SHT implementation against basic SHT."""
#     max_harmonic = 3
#     grid_points = 1000  # Smaller grid for faster testing

#     # Generate random samples
#     samples = np.random.randn(grid_points, 2) + 1j * np.random.randn(grid_points, 2)
#     points = F.normalize(torch.randn(grid_points, 3), dim=-1)
#     points = points.numpy()

#     # Compute transforms using both methods
#     basic_fourier = basic_sht(samples, points, L_max=max_harmonic)
#     fast_fourier = fast_sht(samples, points, L_max=max_harmonic)

#     # Print differences for each l
#     print("\nFast SHT vs Basic SHT - Differences between coefficients:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(basic_fourier[l] - fast_fourier[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Basic: {basic_fourier[l].flatten()}")
#         print(f"Fast: {fast_fourier[l].flatten()}\n")

#     # Check that results are similar
#     assert check_equal(basic_fourier, fast_fourier, eps=1e-2), "Fast SHT results differ significantly from basic SHT"

#     # Test with HEALPix grid
#     nside = 8
#     vecs = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
#     points = np.stack(vecs, axis=-1)
#     weights = np.full(points.shape[0], 4 * np.pi / points.shape[0])

#     # Generate new samples for HEALPix grid
#     samples = np.random.randn(points.shape[0], 2) + 1j * np.random.randn(points.shape[0], 2)

#     # Compute transforms using both methods
#     basic_fourier = basic_sht(samples, points, L_max=max_harmonic, weights=weights)
#     fast_fourier = fast_sht(samples, points, L_max=max_harmonic, weights=weights)

#     # Print differences for each l
#     print("\nFast SHT vs Basic SHT (HEALPix) - Differences between coefficients:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(basic_fourier[l] - fast_fourier[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Basic: {basic_fourier[l].flatten()}")
#         print(f"Fast: {fast_fourier[l].flatten()}\n")

#     # Check that results are similar
#     assert check_equal(basic_fourier, fast_fourier, eps=1e-2), "Fast SHT results differ significantly from basic SHT on HEALPix grid"


# def test_s2fft_sht():
#     """Test s2fft SHT implementation against basic SHT."""
#     max_harmonic = 3
#     grid_points = 1000  # Smaller grid for faster testing

#     # Generate random samples
#     samples = np.random.randn(grid_points, 2) + 1j * np.random.randn(grid_points, 2)
#     points = F.normalize(torch.randn(grid_points, 3), dim=-1)
#     points = points.numpy()

#     # Compute transforms using both methods
#     basic_fourier = basic_sht(samples, points, L_max=max_harmonic)
#     s2fft_fourier = s2fft_sht(samples, points, L_max=max_harmonic)

#     # Print differences for each l
#     print("\ns2fft SHT vs Basic SHT - Differences between coefficients:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(basic_fourier[l] - s2fft_fourier[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Basic: {basic_fourier[l].flatten()}")
#         print(f"s2fft: {s2fft_fourier[l].flatten()}\n")

#     # Check that results are similar
#     assert check_equal(
#         basic_fourier, s2fft_fourier, eps=1e-1
#     ), "s2fft SHT results differ significantly from basic SHT"

#     # Test with HEALPix grid
#     nside = 8
#     vecs = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
#     points = np.stack(vecs, axis=-1)
#     weights = np.full(points.shape[0], 4 * np.pi / points.shape[0])

#     # Generate new samples for HEALPix grid
#     samples = np.random.randn(points.shape[0], 2) + 1j * np.random.randn(
#         points.shape[0], 2
#     )

#     # Compute transforms using both methods
#     basic_fourier = basic_sht(samples, points, L_max=max_harmonic, weights=weights)
#     s2fft_fourier = s2fft_sht(samples, points, L_max=max_harmonic, weights=weights)

#     # Print differences for each l
#     print("\ns2fft SHT vs Basic SHT (HEALPix) - Differences between coefficients:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(basic_fourier[l] - s2fft_fourier[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Basic: {basic_fourier[l].flatten()}")
#         print(f"s2fft: {s2fft_fourier[l].flatten()}\n")

#     # Check that results are similar
#     assert check_equal(
#         basic_fourier, s2fft_fourier, eps=1e-1
#     ), "s2fft SHT results differ significantly from basic SHT on HEALPix grid"


# def test_s2fft_isht():
#     """Test s2fft inverse SHT implementation against basic inverse SHT."""
#     max_harmonic = 3
#     grid_points = 1000  # Smaller grid for faster testing

#     # Generate random Fourier coefficients
#     fourier = {}
#     for l in range(max_harmonic + 1):
#         # Generate complex coefficients: real and imaginary parts
#         real = torch.randn(2 * l + 1, 2, dtype=torch.float32)
#         imag = torch.randn(2 * l + 1, 2, dtype=torch.float32)
#         fourier[l] = torch.complex(real, imag)

#     assert check_format(fourier)

#     # Generate random points
#     points = F.normalize(torch.randn(grid_points, 3), dim=-1)
#     points = points.numpy()

#     # Compute inverse transforms using both methods
#     basic_samples = basic_isht(fourier=fourier, directions=points)
#     s2fft_samples = s2fft_isht(fourier=fourier, directions=points)

#     # Print differences
#     print("\ns2fft iSHT vs Basic iSHT - Differences between samples:")
#     diff = np.abs(basic_samples - s2fft_samples)
#     max_diff = np.max(diff)
#     mean_diff = np.mean(diff)
#     print(f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#     print(f"Basic: {basic_samples.flatten()[:10]}")  # Print first 10 samples
#     print(f"s2fft: {s2fft_samples.flatten()[:10]}\n")

#     # Check that results are similar
#     assert np.allclose(
#         basic_samples, s2fft_samples, atol=1e-2, rtol=1e-2
#     ), "s2fft iSHT results differ significantly from basic iSHT"

#     # Test with HEALPix grid
#     nside = 8
#     vecs = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
#     points = np.stack(vecs, axis=-1)

#     # Compute inverse transforms using both methods
#     basic_samples = basic_isht(fourier=fourier, directions=points)
#     s2fft_samples = s2fft_isht(fourier=fourier, directions=points)

#     # Print differences
#     print("\ns2fft iSHT vs Basic iSHT (HEALPix) - Differences between samples:")
#     diff = np.abs(basic_samples - s2fft_samples)
#     max_diff = np.max(diff)
#     mean_diff = np.mean(diff)
#     print(f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#     print(f"Basic: {basic_samples.flatten()[:10]}")  # Print first 10 samples
#     print(f"s2fft: {s2fft_samples.flatten()[:10]}\n")

#     # Check that results are similar
#     assert np.allclose(
#         basic_samples, s2fft_samples, atol=1e-2, rtol=1e-2
#     ), "s2fft iSHT results differ significantly from basic iSHT on HEALPix grid"


# def test_roundtrip_accuracy():
#     """Test round-trip accuracy of SHT/iSHT for both basic and s2fft implementations."""
#     max_harmonic = 3
#     grid_points = 1000  # Smaller grid for faster testing

#     # Generate random Fourier coefficients
#     fourier = {}
#     for l in range(max_harmonic + 1):
#         # Generate complex coefficients: real and imaginary parts
#         real = torch.randn(2 * l + 1, 2, dtype=torch.float32)
#         imag = torch.randn(2 * l + 1, 2, dtype=torch.float32)
#         fourier[l] = torch.complex(real, imag)

#     assert check_format(fourier)

#     # Test with random grid
#     points = F.normalize(torch.randn(grid_points, 3), dim=-1)
#     points = points.numpy()

#     # Basic implementation round-trip
#     basic_samples = basic_isht(fourier=fourier, directions=points)
#     basic_fourier_roundtrip = basic_sht(
#         samples=basic_samples, directions=points, L_max=max_harmonic
#     )

#     # s2fft implementation round-trip
#     s2fft_samples = s2fft_isht(fourier=fourier, directions=points)
#     s2fft_fourier_roundtrip = s2fft_sht(
#         samples=s2fft_samples, directions=points, L_max=max_harmonic
#     )

#     # Print differences for each l
#     print(
#         "\nRound-trip accuracy (Random Grid) - Differences between input and output coefficients:"
#     )
#     print("\nBasic implementation:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(fourier[l] - basic_fourier_roundtrip[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Input:  {fourier[l].flatten()}")
#         print(f"Output: {basic_fourier_roundtrip[l].flatten()}\n")

#     print("\ns2fft implementation:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(fourier[l] - s2fft_fourier_roundtrip[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Input:  {fourier[l].flatten()}")
#         print(f"Output: {s2fft_fourier_roundtrip[l].flatten()}\n")

#     # Check that results are similar
#     assert check_equal(
#         fourier, basic_fourier_roundtrip, eps=1e-2
#     ), "Basic round-trip results differ significantly from input"
#     assert check_equal(
#         fourier, s2fft_fourier_roundtrip, eps=1e-2
#     ), "s2fft round-trip results differ significantly from input"

#     # Test with HEALPix grid
#     nside = 8
#     vecs = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
#     points = np.stack(vecs, axis=-1)
#     weights = np.full(points.shape[0], 4 * np.pi / points.shape[0])

#     # Basic implementation round-trip with HEALPix
#     basic_samples = basic_isht(fourier=fourier, directions=points)
#     basic_fourier_roundtrip = basic_sht(
#         samples=basic_samples, directions=points, L_max=max_harmonic, weights=weights
#     )

#     # s2fft implementation round-trip with HEALPix
#     s2fft_samples = s2fft_isht(fourier=fourier, directions=points)
#     s2fft_fourier_roundtrip = s2fft_sht(
#         samples=s2fft_samples, directions=points, L_max=max_harmonic, weights=weights
#     )

#     # Print differences for each l
#     print(
#         "\nRound-trip accuracy (HEALPix Grid) - Differences between input and output coefficients:"
#     )
#     print("\nBasic implementation:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(fourier[l] - basic_fourier_roundtrip[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Input:  {fourier[l].flatten()}")
#         print(f"Output: {basic_fourier_roundtrip[l].flatten()}\n")

#     print("\ns2fft implementation:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(fourier[l] - s2fft_fourier_roundtrip[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Input:  {fourier[l].flatten()}")
#         print(f"Output: {s2fft_fourier_roundtrip[l].flatten()}\n")

#     # Check that results are similar
#     assert check_equal(
#         fourier, basic_fourier_roundtrip, eps=1e-2
#     ), "Basic round-trip results differ significantly from input on HEALPix grid"
#     assert check_equal(
#         fourier, s2fft_fourier_roundtrip, eps=1e-2
#     ), "s2fft round-trip results differ significantly from input on HEALPix grid"


# def test_sht_tensor_product():
#     """Test SHT/iSHT with random points on the sphere."""
#     max_harmonic_in = 3
#     max_harmonic_out = 5

#     batch_size = 3
#     N = 3


#     m1 = 1
#     m2 = m1

#     u = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m1 )
#     v = torch.randn(batch_size, N, (max_harmonic_in + 1) ** 2, m2 )

#     uv = sht_tensor_product(u, v, L_max_in=max_harmonic_in, L_max_out=max_harmonic_out)

#     assert uv.shape == (batch_size, N, (max_harmonic_out + 1) ** 2, m1*m2)


# def test_sht_random_grid():
#     """Test SHT/iSHT with random points on the sphere."""
#     max_harmonic = 5

#     batch_size = 1
#     N = 1

#     fourier = {}
#     for l in range(max_harmonic + 1):
#         # Generate complex coefficients: real and imaginary parts
#         real = torch.randn(batch_size, N, 2 * l + 1, 1, dtype=torch.float32)
#         imag = torch.randn(batch_size, N, 2 * l + 1, 1, dtype=torch.float32)
#         fourier[l] = torch.complex(real, imag)

#     assert check_format(fourier)

#     grid_points = 10000
#     points = F.normalize(torch.randn(grid_points, 3), dim=-1)
#     points = points.numpy()  # Convert to numpy for consistency

#     f = basic_isht(fourier=fourier, directions=points)
#     assert f.shape[0] == grid_points, "iSHT wrong dimension"
#     output_fourier = basic_sht(samples=f, directions=points, L_max=max_harmonic)

#     # Print differences for each l
#     print("\nRandom Grid - Differences between input and output coefficients:")
#     for l in range(max_harmonic + 1):
#         diff = torch.abs(fourier[l] - output_fourier[l])
#         max_diff = torch.max(diff)
#         mean_diff = torch.mean(diff)
#         print(f"l={l}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
#         print(f"Input:  {fourier[l].flatten()}")
#         print(f"Output: {output_fourier[l].flatten()}\n")

#     # For random grid, we expect larger errors
#     assert check_equal(fourier, output_fourier, eps=8e-1)
