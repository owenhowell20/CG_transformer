import torch
import pytest
from src.kernels.standard import project_tensor_product

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

# Only import triton if CUDA is available
if CUDA_AVAILABLE:
    try:
        import triton
        from src.kernels.triton_project import project_tensor_product_triton

        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = False

# Skip decorator for tests that require CUDA and Triton
skip_if_no_triton = pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE), reason="Test requires CUDA and Triton"
)


def get_device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if CUDA_AVAILABLE else "cpu")


@skip_if_no_triton
def test_project_tensor_product_intra():
    """Test intra-channel tensor product projection."""
    # Test parameters
    batch_size = 2
    num_points = 4
    l1, l2, l_out = 1, 1, 2  # Test with l=1 inputs and l=2 output
    m = 3  # Number of channels

    # Create input tensors
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    d_out = 2 * l_out + 1

    device = get_device()
    u = torch.randn(batch_size, num_points, d1, m, device=device)
    v = torch.randn(batch_size, num_points, d2, m, device=device)

    # Compute using both implementations
    y_original = project_tensor_product(u, v, l_out, type="intra")
    y_triton = project_tensor_product_triton(u, v, l_out, type="intra")

    # Check shapes
    assert y_original.shape == y_triton.shape
    assert y_original.shape == (batch_size, num_points, d_out, m)

    # Check values
    torch.testing.assert_close(y_original, y_triton, rtol=1e-4, atol=1e-4)


@skip_if_no_triton
def test_project_tensor_product_inter():
    """Test inter-channel tensor product projection."""
    # Test parameters
    batch_size = 2
    num_points = 4
    l1, l2, l_out = 1, 1, 2  # Test with l=1 inputs and l=2 output
    m1, m2 = 3, 4  # Different number of channels

    # Create input tensors
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    d_out = 2 * l_out + 1

    device = get_device()
    u = torch.randn(batch_size, num_points, d1, m1, device=device)
    v = torch.randn(batch_size, num_points, d2, m2, device=device)

    # Compute using both implementations
    y_original = project_tensor_product(u, v, l_out, type="inter")
    y_triton = project_tensor_product_triton(u, v, l_out, type="inter")

    # Check shapes
    assert y_original.shape == y_triton.shape
    assert y_original.shape == (batch_size, num_points, d_out, m1 * m2)

    # Check values
    torch.testing.assert_close(y_original, y_triton, rtol=1e-4, atol=1e-4)


@skip_if_no_triton
def test_project_tensor_product_edge_cases():
    """Test edge cases for tensor product projection."""
    # Test with l=0 inputs
    batch_size = 2
    num_points = 4
    l1, l2, l_out = 0, 0, 0
    m = 3

    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    d_out = 2 * l_out + 1

    device = get_device()
    u = torch.randn(batch_size, num_points, d1, m, device=device)
    v = torch.randn(batch_size, num_points, d2, m, device=device)

    y_original = project_tensor_product(u, v, l_out, type="intra")
    y_triton = project_tensor_product_triton(u, v, l_out, type="intra")

    torch.testing.assert_close(y_original, y_triton, rtol=1e-4, atol=1e-4)

    # Test with larger l values
    l1, l2, l_out = 2, 2, 4
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    d_out = 2 * l_out + 1

    u = torch.randn(batch_size, num_points, d1, m, device=device)
    v = torch.randn(batch_size, num_points, d2, m, device=device)

    y_original = project_tensor_product(u, v, l_out, type="intra")
    y_triton = project_tensor_product_triton(u, v, l_out, type="intra")

    torch.testing.assert_close(y_original, y_triton, rtol=1e-4, atol=1e-4)


@skip_if_no_triton
def test_project_tensor_product_error_cases():
    """Test error cases for tensor product projection."""
    batch_size = 2
    num_points = 4
    l1, l2, l_out = 1, 1, 2
    m1, m2 = 3, 4

    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1

    device = get_device()
    u = torch.randn(batch_size, num_points, d1, m1, device=device)
    v = torch.randn(batch_size, num_points, d2, m2, device=device)

    # Test invalid type
    with pytest.raises(ValueError):
        project_tensor_product_triton(u, v, l_out, type="invalid")

    # Test mismatched channels in intra mode
    with pytest.raises(AssertionError):
        project_tensor_product_triton(u, v, l_out, type="intra")

    # Test invalid l_out value
    with pytest.raises(ValueError):
        project_tensor_product_triton(
            u, v, l_out=5, type="inter"
        )  # l_out too large for l1=l2=1


def test_standard_implementation_cpu():
    """Test the standard implementation on CPU."""
    # Test parameters
    batch_size = 2
    num_points = 4
    l1, l2, l_out = 1, 1, 2
    m = 3

    # Create input tensors
    d1 = 2 * l1 + 1
    d2 = 2 * l2 + 1
    d_out = 2 * l_out + 1

    u = torch.randn(batch_size, num_points, d1, m, device="cpu")
    v = torch.randn(batch_size, num_points, d2, m, device="cpu")

    # Compute using standard implementation
    y = project_tensor_product(u, v, l_out, type="intra")

    # Check shape
    assert y.shape == (batch_size, num_points, d_out, m)

    # Check that output is not all zeros
    assert not torch.allclose(y, torch.zeros_like(y))


if __name__ == "__main__":
    pytest.main([__file__])
