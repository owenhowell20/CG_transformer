import sys
import os
import torch
from fixtures import mock_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.models import HyenaAttention


### Hyena operator test
def test_Hyena():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_tokens, d = 32, 64, 215
    x = torch.randn(batch_size, num_tokens, d, device=device)

    d_out = 215

    hyena = HyenaAttention(
        input_dimension=d,
        hidden_dimension=128,
        output_dimension=d_out,
        num_heads=8,
        order=3,
        device=device,
        kernel_size=7,
    ).to(device)

    x = hyena(x)

    assert x.shape[0] == batch_size
    assert x.shape[1] == num_tokens
    assert x.shape[2] == d_out
