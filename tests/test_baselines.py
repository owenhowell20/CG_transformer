import pytest
import sys
import os
import torch
from fixtures import mock_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.models import StandardAttention


def test_full_model(mock_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    num_tokens = 128
    input_dimension = 512
    output_dimension = input_dimension

    f = torch.randn(batch_size, num_tokens, input_dimension, device=device)

    model = StandardAttention(
        input_dimension=input_dimension,
        hidden_dimension=4 * input_dimension,
        output_dimension=input_dimension,
    ).to(device)

    f = model(f)

    assert f.shape[0] == f.shape[0], "batch dimensions should be the same"
    assert f.shape[1] == f.shape[1], "token dimensions should be the same"
    assert f.shape[2] == output_dimension, "Output dimension mistmatch"

    assert isinstance(f, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(f).any(), "Output contains NaN values"

    assert isinstance(f, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(f).any(), "Output contains NaN values"
