import pytest
import torch
from fixtures import mock_data, so3_group
from escnn.group import SO3
from escnn.gspaces import no_base_space
import escnn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from src.models import StandardAttention, SE3HyenaOperator


def test_Hyena_Operator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 4
    num_tokens = 128
    input_dimension = 512
    output_dimension = input_dimension

    f = torch.randn(batch_size, num_tokens, input_dimension, device=device)

    attention_types = ["Standard"]
    for attention_type in attention_types:
        model = StandardAttention(
            input_dimension=input_dimension,
            hidden_dimension=input_dimension,
            output_dimension=output_dimension,
        ).to(f.device)

        f_out = model(f)

        ### need  f~(b,N,d_out)
        assert f_out.shape[0] == batch_size, "Output dimension mistmatch"
        assert f_out.shape[1] == num_tokens, "Output dimension mistmatch"
        assert f_out.shape[2] == output_dimension, "Output dimension mistmatch"
        assert isinstance(f_out, torch.Tensor), "Output is not a tensor"
        assert not torch.isnan(f_out).any(), "Output contains NaN values"
