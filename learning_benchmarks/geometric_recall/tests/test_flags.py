import pytest
import sys
from unittest.mock import patch
import torch
import pytest
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


from learning_benchmarks.geometric_recall.flags import get_flags


@pytest.fixture
def mock_flags():
    # Simulate passing arguments to argparse as if they were passed on the command line
    # You can change these arguments to match the ones you want to test with
    test_args = [
        "script_name",  # This is typically the script name (ignored by argparse)
        "--model",
        "Standard",
        "--num_layers",
        "4",
        "--num_degrees",
        "4",
        "--num_channels",
        "4",
        "--div",
        "1",
        "--head",
        "1",
        "--batch_size",
        "64",
        "--lr",
        "1e-3",
        "--num_epochs",
        "500",
        "--ri_data_type",
        "charged",
        "--ri_data",
        "data_generation",
        "--data_str",
        "my_datasetfile",
        "--ri_delta_t",
        "10",
        "--ri_burn_in",
        "0",
        "--ri_start_at",
        "all",
        "--log_interval",
        "25",
        "--print_interval",
        "250",
        "--save_dir",
        "models",
        "--restore",
        "None",
        "--verbose",
        "0",
        "--num_workers",
        "4",
        "--profile",
        "False",
        "--seed",
        "1992",
    ]

    # Use monkeypatch to set sys.argv
    with patch.object(sys, "argv", test_args):
        # Call your function to parse arguments
        FLAGS, UNPARSED_ARGV = get_flags()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        FLAGS.device == device
        # Return the FLAGS object for use in tests
        yield FLAGS


def test_flags(mock_flags):
    FLAGS = mock_flags
    assert FLAGS.num_workers == 4, "mock flags not setup correctly"
