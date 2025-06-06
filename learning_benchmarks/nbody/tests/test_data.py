import sys
import os
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import wandb


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from nbody_dataloader import UnifiedDatasetWrapper
from nbody_flags import get_flags

import pytest
import tempfile
import pickle
import os
import numpy as np
import types


@pytest.fixture
def mock_flags():
    FLAGS, UNPARSED_ARGV = get_flags()
    FLAGS.data_str = "5_new"  ### use small data for testing
    FLAGS.ri_data = "/home/zxp/projects/tmp/SE3-HyperHyena/"
    FLAGS.save_dir = "."
    FLAGS.num_epochs = 1
    FLAGS.num_channels = 5
    return FLAGS, UNPARSED_ARGV


### check that data is loaded correctly
def test_dataset(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    train_dataset = UnifiedDatasetWrapper(FLAGS, split="train")
    test_dataset = UnifiedDatasetWrapper(FLAGS, split="test")
    assert True
