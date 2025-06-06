# import sys
# import os
# import pytest
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# import pickle
#
# try:
#     import dgl
#
#     DGL_AVAILABLE = True
# except ImportError:
#     DGL_AVAILABLE = False
#
#
# # Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
#
# # Add the parent directory to sys.path
# sys.path.insert(0, parent_dir)
#
# from nbody_dataloader import RIDataset
# from nbody_flags import get_flags
# from nbody_run import se3_collate, se3_train_epoch, se3_eval_epoch
# from nbody_models import SE3Transformer
# import torch.nn as nn
# from torch.optim import Adam
#
#
# @pytest.fixture
# def mock_argv():
#     """Fixture to mock the command-line arguments for get_flags"""
#     mock_argv = [
#         "--model",
#         "SE3Transformer",
#         "--num_layers",
#         "2",
#         "--num_degrees",
#         "4",
#         "--num_channels",
#         "2",
#         "--head",
#         "1",
#         "--div",
#         "1",
#         "--batch_size",
#         "32",
#         "--lr",
#         "1e-4",
#         "--num_epochs",
#         "100",
#         "--ri_data_type",
#         "charged",
#         "--ri_data",
#         "path/to/data",
#         "--data_str",
#         "test_data",
#         "--ri_delta_t",
#         "5",
#         "--ri_burn_in",
#         "5",
#         "--ri_start_at",
#         "zero",
#         "--seed",
#         "42",
#     ]
#     return mock_argv
#
#
# @pytest.fixture
# def mock_flags(mock_argv):
#     """Fixture to provide mock flags for tests"""
#     sys.argv = mock_argv
#     FLAGS, _ = get_flags()
#
#     FLAGS.data_str = "20_new"
#     FLAGS.ri_data = "nbody_data_generation"
#
#     return FLAGS
#
#
# @pytest.fixture
# def dataset(mock_flags):
#     """Fixture for RIDataset_PyG"""
#     return RIDataset(mock_flags, split="train")
#
#
# def test_ridataset_getitem(dataset):
#     """Test retrieving an item from the RIDataset_PyG"""
#     graph, x_T, v_T = dataset[0]
#     assert isinstance(graph, dgl.DGLGraph)  # Should return a DGL graph
#
#     assert x_T.shape == (
#         20,
#         3,
#     )  # Should match the point size (20 points, 3 coordinates)
#     assert v_T.shape == (20, 3)  # Velocity change should have the same shape as x_T
#
#
# def test_ridataset_connect_fully(dataset):
#     """Test the connect_fully method"""
#     num_atoms = 5
#     src, dst = dataset.connect_fully(num_atoms)
#
#     assert src.shape == (num_atoms * (num_atoms - 1),)
#     assert dst.shape == (num_atoms * (num_atoms - 1),)
#
#     for i, j in zip(src, dst):
#         assert i != j
#
#
# def test_se3_transformer(mock_flags, dataset):
#     """Test the SE3Transformer with mocked flags and dataset"""
#
#     # Set flags for SE3Transformer model
#     FLAGS = mock_flags
#     train_dataset = dataset
#
#     # Set device
#     if torch.cuda.is_available():
#         FLAGS.device = torch.device("cuda:0")
#     else:
#         FLAGS.device = torch.device("cpu")
#
#     # Set the number of points (sequence length)
#     FLAGS.sequence_length = train_dataset.n_points
#
#     # Initialize the SE3Transformer model
#     model = SE3Transformer(
#         num_layers=FLAGS.num_layers,
#         num_channels=FLAGS.num_channels,
#         num_degrees=6,
#         div=FLAGS.div,
#         n_heads=FLAGS.head,
#     ).to(FLAGS.device)
#
#     # Create DataLoader for the training dataset
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=FLAGS.batch_size,
#         shuffle=True,
#         collate_fn=se3_collate,
#         num_workers=FLAGS.num_workers,
#         drop_last=True,
#     )
#
#     # Test the model with the training data
#     for i, (g, y1, y2) in enumerate(train_loader):
#         g = g.to(FLAGS.device)
#         y1 = y1.to(FLAGS.device)
#         y2 = y2.to(FLAGS.device)
#
#         # Reshape y1 and y2
#         x_T = y1.view(-1, 3)
#         v_T = y2.view(-1, 3)
#         y = torch.stack([x_T, v_T], dim=1)
#
#         # Forward pass through the model
#         pred = model(g)
#         pred = pred.view(FLAGS.batch_size * FLAGS.sequence_length, 2, 3)
#
#         # Assert the prediction shape matches the ground truth
#         assert (
#             y.shape == pred.shape
#         ), f"Prediction shape mismatch: {y.shape} vs {pred.shape}"
#         assert False
#

#
# def test_train_epoch(mock_flags, dataset):
#     """Test a single train epoch without crashing"""
#     FLAGS = mock_flags
#     FLAGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     FLAGS.num_workers = 0
#     FLAGS.print_interval = 9999  # Suppress printing
#     FLAGS.log_interval = 9999  # Suppress wandb logging
#     FLAGS.profile = False
#
#     model = SE3Transformer(
#         num_layers=FLAGS.num_layers,
#         num_channels=FLAGS.num_channels,
#         num_degrees=FLAGS.num_degrees,
#         div=FLAGS.div,
#         n_heads=FLAGS.head,
#     ).to(FLAGS.device)
#
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
#     optimizer = Adam(model.parameters(), lr=1e-4)
#     scheduler = torch.optim.lr_scheduler.ConstantLR(
#         optimizer, factor=1.0, total_iters=1
#     )
#     criterion = nn.MSELoss()
#
#     # Monkeypatch wandb.log to avoid logging during tests
#     import wandb
#
#     wandb.log = lambda *args, **kwargs: None
#
#     train_epoch(0, model, criterion, dataloader, optimizer, scheduler, FLAGS)
#
#
# def test_eval_epoch(mock_flags, dataset):
#     """Test a single eval epoch without crashing"""
#     FLAGS = mock_flags
#     FLAGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     FLAGS.num_workers = 0
#
#     model = SE3Transformer(
#         num_layers=FLAGS.num_layers,
#         num_channels=FLAGS.num_channels,
#         num_degrees=FLAGS.num_degrees,
#         div=FLAGS.div,
#         n_heads=FLAGS.head,
#     ).to(FLAGS.device)
#
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)
#     criterion = nn.MSELoss()
#
#     # Monkeypatch wandb.log to avoid logging during tests
#     import wandb
#
#     wandb.log = lambda *args, **kwargs: None
#
#     dT = 1.0  # dummy value
#     eval_epoch(0, model, criterion, dataloader, FLAGS, dT)
