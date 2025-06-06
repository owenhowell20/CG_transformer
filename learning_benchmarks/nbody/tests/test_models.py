import sys
import os
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch
import wandb


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from nbody_dataloader import RIDataset_PyG
from nbody_flags import get_flags
from nbody_run import collate
from test_data import mock_flags
from nbody_models import nbody_SE3Hyenea, nbody_Hyena, nbody_Standard, nbody_GATr


# ### check that data is loaded correctly
# def test_standard_model(mock_flags):
#     FLAGS, UNPARSED_ARGV = mock_flags
#     train_dataset = RIDataset_PyG(FLAGS, split="train")
#
#     ### get number of points
#     FLAGS.sequence_length = train_dataset.n_points
#
#     model = nbody_Standard(
#         sequence_length=2 * FLAGS.sequence_length,
#         positional_encoding_dimension=FLAGS.positional_encoding_dimension,
#         input_dimension_1=16,
#         input_dimension_2=16,
#         input_dimension_3=16,
#     ).to(FLAGS.device)
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=FLAGS.batch_size,
#         shuffle=True,
#         collate_fn=collate,
#         num_workers=FLAGS.num_workers,
#         drop_last=True,
#     )
#
#     for i, (g, y1, y2) in enumerate(train_loader):
#         g = g.to(FLAGS.device)
#         y1 = y1.to(FLAGS.device)
#         y2 = y2.to(FLAGS.device)
#
#         x_T = y1.view(-1, 3)
#         v_T = y2.view(-1, 3)
#         y = torch.stack([x_T, v_T], dim=1)
#
#         pred_xt, pred_vt = model(g)
#         pred = torch.cat([pred_xt, pred_vt], dim=0)
#         pred = pred.view(FLAGS.batch_size * FLAGS.sequence_length, 2, 3)
#
#
#
#         assert y.shape == pred.shape, "model prediction not right shape"
#
# #
#
# ### check that data is loaded correctly
# def test_baseline_model(mock_flags):
#     FLAGS, UNPARSED_ARGV = mock_flags
#     train_dataset = RIDataset_PyG(FLAGS, split="train")
#
#     ### get number of points
#     FLAGS.sequence_length = train_dataset.n_points
#
#     model = nbody_Hyena(
#         sequence_length=2 * FLAGS.sequence_length,
#         positional_encoding_dimension=FLAGS.positional_encoding_dimension,
#         input_dimension_1=10,
#         input_dimension_2=10,
#         input_dimension_3=10,
#     ).to(FLAGS.device)
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=FLAGS.batch_size,
#         shuffle=True,
#         collate_fn=collate,
#         num_workers=FLAGS.num_workers,
#         drop_last=True,
#     )
#
#     for i, (g, y1, y2) in enumerate(train_loader):
#         g = g.to(FLAGS.device)
#         y1 = y1.to(FLAGS.device)
#         y2 = y2.to(FLAGS.device)
#
#         x_T = y1.view(-1, 3)
#         v_T = y2.view(-1, 3)
#         y = torch.stack([x_T, v_T], dim=1)
#
#         pred_xt, pred_vt = model(g)
#         pred = torch.cat([pred_xt, pred_vt], dim=0)
#         pred = pred.view(FLAGS.batch_size * FLAGS.sequence_length, 2, 3)
#
#         assert y.shape == pred.shape, "model prediction not right shape"


### check that data is loaded correctly
def test_SE3Hyena_model(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    train_dataset = RIDataset_PyG(FLAGS, split="train")

    ### get number of points
    FLAGS.sequence_length = train_dataset.n_points

    model = nbody_SE3Hyenea(
        sequence_length=2 * FLAGS.sequence_length,
        positional_encoding_dimension=FLAGS.positional_encoding_dimension,
        input_dimension_1=10,
        input_dimension_2=10,
        input_dimension_3=10,
    ).to(FLAGS.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    for i, (g, y1, y2) in enumerate(train_loader):
        g = g.to(FLAGS.device)
        y1 = y1.to(FLAGS.device)
        y2 = y2.to(FLAGS.device)

        x_T = y1.view(-1, 3)
        v_T = y2.view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1)

        pred_xt, pred_vt = model(g)
        pred = torch.cat([pred_xt, pred_vt], dim=0)
        pred = pred.view(FLAGS.batch_size * FLAGS.sequence_length, 2, 3)

        assert y.shape == pred.shape, "model prediction not right shape"


### check that data is loaded correctly
def test_GATr_model(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    train_dataset = RIDataset_PyG(FLAGS, split="train")

    ### get number of points
    FLAGS.sequence_length = train_dataset.n_points

    model = nbody_GATr(
        sequence_length=2 * FLAGS.sequence_length,
        positional_encoding_dimension=16,
        input_dimension_1=256,
        input_dimension_2=128,
        input_dimension_3=64,
        blocks=3,
    ).to(FLAGS.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    for i, (g, y1, y2) in enumerate(train_loader):
        g = g.to(FLAGS.device)
        y1 = y1.to(FLAGS.device)
        y2 = y2.to(FLAGS.device)

        x_T = y1.view(-1, 3)
        v_T = y2.view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1)

        pred_xt, pred_vt = model(g)
        pred = torch.cat([pred_xt, pred_vt], dim=0)
        pred = pred.view(FLAGS.batch_size * FLAGS.sequence_length, 2, 3)

        assert y.shape == pred.shape, "model prediction not right shape"
