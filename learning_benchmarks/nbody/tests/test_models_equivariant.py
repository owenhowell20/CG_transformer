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

from nbody_dataloader import UnifiedDatasetWrapper
from nbody_flags import get_flags
from nbody_run import collate, RandomRotation
from test_data import mock_flags
from nbody_models import nbody_SE3Hyenea, nbody_Hyena, nbody_Standard, nbody_GATr
import copy


def test_SE3Hyena_equivariant(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    train_dataset = UnifiedDatasetWrapper(FLAGS, split="train")
    FLAGS.sequence_length = train_dataset.n_points  ### number of points in model

    model = nbody_SE3Hyenea(
        sequence_length=2 * FLAGS.sequence_length,
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
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
        rot = RandomRotation()
        R = rot.get_rotation_matrix().to(FLAGS.device)  # shape (3, 3)

        ### unrotated model inputs
        g = g.to(FLAGS.device)
        g_rot = copy.deepcopy(g)

        ### Rotate Node features in graph the (positions and velocities), g.x ~ [b*N,7]
        g_rot.x[:, 0:3] = torch.einsum("ij, nj -> ni", R, g.x[:, 0:3])
        g_rot.x[:, 3:6] = torch.einsum("ij, nj -> ni", R, g.x[:, 3:6])

        ### run model forward and compute loss
        rot_pred_xt, rot_pred_vt = model(g_rot)
        rot_pred_xt = rot_pred_xt.detach()
        rot_pred_vt = rot_pred_vt.detach()
        rot_pred = torch.cat([rot_pred_xt, rot_pred_vt], dim=0)

        ### now, rotate inputs and check if equivariant
        pred_xt, pred_vt = model(g)

        ##pred_xt ~[b,N,3], pred_vt ~[b,N,3], R~(3,3)
        pred_xt = torch.einsum("ij, bnj -> bni", R, pred_xt)
        pred_vt = torch.einsum("ij, bnj -> bni", R, pred_vt)
        pred_xt = pred_xt.detach()
        pred_vt = pred_vt.detach()
        pred = torch.cat([pred_xt, pred_vt], dim=0)

        assert pred.shape == rot_pred.shape, "Shape mismatch"
        assert torch.allclose(pred, rot_pred, atol=1e-3), "Error model not equivariant"
        break


def test_GATr_equivariant(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    train_dataset = UnifiedDatasetWrapper(FLAGS, split="train")
    FLAGS.sequence_length = train_dataset.n_points  ### number of points in model

    model = nbody_GATr(
        sequence_length=2 * FLAGS.sequence_length,
        positional_encoding_dimension=8,
        input_dimension_1=20,
        input_dimension_2=20,
        input_dimension_3=20,
        blocks=1,
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
        rot = RandomRotation()
        R = rot.get_rotation_matrix().to(FLAGS.device)  # shape (3, 3)

        ### unrotated model inputs
        g = g.to(FLAGS.device)
        g_rot = copy.deepcopy(g)

        ### Rotate Node features in graph the (positions and velocities), g.x ~ [b*N,7]
        g_rot.x[:, 0:3] = torch.einsum("ij, nj -> ni", R, g.x[:, 0:3])
        g_rot.x[:, 3:6] = torch.einsum("ij, nj -> ni", R, g.x[:, 3:6])

        ### run model forward and compute loss
        rot_pred_xt, rot_pred_vt = model(g_rot)
        rot_pred_xt = rot_pred_xt.detach()
        rot_pred_vt = rot_pred_vt.detach()
        rot_pred = torch.cat([rot_pred_xt, rot_pred_vt], dim=0)

        ### now, rotate inputs and check if equivariant
        pred_xt, pred_vt = model(g)

        ##pred_xt ~[b,N,3], pred_vt ~[b,N,3], R~(3,3)
        pred_xt = torch.einsum("ij, bnj -> bni", R, pred_xt)
        pred_vt = torch.einsum("ij, bnj -> bni", R, pred_vt)
        pred_xt = pred_xt.detach()
        pred_vt = pred_vt.detach()
        pred = torch.cat([pred_xt, pred_vt], dim=0)

        assert pred.shape == rot_pred.shape, "Shape mismatch"
        assert torch.allclose(pred, rot_pred, atol=1e-3), "Error model not equivariant"
        break
