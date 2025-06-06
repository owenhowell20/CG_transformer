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
from nbody_run import main, train_epoch, collate, run_test_epoch
from test_data import mock_flags
from nbody_models import nbody_SE3Hyenea, nbody_Hyena, nbody_Standard, nbody_GATr


def test_train_epoch(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    train_dataset = UnifiedDatasetWrapper(FLAGS, split="train")

    ### get number of points
    FLAGS.sequence_length = train_dataset.n_points

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    model = nbody_SE3Hyenea(
        sequence_length=2 * FLAGS.sequence_length,
        positional_encoding_dimension=FLAGS.positional_encoding_dimension,
        input_dimension_1=FLAGS.input_dimension_1,
        input_dimension_2=FLAGS.input_dimension_2,
        input_dimension_3=FLAGS.input_dimension_3,
    ).to(FLAGS.device)

    ### Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, FLAGS.num_epochs, eta_min=1e-4
    )

    wandb.init(project="test-debug", name=FLAGS.name, config=FLAGS)
    for epoch in range(1):
        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)


def test_test_epoch(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    test_dataset = UnifiedDatasetWrapper(FLAGS, split="test")

    ### get number of points
    FLAGS.sequence_length = test_dataset.n_points

    test_loader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    model = nbody_SE3Hyenea(
        sequence_length=2 * FLAGS.sequence_length,
        positional_encoding_dimension=FLAGS.positional_encoding_dimension,
        input_dimension_1=FLAGS.input_dimension_1,
        input_dimension_2=FLAGS.input_dimension_2,
        input_dimension_3=FLAGS.input_dimension_3,
    ).to(FLAGS.device)

    wandb.init(project="test-debug", name=FLAGS.name, config=FLAGS)
    for epoch in range(1):
        dT = 0.1
        run_test_epoch(epoch, model, task_loss, test_loader, FLAGS)


def test_main(mock_flags):
    FLAGS, UNPARSED_ARGV = mock_flags
    wandb.init(project="test-debug", name=FLAGS.name, config=FLAGS)
    main(FLAGS, UNPARSED_ARGV)
    assert True
