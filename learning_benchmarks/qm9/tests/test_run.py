from fixture import mock_qm9_batch, mock_qm9_dataloader
from qm9_models import qm9_SE3Hyenea
from qm9_flags import get_flags
from torch import nn, optim
import os
import torch
import numpy as np
from qm9_run import train_epoch, val_epoch, run_test_epoch, main
from qm9_loss import mock_task_loss
import wandb


def test_train(mock_qm9_dataloader):
    FLAGS, UNPARSED_ARGV = get_flags()

    model = qm9_SE3Hyenea(
        positional_encoding_dimension=FLAGS.positional_encoding_dimension,
        input_dimension_1=FLAGS.input_dimension_1,
        input_dimension_2=FLAGS.input_dimension_2,
        input_dimension_3=FLAGS.input_dimension_3,
        output_dimension=FLAGS.output_dimension,
        node_feature_dimension=11,
    ).to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, FLAGS.num_epochs, eta_min=1e-4
    )

    wandb.init(project="SE3-Hyena", name=FLAGS.name, config=vars(FLAGS))
    epoch = 1
    train_epoch(
        epoch, model, mock_task_loss, mock_qm9_dataloader, optimizer, scheduler, FLAGS
    )


def test_val_epoch(mock_qm9_dataloader):
    FLAGS, UNPARSED_ARGV = get_flags()

    model = qm9_SE3Hyenea(
        positional_encoding_dimension=FLAGS.positional_encoding_dimension,
        input_dimension_1=FLAGS.input_dimension_1,
        input_dimension_2=FLAGS.input_dimension_2,
        input_dimension_3=FLAGS.input_dimension_3,
        output_dimension=FLAGS.output_dimension,
        node_feature_dimension=11,
    ).to(FLAGS.device)

    epoch = 1
    val_epoch(epoch, model, mock_task_loss, mock_qm9_dataloader, FLAGS)


def test_test_epoch(mock_qm9_dataloader):
    FLAGS, UNPARSED_ARGV = get_flags()

    model = qm9_SE3Hyenea(
        positional_encoding_dimension=FLAGS.positional_encoding_dimension,
        input_dimension_1=FLAGS.input_dimension_1,
        input_dimension_2=FLAGS.input_dimension_2,
        input_dimension_3=FLAGS.input_dimension_3,
        output_dimension=FLAGS.output_dimension,
        node_feature_dimension=11,
    ).to(FLAGS.device)

    epoch = 1
    run_test_epoch(epoch, model, mock_task_loss, mock_qm9_dataloader, FLAGS)
