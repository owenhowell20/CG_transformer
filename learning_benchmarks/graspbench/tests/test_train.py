import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import wandb
import escnn

from escnn.group import SO3
from escnn.gspaces import no_base_space

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from models import (
    SE3HyenaNormal,
    StandardNormal,
    NormalDGCNN,
    SE3HyperHyenaNormal,
    StandardGrasp,
    SE3HyenaGrasp,
    GraspDGCNN,
)

from train import train_epoch, run_test_epoch
from flags import get_flags
from dataset import GraspingDataset


def test_flags():
    FLAGS, UNPARSED_ARGV = get_flags()
    assert True


def test_train_epoch():
    FLAGS, UNPARSED_ARGV = get_flags()
    FLAGS.task = "Normals"

    if FLAGS.task == "Normals":
        if FLAGS.model == "SE3Hyena":
            model = SE3HyenaNormal(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=8,
                input_dimension_1=8,
                input_dimension_2=8,
                input_dimension_3=8,
            ).to(FLAGS.device)

        elif FLAGS.model == "Standard":
            model = StandardNormal(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=8,
                input_dimension_1=8,
                input_dimension_2=8,
                input_dimension_3=8,
            ).to(FLAGS.device)
        elif FLAGS.model == "dgcnn":
            model = NormalDGCNN(k=20, emb_dims=1024, dropout=0.5).to(FLAGS.device)
        else:
            raise ValueError(f"Unknown model type: {FLAGS.model}")

    elif FLAGS.task == "Grasp":
        if FLAGS.model == "SE3Hyena":
            model = SE3HyenaGrasp(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=FLAGS.positional_encoding_dimension,
                input_dimension_1=FLAGS.input_dimension_1,
                input_dimension_2=FLAGS.input_dimension_2,
                input_dimension_3=FLAGS.input_dimension_3,
            ).to(FLAGS.device)

        elif FLAGS.model == "Standard":
            model = StandardGrasp(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=8,
                input_dimension_1=8,
                input_dimension_2=8,
                input_dimension_3=8,
            ).to(FLAGS.device)
        elif FLAGS.model == "dgcnn":
            model = GraspDGCNN(k=20, emb_dims=1024, dropout=0.5).to(FLAGS.device)
        else:
            raise ValueError(f"Unknown model type: {FLAGS.model}")
    else:
        raise ValueError(f"Unknown task type: {FLAGS.task}")

    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    # Load full dataset
    dataset = GraspingDataset(
        root_dir=FLAGS.data_dir,
        resolution="pts_" + str(FLAGS.resolution),
        transform=None,
    )

    # Create DataLoaders
    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
    )

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, FLAGS.num_epochs, eta_min=1e-4
    )

    ### Log all args to wandb
    wandb.init(
        project="SE3-Hyena", name=FLAGS.name + str(FLAGS.resolution), config=vars(FLAGS)
    )
    wandb.save("*.txt")

    epoch = 1
    train_epoch(epoch, model, task_loss, dataloader, optimizer, scheduler, FLAGS)


def test_run_test_epoch():
    FLAGS, UNPARSED_ARGV = get_flags()
    FLAGS.task = "Normals"

    if FLAGS.task == "Normals":
        if FLAGS.model == "SE3Hyena":
            model = SE3HyenaNormal(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=FLAGS.positional_encoding_dimension,
                input_dimension_1=FLAGS.input_dimension_1,
                input_dimension_2=FLAGS.input_dimension_2,
                input_dimension_3=FLAGS.input_dimension_3,
            ).to(FLAGS.device)

        elif FLAGS.model == "Standard":
            model = StandardNormal(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=FLAGS.positional_encoding_dimension,
                input_dimension_1=FLAGS.input_dimension_1,
                input_dimension_2=FLAGS.input_dimension_2,
                input_dimension_3=FLAGS.input_dimension_3,
            ).to(FLAGS.device)
        elif FLAGS.model == "dgcnn":
            model = NormalDGCNN(k=20, emb_dims=1024, dropout=0.5).to(FLAGS.device)
        else:
            raise ValueError(f"Unknown model type: {FLAGS.model}")

    elif FLAGS.task == "Grasp":
        if FLAGS.model == "Standard":
            model = StandardGrasp(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=FLAGS.positional_encoding_dimension,
                input_dimension_1=FLAGS.input_dimension_1,
                input_dimension_2=FLAGS.input_dimension_2,
                input_dimension_3=FLAGS.input_dimension_3,
            ).to(FLAGS.device)

        else:
            raise ValueError(f"Unknown model type: {FLAGS.model}")
    else:
        raise ValueError(f"Unknown task type: {FLAGS.task}")

    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    # Load full dataset
    dataset = GraspingDataset(
        root_dir=FLAGS.data_dir,
        resolution="pts_" + str(FLAGS.resolution),
        transform=None,
    )

    # Create DataLoaders
    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
    )

    ### Log all args to wandb
    wandb.init(
        project="SE3-Hyena", name=FLAGS.name + str(FLAGS.resolution), config=vars(FLAGS)
    )
    wandb.save("*.txt")

    epoch = 1
    run_test_epoch(epoch, model, task_loss, dataloader, FLAGS)
