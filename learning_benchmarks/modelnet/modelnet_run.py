import argparse
import os
import sys
import warnings
import time
from datetime import datetime

warnings.simplefilter(action="ignore", category=FutureWarning)

import math
import numpy as np
import torch
import wandb

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from modelnet_se3hyena import ModelNetSE3Hyena
from modelnet_dgcnn import ModelNetDGCNN, ModelNetDGCNNLoss
from modelnet_pointnet2 import ModelNetPointNet2
from modelnet_dataset import get_modelnet_dataset
from modelnet_flags import get_flags
from modelnet_loss import get_classification_loss
from modelnet_utils import to_np, rotate_point_cloud, scale_point_cloud


def train_epoch(epoch, model, dataloader, optimizer, scheduler, FLAGS):
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    num_iters = len(dataloader)
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        try:
            batch = batch.to(FLAGS.device)
            target = batch.y
            if target.dim() > 1:
                target = target.squeeze(-1)  # Remove extra dimension if present

            optimizer.zero_grad()

            # Data augmentation (random rotation and scaling) - only for DGCNN
            if FLAGS.use_random_rotation:
                # Apply rotation to each batch separately
                batch_pos = batch.pos
                batch_indices = batch.batch
                unique_batches = batch_indices.unique()

                # Import here to avoid circular import
                from modelnet_utils import random_rotation

                for b_idx in unique_batches:
                    mask = batch_indices == b_idx
                    points = batch_pos[mask]

                    # Generate random rotation
                    rotation = random_rotation().to(batch_pos.device)

                    # Apply rotation
                    batch_pos[mask] = torch.matmul(points, rotation.t())

                batch.pos = batch_pos  # Update positions

            # Forward pass
            try:
                pred = model(batch)

                # Calculate loss and accuracy
                loss, accuracy = get_classification_loss(pred, target)

                # Backprop
                loss.backward()

                # Clip gradients (norm-based, max norm = 1.0)
                if FLAGS.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                batch_size = target.size(0)
                total_loss += loss.item() * batch_size
                total_accuracy += accuracy.item() * batch_size
                total_samples += batch_size

                if i % FLAGS.print_interval == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{epoch}|{i}/{num_iters}] loss: {loss.item():.5f} "
                        f"accuracy: {accuracy.item():.5f} "
                        f"time: {elapsed:.2f}s"
                    )
                    start_time = time.time()

                if i % FLAGS.log_interval == 0:
                    wandb.log(
                        {
                            "Train Loss": loss.item(),
                            "Train Accuracy": accuracy.item(),
                            "Learning Rate": scheduler.get_last_lr()[0],
                            "Batch": epoch * num_iters + i,
                        }
                    )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM in batch {i}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Calculate epoch averages
    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0

    print(
        f"Epoch {epoch} | Train Loss: {avg_loss:.5f} | Train Accuracy: {avg_accuracy:.5f}"
    )
    wandb.log(
        {
            "Epoch": epoch,
            "Epoch Train Loss": avg_loss,
            "Epoch Train Accuracy": avg_accuracy,
        }
    )

    return avg_loss, avg_accuracy


def eval_epoch(epoch, model, dataloader, FLAGS, type="Val"):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(FLAGS.device)
            target = batch.y.squeeze(-1)  # Remove extra dimension if present

            # Forward pass
            pred = model(batch)

            # Calculate loss and accuracy
            loss, accuracy = get_classification_loss(pred, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_accuracy += accuracy.item() * batch_size
            total_samples += batch_size

    # Calculate epoch averages
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples

    print(
        f"Epoch {epoch} | {type} Loss: {avg_loss:.5f} | {type} Accuracy: {avg_accuracy:.5f}"
    )
    wandb.log(
        {
            f"Epoch {type} Loss": avg_loss,
            f"Epoch {type} Accuracy": avg_accuracy,
        }
    )

    return avg_accuracy


def main(FLAGS, UNPARSED_ARGV):
    # Get dataset
    train_loader, test_loader, num_classes = get_modelnet_dataset(
        root=os.path.join(FLAGS.dataset_path, f"ModelNet{FLAGS.modelnet_version}"),
        name=FLAGS.modelnet_version,
        num_points=FLAGS.num_points,
        use_normals=FLAGS.use_normals,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
    )

    # Initialize model
    if FLAGS.model == "SE3Hyena":
        model = ModelNetSE3Hyena(
            num_classes=num_classes,
            sequence_length=FLAGS.num_points,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
            kernel_size=3,
            scalar_attention_type="Standard",
        )
    elif FLAGS.model == "PointNet2":
        model = ModelNetPointNet2(
            num_classes=num_classes,
            set_abstraction_ratio_1=FLAGS.set_abstraction_ratio_1,
            set_abstraction_ratio_2=FLAGS.set_abstraction_ratio_2,
            set_abstraction_radius_1=FLAGS.set_abstraction_radius_1,
            set_abstraction_radius_2=FLAGS.set_abstraction_radius_2,
            feature_dim_1=FLAGS.feature_dim_1,
            feature_dim_2=FLAGS.feature_dim_2,
            feature_dim_3=FLAGS.feature_dim_3,
            dropout=FLAGS.dropout,
        )
    else:  # DGCNN
        model = ModelNetDGCNN(
            num_classes=num_classes, k=FLAGS.k, emb_dims=FLAGS.input_dimension_1
        )

    # Print model info
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable:        {total - trainable:,}")

    # Load model if specified
    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))

    model.to(FLAGS.device)
    wandb.watch(model)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FLAGS.num_epochs, eta_min=1e-5
    )

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + ".pt")
    best_save_path = os.path.join(FLAGS.save_dir, FLAGS.name + "_best.pt")

    # Training loop
    best_accuracy = 0.0

    for epoch in range(FLAGS.num_epochs):
        # Save current model
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        # Train and evaluate
        train_loss, train_accuracy = train_epoch(
            epoch, model, train_loader, optimizer, scheduler, FLAGS
        )
        val_accuracy = eval_epoch(epoch, model, test_loader, FLAGS, type="Test")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_save_path)
            print(f"New best accuracy: {best_accuracy:.5f}")
            print(f"Saved best model: {best_save_path}")

    # Final test with best model
    model.load_state_dict(torch.load(best_save_path))
    eval_epoch(FLAGS.num_epochs, model, test_loader, FLAGS, type="Final Test")


if __name__ == "__main__":
    FLAGS, UNPARSED_ARGV = get_flags()

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Fix seed for random numbers
    if not FLAGS.seed:
        FLAGS.seed = 42
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    FLAGS.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Log all args to wandb
    wandb.init(project="SE3-Hyena", name=FLAGS.name, config=vars(FLAGS))
    wandb.save("*.txt")
    UNPARSED_ARGV = None

    # Run main
    main(FLAGS, UNPARSED_ARGV)
