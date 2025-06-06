import argparse
import os
import sys
import warnings
import time
from datetime import datetime
from torch.utils.data import random_split, DataLoader

warnings.simplefilter(action="ignore", category=FutureWarning)

import math
import numpy as np
import torch
import wandb

from flags import get_flags

from models import (
    SE3HyenaNormal,
    StandardGrasp,
    StandardNormal,
    NormalDGCNN,
    SE3HyenaGrasp,
    GraspDGCNN,
)
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import GraspingDataset


from src.utils import to_np, random_rotation_matrices


class CosineDistanceLoss(nn.Module):
    def forward(self, prediction, target):
        pred_flat = prediction.reshape(-1, 3)
        target_flat = target.reshape(-1, 3)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
        cos_dist = 1 - cos_sim
        return cos_dist.mean()


def frobenius_distance_scaled(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Stable, normalized Frobenius distance between rotation matrices in [0, 2].

    Args:
        R1: Tensor of shape [B, 3, 3]
        R2: Tensor of shape [B, 3, 3]

    Returns:
        Tensor of shape [B], with values in [0, 2].
    """
    relative_rot = torch.matmul(R1.transpose(1, 2), R2)  # [B, 3, 3]
    trace = relative_rot[:, 0, 0] + relative_rot[:, 1, 1] + relative_rot[:, 2, 2]  # [B]
    return (3 - trace) / 2  # ∈ [0, 2]


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, schedule, FLAGS):
    model.train()
    loss_epoch = 0

    num_iters = len(dataloader)
    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, commit=False)
    for i, batch in enumerate(dataloader):
        pos = batch["pos"].to(FLAGS.device)  ### [b, N,3] point cloud
        normals = batch["normals"].to(FLAGS.device)  ### [b, N,3] vector normals
        optimal_idx = batch["idx"].to(FLAGS.device)  ### [b] ints, invariant
        optimal_rot = batch["rotation"].to(FLAGS.device)  ### [b,3,3], type 1 left
        optimal_depth = batch["depth"].to(FLAGS.device)  ### [b], invariant

        b = pos.shape[0]
        N = pos.shape[1]

        ### apply random rotation to pos and normals
        R = random_rotation_matrices(pos.shape[0]).to(FLAGS.device)

        ### Apply random rotation:
        pos = torch.einsum("bij,bnj->bni", R, pos)
        normals = torch.einsum("bij,bnj->bni", R, normals)
        optimal_rot = torch.bmm(R, optimal_rot)

        ### zero mean pos
        pos = pos - pos.mean(dim=1, keepdim=True)
        optimizer.zero_grad()

        # run model forward and compute loss
        if FLAGS.task == "Normals":
            pred = model(pos)
            assert pred.shape == normals.shape, "dimension mismatch"

            loss = loss_fnc(pred, normals)  ### compare prediction with normals
            loss_epoch += to_np(loss)

        elif FLAGS.task == "Grasp":
            probs, rots, dists = model(pos, normals)

            ### compute target probility distrubiton on nodes
            indices_unsqueezed = optimal_idx.unsqueeze(1)
            one_hot = torch.zeros(b, N, device=FLAGS.device).scatter_(
                1, indices_unsqueezed, 1
            )

            with torch.no_grad():
                # Compute pairwise distances per batch element: [B, N, N]
                emd_dists = torch.cdist(pos, pos, p=2)  # batch-wise pairwise distances

            ### Compute expected distance: sum_{i,j} p_i d_{ij} q_j
            loss_emd_approx = torch.einsum(
                "bi,bij,bj->b", one_hot, emd_dists, probs
            ).mean()

            ### Compute the cross-entropy: maybe need to weight by distances here
            loss_cross_ent = -(one_hot * torch.log(probs)).sum(dim=1).mean()
            loss = loss_emd_approx

            ### now compute the rotational part of loss; standard l2
            loss_rot = frobenius_distance_scaled(R1=rots, R2=optimal_rot).mean()
            loss += loss_rot

            ### now compute the depth error
            loss_dist = 0.1 * torch.abs(dists - optimal_depth).mean()
            loss += loss_dist

            loss_epoch += to_np(loss)
        else:
            raise ValueError(f"Unknown task type: {FLAGS.task}")

        ### backprop
        loss.backward()
        optimizer.step()

        ### print to console
        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss:.5f}")

        ### log to wandb
        if i % FLAGS.log_interval == 0:
            ### 'commit' is only set to True here, meaning that this is where
            ### wandb counts the steps
            if FLAGS.task == "Normals":
                wandb.log({"Train Batch Loss": to_np(loss)}, commit=True)
            else:
                wandb.log(
                    {
                        "Train Batch Loss": to_np(loss),
                        "Cross Entropy": to_np(loss_cross_ent),
                        "Expected Distance": to_np(loss_emd_approx),
                        "Rotational Error": to_np(loss_rot),
                        "Distance Error": to_np(loss_dist),
                    },
                    commit=True,
                )

        schedule.step(epoch + i / num_iters)

    # log train accuracy for entire epoch to wandb
    loss_epoch /= len(dataloader)
    wandb.log({"Train Epoch Loss": loss_epoch}, commit=False)


def run_test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    loss_epoch = 0
    for i, batch in enumerate(dataloader):
        pos = batch["pos"].to(FLAGS.device)  ### [b, N, 3] point cloud
        normals = batch["normals"].to(FLAGS.device)  ### [b, N,3] vector normals
        optimal_idx = batch["idx"].to(FLAGS.device)  ### [b] ints, invariant
        optimal_rot = batch["rotation"].to(FLAGS.device)  ### [b,3,3], type 1 left
        optimal_depth = batch["depth"].to(FLAGS.device)  ### [b], invariant

        b = pos.shape[0]
        N = pos.shape[1]

        ### apply random rotation to pos and normals
        R = random_rotation_matrices(pos.shape[0]).to(FLAGS.device)

        ### Apply rotation: (b, N, 3) x (b, 3, 3)^T => (b, N, 3)
        pos = torch.einsum("bij,bnj->bni", R, pos)
        normals = torch.einsum("bij,bnj->bni", R, normals)

        ### zero mean pos
        pos = pos - pos.mean(dim=1, keepdim=True)

        # run model forward and compute loss
        if FLAGS.task == "Normals":
            pred = model(pos)
            assert pred.shape == normals.shape, "dimension mismatch"

            loss = loss_fnc(pred, normals)  ### compare prediction with normals
            loss_epoch += to_np(loss)

        elif FLAGS.task == "Grasp":
            probs, rots, dists = model(pos, normals)

            ### compute target distribution on nodes
            indices_unsqueezed = optimal_idx.unsqueeze(1)
            one_hot = torch.zeros(b, N, device=FLAGS.device).scatter_(
                1, indices_unsqueezed, 1
            )

            with torch.no_grad():
                # Compute pairwise distances per batch element: [B, N, N]
                emd_dists = torch.cdist(pos, pos, p=2)  # batch-wise pairwise distances

            ### Compute expected distance: sum_{i,j} p_i d_{ij} q_j
            loss_emd_approx = torch.einsum(
                "bi,bij,bj->b", one_hot, emd_dists, probs
            ).mean()  #### this should be positive definite!

            ### Compute the cross-entropy: maybe need to weight by distances here
            loss_cross_ent = -(one_hot * torch.log(probs)).sum(dim=1).mean()
            loss = loss_emd_approx

            ### now compute the rotational part of loss; geodesdic SO(3) distance
            loss_rot = frobenius_distance_scaled(R1=rots, R2=optimal_rot).mean()
            loss += loss_rot

            ### now compute the depth error
            loss_dist = 0.1 * torch.abs(dists - optimal_depth).mean()
            loss += loss_dist

            loss_epoch += to_np(loss)
        else:
            raise ValueError(f"Unknown task type: {FLAGS.task}")

        loss_epoch += to_np(loss)

    print(f"...[{epoch}|test] loss: {loss_epoch:.5f}")
    wandb.log({"Test loss": loss_epoch}, commit=False)


def main(FLAGS, UNPARSED_ARGV):
    # Load full dataset
    dataset = GraspingDataset(
        root_dir=FLAGS.data_dir,
        resolution="pts_" + str(FLAGS.resolution),
        transform=None,
    )

    # Define split ratio
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
    )

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
        if FLAGS.model == "SE3Hyena":
            model = SE3HyenaGrasp(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=FLAGS.positional_encoding_dimension,
                input_dimension_1=FLAGS.input_dimension_1,
                input_dimension_2=FLAGS.input_dimension_2,
                input_dimension_3=FLAGS.input_dimension_3,
                use_normals=FLAGS.use_normals,
            ).to(FLAGS.device)

        elif FLAGS.model == "Standard":
            model = StandardGrasp(
                sequence_length=FLAGS.resolution,
                positional_encoding_dimension=FLAGS.positional_encoding_dimension,
                input_dimension_1=FLAGS.input_dimension_1,
                input_dimension_2=FLAGS.input_dimension_2,
                input_dimension_3=FLAGS.input_dimension_3,
                use_normals=FLAGS.use_normals,
            ).to(FLAGS.device)

        elif FLAGS.model == "dgcnn":
            model = GraspDGCNN(k=20, emb_dims=1024, use_normals=True, dropout=0.5).to(
                FLAGS.device
            )
        else:
            raise ValueError(f"Unknown model type: {FLAGS.model}")
    else:
        raise ValueError(f"Unknown task type: {FLAGS.task}")

    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, FLAGS.num_epochs, eta_min=1e-4
    )

    ### loss function
    if FLAGS.loss_type == "mse":
        criterion = CosineDistanceLoss()
    else:
        criterion = CosineDistanceLoss()

    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + ".pt")

    # Run training
    print("Begin training")
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
        run_test_epoch(epoch, model, task_loss, test_loader, FLAGS)


def wrap_main(FLAGS, UNPARSED_ARGV):
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    print("Start run:", FLAGS.resolution, FLAGS.model, FLAGS.task)

    ### Log all args to wandb
    wandb.init(
        project="SE3-Hyena",
        name=FLAGS.model + str(FLAGS.resolution),
        config=vars(FLAGS),
    )
    wandb.save("*.txt")

    try:
        main(FLAGS, UNPARSED_ARGV)
    except Exception:
        import pdb, traceback

        traceback.print_exc()
        pdb.post_mortem()

    wandb.finish()


if __name__ == "__main__":
    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    ### Log all args to wandb
    wandb.init(
        project="SE3-Hyena", name=FLAGS.name + str(FLAGS.resolution), config=vars(FLAGS)
    )
    wandb.save("*.txt")

    try:
        main(FLAGS, UNPARSED_ARGV)
    except Exception:
        import pdb, traceback

        traceback.print_exc()
        pdb.post_mortem()
