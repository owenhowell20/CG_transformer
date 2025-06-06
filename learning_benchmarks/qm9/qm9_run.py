### from utils.utils_profiling import *  # load before other local modules

import argparse
import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import math
import numpy as np
import torch
import wandb

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from qm9_models import model_selection
from qm9_dataset import collate
from qm9_flags import get_flags
from qm9_loss import norm2units, convert_to_ev
from torch_geometric.datasets import QM9
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from torch_geometric.utils import index_to_mask

from src.utils import to_np


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()

    num_iters = len(dataloader)
    for i, G in enumerate(dataloader):
        G = G.to(FLAGS.device)

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(G)
        targets = G.y
        l1_loss, __, rescale_loss = loss_fnc(pred, targets)

        # backprop
        l1_loss.backward()

        # clip gradients (norm-based, max norm = 1.0)
        if FLAGS.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if i % FLAGS.print_interval == 0:
            print(
                f"[{epoch}|{i}] l1 loss: {l1_loss:.5f} rescale loss: {rescale_loss:.5f} [units]"
            )

        if i % FLAGS.log_interval == 0:
            wandb.log(
                {
                    "Train L1 loss": to_np(l1_loss),
                    "Train Rescale loss": to_np(rescale_loss),
                    "Learning Rate": scheduler.get_last_lr()[0],
                }
            )

        scheduler.step(epoch + i / num_iters)


def val_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    for i, G in enumerate(dataloader):
        G = G.to(FLAGS.device)
        targets = G.y

        # run model forward and compute loss
        pred = model(G).detach()
        __, __, rl = loss_fnc(pred, targets, use_mean=False)
        rloss += rl
    rloss /= len(dataloader.dataset)

    print(f"...[{epoch}|val] rescale loss: {rloss:.5f} [units]")
    wandb.log({"Val Rescale loss": to_np(rloss)})


def run_test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    for i, G in enumerate(dataloader):
        G = G.to(FLAGS.device)
        targets = G.y

        # run model forward and compute loss
        pred = model(G).detach()
        __, __, rl = loss_fnc(pred, targets, use_mean=False)
        rloss += rl
    rloss /= len(dataloader.dataset)

    print(f"...[{epoch}|test] rescale loss: {rloss:.5f} [units]")
    wandb.log({"Test Rescale loss": to_np(rloss)})


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def main(FLAGS, UNPARSED_ARGV):
    target_names = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "u0",
        "u298",
        "h298",
        "g298",
        "cv",
    ]

    ### qm9 dataset
    dataset = QM9(root=FLAGS.dataset_path)

    task_name = FLAGS.task
    target_index = target_names.index(task_name)

    # targets = dataset.data.y[:, target_index]
    # #
    # # mean = targets.mean().item()
    # # std = targets.std().item()

    def task_loss(pred, target, use_mean=True):
        diff = pred - target[:, target_index]
        l1_loss = torch.sum(torch.abs(diff))
        l2_loss = torch.sum((diff) ** 2)
        if use_mean:
            l1_loss /= pred.shape[0]
            l2_loss /= pred.shape[0]

        # Rescale the l1 loss to original units
        rescale_loss = norm2units(
            l1_loss,
            task=FLAGS.task,
        )

        return l1_loss, l2_loss, rescale_loss

    num_data = len(dataset)

    try:
        # Canonical split from PyG
        split_idx = dataset.get_idx_split()

        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["val"]]
        test_dataset = dataset[split_idx["test"]]

    except:
        print("PyG split not found!!!")
        # Random permutation of indices
        perm = torch.randperm(num_data)

        # Define split sizes
        train_ratio, val_ratio = 0.6, 0.1
        num_train = int(train_ratio * num_data)
        num_val = int(val_ratio * num_data)
        num_test = num_data - num_train - num_val

        # Slice indices
        train_idx = perm[:num_train]
        val_idx = perm[num_train : num_train + num_val]
        test_idx = perm[num_train + num_val :]

        # Create subsets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=FLAGS.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=FLAGS.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=FLAGS.num_workers,
    )

    FLAGS.train_size = len(train_dataset)
    FLAGS.val_size = len(val_dataset)
    FLAGS.test_size = len(test_dataset)

    model = model_selection(FLAGS)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable:        {total - trainable:,}")

    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)
    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, eta_min=1e-4
    )

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + ".pt")

    ### Run training
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
        val_epoch(epoch, model, task_loss, val_loader, FLAGS)
        run_test_epoch(epoch, model, task_loss, test_loader, FLAGS)


if __name__ == "__main__":
    FLAGS, UNPARSED_ARGV = get_flags()

    ### Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Fix seed for random numbers
    if not FLAGS.seed:
        FLAGS.seed = 1992
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    FLAGS.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    ### Log all args to wandb
    wandb.init(project="SE3-Hyena", name=FLAGS.name, config=vars(FLAGS))
    wandb.save("*.txt")
    UNPARSED_ARGV = None

    ### run main
    main(FLAGS, UNPARSED_ARGV)
