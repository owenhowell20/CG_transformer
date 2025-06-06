import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch
import numpy as np
import torch
import wandb

from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from nbody_dataloader import UnifiedDatasetWrapper
import dgl
from nbody_models import (
    nbody_SE3Hyenea,
    nbody_Standard,
    nbody_Hyena,
    nbody_GATr,
    nbody_SE3Transformer,
    nbody_TFN,
)
from nbody_flags import get_flags
from src.utils import to_np


def get_acc(pred, x_T, v_T, y=None, verbose=True):
    acc_dict = {}
    pred = to_np(pred)
    x_T = to_np(x_T)
    v_T = to_np(v_T)
    assert len(pred) == len(x_T)

    if verbose:
        y = np.asarray(y.cpu())
        _sq = (pred - y) ** 2
        acc_dict["mse"] = np.mean(_sq)

    _sq = (pred[:, 0, :] - x_T) ** 2
    acc_dict["pos_mse"] = np.mean(_sq)

    _sq = (pred[:, 1, :] - v_T) ** 2
    acc_dict["vel_mse"] = np.mean(_sq)

    return acc_dict


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, schedul, FLAGS):
    model.train()
    loss_epoch = 0

    N = FLAGS.sequence_length

    num_iters = len(dataloader)
    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, commit=False)
    for i, (g, y1, y2) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        # print(y1.shape, y2.shape)
        x_T = y1.to(FLAGS.device).view(-1, 3)
        v_T = y2.to(FLAGS.device).view(-1, 3)
        y = torch.stack(
            [x_T, v_T], dim=1
        )  ### stack target positions and target velocities
        optimizer.zero_grad()

        pred_xt, pred_vt = model(g)
        pred = torch.cat([pred_xt, pred_vt], dim=0)
        pred = pred.view(FLAGS.batch_size * FLAGS.sequence_length, 2, 3)
        assert pred.shape == y.shape, "dimension mismatch"
        loss = loss_fnc(pred, y)
        loss_epoch += to_np(loss)

        loss.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss:.5f}")

        if i % FLAGS.log_interval == 0:
            wandb.log({"Train Batch Loss": to_np(loss)}, commit=True)

        schedul.step(epoch + i / num_iters)

    loss_epoch /= len(dataloader)
    wandb.log({"Train Epoch Loss": loss_epoch}, commit=False)


def run_test_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    keys = ["pos_mse", "vel_mse"]
    acc_epoch = {k: 0.0 for k in keys}
    loss_epoch = 0.0
    for i, (g, y1, y2) in enumerate(dataloader):
        rot = RandomRotation()
        R = rot.get_rotation_matrix().to(FLAGS.device)

        g = g.to(FLAGS.device)
        y1 = y1.to(FLAGS.device)
        y2 = y2.to(FLAGS.device)

        # # Rotate node features in graph (positions and velocities)
        # x = g.x[:, 0:3]
        # v = g.x[:, 3:6]
        # x_rot = x @ R
        # v_rot = v @ R
        # g.x[:, 0:3] = x_rot
        # g.x[:, 3:6] = v_rot

        x_T = y1.view(-1, 3)  # @ R
        v_T = y2.view(-1, 3)  # @ R
        y = torch.stack([x_T, v_T], dim=1).to(FLAGS.device)

        # run model forward and compute loss
        pred_xt, pred_vt = model(g)
        pred_xt = pred_xt.detach()
        pred_vt = pred_vt.detach()
        pred = torch.cat([pred_xt, pred_vt], dim=0)

        pred = pred.view(FLAGS.batch_size * FLAGS.sequence_length, 2, 3)

        assert pred.shape == y.shape, "dimension mismatch"
        loss_epoch += to_np(loss_fnc(pred, y) / len(dataloader))
        acc = get_acc(pred, x_T, v_T, y=y)

        for k in keys:
            acc_epoch[k] += acc[k] / len(dataloader)

    print(f"...[{epoch}|test] loss: {loss_epoch:.5f}")
    print(pred[:5], y[:5])
    wandb.log({"Test loss": loss_epoch}, commit=False)
    for k in keys:
        wandb.log({"Test " + k: acc_epoch[k]}, commit=False)


def se3_get_acc(pred, x_T, v_T, y=None, verbose=True):
    acc_dict = {}
    pred = to_np(pred)
    x_T = to_np(x_T)
    v_T = to_np(v_T)
    assert len(pred) == len(x_T)

    if verbose:
        y = np.asarray(y.cpu())
        _sq = (pred - y) ** 2
        acc_dict["mse"] = np.mean(_sq)

    _sq = (pred[:, 0, :] - x_T) ** 2
    acc_dict["pos_mse"] = np.mean(_sq)

    _sq = (pred[:, 1, :] - v_T) ** 2
    acc_dict["vel_mse"] = np.mean(_sq)

    return acc_dict


def se3_train_epoch(epoch, model, loss_fnc, dataloader, optimizer, schedul, FLAGS):
    model.train()
    loss_epoch = 0

    num_iters = len(dataloader)
    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, commit=False)
    for i, (g, y1, y2) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        x_T = y1.to(FLAGS.device).view(-1, 3)
        v_T = y2.to(FLAGS.device).view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1)

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(g)
        loss = loss_fnc(pred, y)
        loss_epoch += to_np(loss)

        if torch.isnan(loss):
            import pdb

            pdb.set_trace()

        # backprop
        loss.backward()
        optimizer.step()

        # print to console
        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss:.5f}")

        # log to wandb
        if i % FLAGS.log_interval == 0:
            # 'commit' is only set to True here, meaning that this is where
            # wandb counts the steps
            wandb.log({"Train Batch Loss": to_np(loss)}, commit=True)

        # exit early if only do profiling
        if FLAGS.profile and i == 10:
            sys.exit()

        schedul.step(epoch + i / num_iters)

    # log train accuracy for entire epoch to wandb
    loss_epoch /= len(dataloader)
    wandb.log({"Train Epoch Loss": loss_epoch}, commit=False)


def se3_eval_epoch(epoch, model, loss_fnc, dataloader, FLAGS, dT):
    model.eval()

    keys = ["pos_mse", "vel_mse"]
    acc_epoch = {k: 0.0 for k in keys}
    acc_epoch_blc = {k: 0.0 for k in keys}  # for constant baseline
    acc_epoch_bll = {k: 0.0 for k in keys}  # for linear baseline
    loss_epoch = 0.0
    for i, (g, y1, y2) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        x_T = y1.view(-1, 3)
        v_T = y2.view(-1, 3)
        y = torch.stack([x_T, v_T], dim=1).to(FLAGS.device)

        # run model forward and compute loss
        pred = model(g).detach()
        loss_epoch += to_np(loss_fnc(pred, y) / len(dataloader))
        acc = get_acc(pred, x_T, v_T, y=y)
        for k in keys:
            acc_epoch[k] += acc[k] / len(dataloader)

        # eval constant baseline
        bl_pred = torch.zeros_like(pred)
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)
        for k in keys:
            acc_epoch_blc[k] += acc[k] / len(dataloader)

        # eval linear baseline
        # Apply linear update to locations.
        bl_pred[:, 0, :] = dT * g.ndata["v"][:, 0, :]
        acc = get_acc(bl_pred, x_T, v_T, verbose=False)
        for k in keys:
            acc_epoch_bll[k] += acc[k] / len(dataloader)

    print(f"...[{epoch}|test] loss: {loss_epoch:.5f}")
    wandb.log({"Test loss": loss_epoch}, commit=False)
    for k in keys:
        wandb.log({"Test " + k: acc_epoch[k]}, commit=False)
    wandb.log({"Const. BL pos_mse": acc_epoch_blc["pos_mse"]}, commit=False)
    wandb.log({"Linear BL pos_mse": acc_epoch_bll["pos_mse"]}, commit=False)
    wandb.log({"Linear BL vel_mse": acc_epoch_bll["vel_mse"]}, commit=False)


class se3_RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


class RandomRotation:
    def __init__(self):
        pass

    def get_rotation_matrix(self):
        M = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(M)
        return torch.tensor(Q, dtype=torch.float32)


def collate(samples):
    graphs = [sample for sample in samples]  # Each sample is already a Data object
    y1 = [graph.y_pos for graph in graphs]  # Extract target position (y_pos)
    y2 = [graph.y_vel for graph in graphs]  # Extract target velocity (y_vel)

    # Combine individual graphs into a batched graph using the Batch class
    batched_graph = Batch.from_data_list(graphs)
    y1 = torch.stack(y1)
    y2 = torch.stack(y2)

    return batched_graph, y1, y2


def se3_collate(samples):
    graphs, y1, y2 = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(y1), torch.stack(y2)


def main(FLAGS, UNPARSED_ARGV):
    # Prepare data
    train_dataset = UnifiedDatasetWrapper(FLAGS, split="train")
    if FLAGS.model == "SE3Transformer":
        train_loader = DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            collate_fn=se3_collate,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )

    sequence_length_train = train_dataset.n_points

    test_dataset = UnifiedDatasetWrapper(FLAGS, split="test")
    # drop_last is only here so that we can count accuracy correctly;
    if FLAGS.model == "SE3Transformer":
        test_loader = DataLoader(
            test_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            collate_fn=se3_collate,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )

    sequence_length_test = test_dataset.n_points

    if FLAGS.model != "SEGNN":
        dT = (
            train_dataset.data["delta_T"]
            * train_dataset.data["sample_freq"]
            * FLAGS.ri_delta_t
        )

    assert sequence_length_train == sequence_length_test, "Mismatch number particles"
    FLAGS.sequence_length = sequence_length_train

    FLAGS.train_size = len(train_dataset)
    FLAGS.test_size = len(test_dataset)

    if FLAGS.model == "SE3Hyena":
        model = nbody_SE3Hyenea(
            sequence_length=2 * FLAGS.sequence_length,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
        )
    elif FLAGS.model == "Standard":
        model = nbody_Standard(
            sequence_length=2 * FLAGS.sequence_length,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
        )
    elif FLAGS.model == "Hyena":
        model = nbody_Hyena(
            sequence_length=2 * FLAGS.sequence_length,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
        )
    elif FLAGS.model == "GATr":
        model = nbody_GATr(
            sequence_length=2 * FLAGS.sequence_length,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
        )

    elif FLAGS.model == "SE3Transformer":
        model = nbody_SE3Transformer(
            FLAGS.num_layers,
            FLAGS.num_channels,
            num_degrees=FLAGS.num_degrees,
            div=FLAGS.div,
            n_heads=FLAGS.head,
            si_m=FLAGS.simid,
            si_e=FLAGS.siend,
            x_ij=FLAGS.xij,
        )
    elif FLAGS.mode == "TFN":
        model = nbody_TFN(
            FLAGS.num_layers,
            FLAGS.num_channels,
            num_degrees=FLAGS.num_degrees,
            div=FLAGS.div,
            n_heads=FLAGS.head,
            si_m=FLAGS.simid,
            si_e=FLAGS.siend,
            x_ij=FLAGS.xij,
        )

    else:
        raise ValueError(f"Unknown model type: {FLAGS.model}")

    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, FLAGS.num_epochs, eta_min=1e-4
    )
    criterion = nn.MSELoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + ".pt")

    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        if FLAGS.model == "SE3Transformer":
            se3_train_epoch(
                epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS
            )
            se3_eval_epoch(epoch, model, task_loss, test_loader, FLAGS, dT)
        else:
            train_epoch(
                epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS
            )
            run_test_epoch(epoch, model, task_loss, test_loader, FLAGS)


if __name__ == "__main__":
    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    ### Log all args to wandb
    wandb.init(project="SE3-Hyena", name=FLAGS.name, config=vars(FLAGS))
    wandb.save("*.txt")

    try:
        main(FLAGS, UNPARSED_ARGV)
    except Exception:
        import pdb, traceback

        traceback.print_exc()
        pdb.post_mortem()
