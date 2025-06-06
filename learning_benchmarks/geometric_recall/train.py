import os
import sys

from learning_benchmarks.geometric_recall.models import (
    GRDSE3Hyena,
    GRDHyena,
    GRDStandard,
    GRDSE3HyperHyena,
)
from learning_benchmarks.geometric_recall.loss import compute_loss
from learning_benchmarks.geometric_recall.GeometricRecallDataset import (
    VectorEquivariantRecall,
)
from learning_benchmarks.geometric_recall.flags import get_flags

from src.utils import random_rotation_batch, to_np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import wandb

from torch import optim
from torch.utils.data import DataLoader, random_split


def train_epoch(
    global_step, epoch, model, loss_function, dataloader, optimizer, schedule, FLAGS
):
    model = model.to(FLAGS.device)
    model.train()

    num_iters = len(dataloader)
    assert num_iters > 0, "data not loaded correctly"
    loss_log = []
    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, commit=True)

    for i, g in enumerate(dataloader):
        key = g["key"].to(FLAGS.device)  # (B, N, 3)
        value = g["value"].to(FLAGS.device)  # (B, N, 3)
        question = g["question"].to(FLAGS.device)  # (B, 3)
        answer = g["answer"].to(FLAGS.device)  # (B, 3)

        B, N, _ = key.shape

        ### Generate random SE(3) transforms
        R = random_rotation_batch(B, device=FLAGS.device)  # (B, 3, 3)
        t = torch.randn(B, 1, 3, device=FLAGS.device)  # (B, 1, 3)

        # Apply rotation + translation to key and value
        key = torch.bmm(key, R.transpose(1, 2)) + t  # (B, N, 3)
        value = torch.bmm(value, R.transpose(1, 2)) + t  # (B, N, 3)

        # Apply same rotation + translation to question and answer
        question = torch.bmm(question.unsqueeze(1), R.transpose(1, 2)).squeeze(
            1
        ) + t.squeeze(
            1
        )  # (B, 3)
        answer = torch.bmm(answer.unsqueeze(1), R.transpose(1, 2)).squeeze(
            1
        ) + t.squeeze(
            1
        )  # (B, 3)

        # Reshape the key and value tensors to interleave them
        b, N, _ = key.shape
        output = torch.zeros((b, 2 * N, 3), device=FLAGS.device)

        # Fill the output tensor by interleaving key and value
        output[:, 0::2, :] = key  # Place the keys at even indices (0, 2, 4, ...)
        output[:, 1::2, :] = value  # Place the values at odd indices (1, 3, 5, ...)

        ### form model input:
        result = torch.cat((output, question.unsqueeze(1)), dim=1)

        optimizer.zero_grad()
        prediction = model(result)
        loss = loss_function(prediction, answer)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if i % FLAGS.print_interval == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    wandb.log({f"gradient_norm/{name}": grad_norm})

        optimizer.step()
        global_step = global_step + 1

        loss_log.append(to_np(loss))
        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss: {loss:.5f}")

        if i % FLAGS.log_interval == 0:
            wandb.log({"Train Batch Loss": loss.item()}, commit=True)

        schedule.step(epoch + i / num_iters)

    # Loss logging
    average_loss = np.mean(np.array(loss_log))
    wandb.log({"Train Epoch Loss": average_loss.item()}, commit=False)


### evaluate on test set
def run_forward_pass(global_step, model, *, dataloader, loss_function, FLAGS):
    model = model.to(FLAGS.device)
    if hasattr(model, "eval"):
        model.eval()

    num_iter = len(dataloader)

    losses = []
    for i, g in enumerate(dataloader):
        key = g["key"].to(FLAGS.device)
        value = g["value"].to(FLAGS.device)
        question = g["question"].to(FLAGS.device)  ## (b, 3)
        ### form model input:
        result = torch.cat((key, value, question.unsqueeze(1)), dim=1)
        answer = g["answer"].to(FLAGS.device)  ### (b,3)

        pred = model(result)
        loss = loss_function(pred, answer)
        losses.append(to_np(loss))

        wandb.log({"Test Batch Loss": loss.item()}, commit=True)

    average_loss = np.mean(np.array(losses))
    wandb.log({"Test Epoch Loss": average_loss.item()}, commit=True)

    return True


def main(FLAGS, UNPARSED_ARGV):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Using CPU.")
        device = "cpu"

    ### Instantiate the vector dataset
    vector_dataset = VectorEquivariantRecall(
        sequence_length=FLAGS.sequence_length,
        random=True,
    )

    ### Define the split sizes
    train_ratio = 0.8
    total_size = len(vector_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    ### Split the dataset
    train_dataset, test_dataset = random_split(vector_dataset, [train_size, test_size])

    ### Wrap the datasets in DataLoaders
    dataloader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=FLAGS.num_workers,
        drop_last=False,
    )

    ### model selection
    if FLAGS.model == "SE3Hyena":
        model = GRDSE3Hyena(
            sequence_length=2 * FLAGS.sequence_length + 1,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
            vector_attention_type="FFT",
        )
    elif FLAGS.model == "Standard":
        model = GRDStandard(
            sequence_length=2 * FLAGS.sequence_length + 1,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_dimension_1=FLAGS.input_dimension_1,
            input_dimension_2=FLAGS.input_dimension_2,
            input_dimension_3=FLAGS.input_dimension_3,
        )
    elif FLAGS.model == "SE3HyperHyena":
        model = GRDSE3HyperHyena(
            sequence_length=2 * FLAGS.sequence_length + 1,
            positional_encoding_dimension=FLAGS.positional_encoding_dimension,
            input_multiplicity_1=64,
            input_harmonic_1=1,
            hidden_multiplicity_1=8,
            hidden_harmonic_1=5,
            hidden_multiplicity_2=8,
            hidden_harmonic_2=5,
            hidden_multiplicity_3=8,
            hidden_harmonic_3=5,
        )
    else:
        raise ValueError(f"Unknown model type: {FLAGS.model}")

    ### print total model paramaters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {total_params}")

    ### print param intialization
    for name, param in model.named_parameters():
        print(name, param.data.std())

    if isinstance(model, torch.nn.Module):
        model.to(device)

    ### check all layers trainable
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Layer {name} is non-trainable.")
        else:
            print(f"Layer {name} is trainable.")

    wandb.watch(model, log="all", log_freq=FLAGS.log_interval)

    ###GRD loss
    task_loss = compute_loss

    # if FLAGS.restore:
    #    model.load_state_dict(torch.load(FLAGS.restore))

    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FLAGS.num_epochs, eta_min=0.1 * FLAGS.lr
    )
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + ".pt")

    global_step = 0
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        run_forward_pass(
            global_step,
            model,
            dataloader=test_dataloader,
            loss_function=task_loss,
            FLAGS=FLAGS,
        )

        train_epoch(
            global_step,
            epoch,
            model,
            task_loss,
            dataloader,
            optimizer,
            scheduler,
            FLAGS,
        )


def wrap_main(FLAGS, UNPARSED_ARGV):
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    for run in range(FLAGS.num_runs):
        wandb.init(
            project="SE3-Hyena",
            name=f"{FLAGS.model}{FLAGS.sequence_length}",
            config=vars(FLAGS),
            reinit=True,
        )
        main(FLAGS, UNPARSED_ARGV)


if __name__ == "__main__":
    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    for run in range(FLAGS.num_runs):
        wandb.init(
            project="SE3-Hyena",
            name=f"{FLAGS.model}{FLAGS.sequence_length}",
            config=vars(FLAGS),
            reinit=True,
        )
        main(FLAGS, UNPARSED_ARGV)
        FLAGS.seed += 1
