import pytest
from torch.utils.data import DataLoader
import wandb
from torch import optim
import torch


from .test_flags import mock_flags
from learning_benchmarks.geometric_recall.loss import compute_loss
from learning_benchmarks.geometric_recall.models import (
    GRDSE3Hyena,
    GRDHyena,
    GRDStandard,
    GRDSE3HyperHyena,
)
from learning_benchmarks.geometric_recall.GeometricRecallDataset import (
    VectorEquivariantRecall,
)
from learning_benchmarks.geometric_recall.train import run_forward_pass, train_epoch


def test_vector_data():
    # Initialize the dataset
    data = VectorEquivariantRecall(sequence_length=8, random=True)

    # Get the first sample
    sample = data[0]

    # Extract key, value, query, question, and answer
    key = sample["key"]  # Shape: (sequence_length, 3)
    value = sample["value"]  # Shape: (sequence_length, 3)
    query = sample["query"]  # Shape: (sequence_length, 3)
    question = sample["question"]  # Shape: (3,)
    answer = sample["answer"]  # Shape: (3,)

    # Check that the key and value are 3-dimensional vectors
    assert key.shape[1] == 3, "Key vectors must be 3-dimensional"
    assert value.shape[1] == 3, "Value vectors must be 3-dimensional"
    assert query.shape[1] == 3, "Query vector must be 3-dimensional"

    # Test that the question and answer are correctly aligned in size
    assert question.shape[0] == 3, "Question vector wrong size"
    assert answer.shape[0] == 3, "Answer vector wrong size"


### test wandb configured correctly
def test_wandb():
    try:
        # Initialize a new run with a test project name
        wandb.init(project="SE3-Hyena", name="test_run_debug")

        # Log a simple metric to check if it's working
        wandb.log({"test_metric": 1.0})

        # Finish the run
        wandb.finish()

        assert True  # If no exception occurs, the test passes
    except Exception as e:
        print(f"WandB test failed: {e}")
        assert False


def test_run_forward_pass(mock_flags):
    FLAGS = mock_flags
    FLAGS.sequence_length = 8

    models = ["SE3Hyena", "Standard", "SE3HyperHyena"]

    for name_model in models:
        FLAGS.model = name_model

        ### model selection
        if FLAGS.model == "SE3Hyena":
            model = GRDSE3Hyena(
                sequence_length=2 * FLAGS.sequence_length + 1,
                positional_encoding_dimension=16,
                input_dimension_1=256,
                input_dimension_2=128,
                input_dimension_3=64,
            )
        elif FLAGS.model == "Standard":
            model = GRDStandard(
                sequence_length=2 * FLAGS.sequence_length + 1,
                positional_encoding_dimension=16,
                input_dimension_1=256,
                input_dimension_2=128,
                input_dimension_3=64,
            )
        elif FLAGS.model == "SE3HyperHyena":
            model = GRDSE3HyperHyena(
                sequence_length=2 * FLAGS.sequence_length + 1,
                input_multiplicity_1=8,
                input_harmonic_1=3,
                hidden_multiplicity_1=8,
                hidden_harmonic_1=3,
                input_multiplicity_2=8,
                input_harmonic_2=3,
                hidden_multiplicity_2=8,
                hidden_harmonic_2=3,
                input_multiplicity_3=8,
                input_harmonic_3=3,
                hidden_multiplicity_3=8,
                hidden_harmonic_3=3,
                input_multiplicity_4=8,
                input_harmonic_4=3,
            )
        else:
            raise ValueError(f"Unknown model type: {FLAGS.model}")

        vector_dataset = VectorEquivariantRecall(
            sequence_length=FLAGS.sequence_length, random=True
        )

        ### Wrap it in a DataLoader
        dataloader = DataLoader(
            vector_dataset,
            batch_size=4,  # FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )

        ### set number of iterations equal to length of dataset
        FLAGS.num_iter = len(dataloader)

        wandb.init(
            project="SE3-Hyena",
            name="test_" + str(model),
        )

        loss_function = compute_loss
        global_step = 0
        bool_val = run_forward_pass(
            global_step,
            model=model,
            dataloader=dataloader,
            loss_function=loss_function,
            FLAGS=FLAGS,
        )

        # Finish the run
        wandb.finish()

        assert True


def test_train_epoch(mock_flags):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    else:
        print("CUDA is not available. Using CPU.")
        device = "cpu"

    FLAGS = mock_flags
    FLAGS.sequence_length = 8

    FLAGS.model = "SE3HyperHyena"

    ### model selection
    if FLAGS.model == "SE3Hyena":
        model = GRDSE3Hyena(
            sequence_length=2 * FLAGS.sequence_length + 1,
            positional_encoding_dimension=16,
            input_dimension_1=256,
            input_dimension_2=128,
            input_dimension_3=64,
        )
    elif FLAGS.model == "Standard":
        model = GRDStandard(
            sequence_length=2 * FLAGS.sequence_length + 1,
            positional_encoding_dimension=16,
            input_dimension_1=256,
            input_dimension_2=128,
            input_dimension_3=64,
        )
    elif FLAGS.model == "SE3HyperHyena":
        model = GRDSE3HyperHyena(
            sequence_length=2 * FLAGS.sequence_length + 1,
            input_multiplicity_1=8,
            input_harmonic_1=3,
            hidden_multiplicity_1=8,
            hidden_harmonic_1=3,
            input_multiplicity_2=8,
            input_harmonic_2=3,
            hidden_multiplicity_2=8,
            hidden_harmonic_2=3,
            input_multiplicity_3=8,
            input_harmonic_3=3,
            hidden_multiplicity_3=8,
            hidden_harmonic_3=3,
            input_multiplicity_4=8,
            input_harmonic_4=3,
        )
    else:
        raise ValueError(f"Unknown model type: {FLAGS.model}")

    if isinstance(model, torch.nn.Module):
        model.to(device)

    vector_dataset = VectorEquivariantRecall(
        sequence_length=FLAGS.sequence_length, random=True
    )

    ### Wrap it in a DataLoader
    dataloader = DataLoader(
        vector_dataset,
        batch_size=4,  # FLAGS.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    ### set number of iterations equal to length of dataset
    FLAGS.num_iter = len(dataloader)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, FLAGS.num_epochs, eta_min=0.1 * FLAGS.lr
    )

    wandb.init(
        project="SE3-Hyena",
        name="test_train_epoch",
    )

    loss_function = compute_loss
    epoch = 0
    global_step = 0

    train_epoch(
        global_step,
        epoch,
        model,
        loss_function,
        dataloader,
        optimizer,
        scheduler,
        FLAGS,
    )

    # Finish the run
    wandb.finish()

    assert True
