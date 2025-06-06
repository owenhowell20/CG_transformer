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
from models import StandardGrasp, SE3HyenaGrasp, GraspDGCNN
from train import train_epoch, run_test_epoch
from flags import get_flags
from dataset import GraspingDataset


def test_baseline_model():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    b = 4
    N = 10

    pos = torch.randn(b, N, 3, device=device)
    normals = torch.randn(b, N, 3, device=device)

    model = StandardGrasp(
        sequence_length=N,  # Max number of points
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
        positional_encoding_type="pos_only",  ### pos_only, none
        use_normals=True,
    ).to(device)

    out, rot, dist = model(pos, normals)

    assert out.shape[0] == b, "Shape Mismatch"
    assert out.shape[1] == N, "Shape Mismatch"

    assert rot.shape[0] == b, "Shape Mismatch"
    assert rot.shape[1] == 3, "Shape Mismatch"
    assert rot.shape[2] == 3, "Shape Mismatch"

    assert dist.shape[0] == b, "Shape Mismatch"

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"

    assert isinstance(rot, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(rot).any(), "Output contains NaN values"

    assert isinstance(dist, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(dist).any(), "Output contains NaN values"


def test_DGCNN_model():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    b = 4
    N = 100

    pos = torch.randn(b, N, 3, device=device)
    normals = torch.randn(b, N, 3, device=device)

    model = GraspDGCNN(k=20, emb_dims=1024, use_normals=True, dropout=0.5).to(device)

    out, rot, dist = model(pos, normals)

    assert out.shape[0] == b, "Shape Mismatch"
    assert out.shape[1] == N, "Shape Mismatch"

    assert rot.shape[0] == b, "Shape Mismatch"
    assert rot.shape[1] == 3, "Shape Mismatch"
    assert rot.shape[2] == 3, "Shape Mismatch"

    assert dist.shape[0] == b, "Shape Mismatch"

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"

    assert isinstance(rot, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(rot).any(), "Output contains NaN values"

    assert isinstance(dist, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(dist).any(), "Output contains NaN values"


def test_model():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    b = 4
    N = 10

    pos = torch.randn(b, N, 3, device=device)
    normals = torch.randn(b, N, 3, device=device)

    model = SE3HyenaGrasp(
        sequence_length=N,  # Max number of points
        positional_encoding_dimension=8,
        input_dimension_1=8,
        input_dimension_2=8,
        input_dimension_3=8,
        positional_encoding_type="pos_only",  ### pos_only, none
        use_normals=True,
    ).to(device)

    out, rot, dist = model(pos, normals)

    assert out.shape[0] == b, "Shape Mismatch"
    assert out.shape[1] == N, "Shape Mismatch"

    assert rot.shape[0] == b, "Shape Mismatch"
    assert rot.shape[1] == 3, "Shape Mismatch"
    assert rot.shape[2] == 3, "Shape Mismatch"

    assert dist.shape[0] == b, "Shape Mismatch"

    assert isinstance(out, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(out).any(), "Output contains NaN values"

    assert isinstance(rot, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(rot).any(), "Output contains NaN values"

    assert isinstance(dist, torch.Tensor), "Output is not a tensor"
    assert not torch.isnan(dist).any(), "Output contains NaN values"


#
# def test_full_model_equivariant():
#     # Set device
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")
#
#     batch_size = 4
#     num_tokens = 10
#
#     pos = torch.randn(batch_size, num_tokens, 3, device=device)
#     normals = torch.randn(batch_size, num_tokens, 3, device=device)
#
#     model = SE3HyenaGrasp(
#         sequence_length=num_tokens,  # Max number of points
#         positional_encoding_dimension=8,
#         input_dimension_1=8,
#         input_dimension_2=8,
#         input_dimension_3=8,
#         positional_encoding_type="pos_only",  ### pos_only, none
#     ).to(device)
#
#     ### apply model
#     out = model(pos)
#     so3_group = no_base_space(SO3())
#
#     # Type-1 (vector) representations
#     type1_representation = so3_group.irrep(1)
#
#     pos = pos.reshape(batch_size * num_tokens, 3)  # (BN,3)
#     normals = normals.reshape(batch_size * num_tokens, 3)  # (BN,3 )
#     out = out.reshape(batch_size * num_tokens, 3)
#
#     # Field types
#     vector_type = escnn.nn.FieldType(so3_group, [type1_representation])
#
#     ### Wrap tensors
#     pos = escnn.nn.GeometricTensor(pos, vector_type)
#     normals = escnn.nn.GeometricTensor(normals, vector_type)
#     out = escnn.nn.GeometricTensor(out, vector_type)
#
#     # apply G transformation
#     g = so3_group.fibergroup.sample()
#
#     # Apply the transformation to the vector features (x)
#     pos_g = pos.transform(g)
#     normals_g = normals.transform(g)
#     out_g = out.transform(g)
#
#     ### apply model to transform
#     pos_g = pos_g.tensor.reshape(batch_size, num_tokens, 3)
#     normals_g = normals_g.tensor.reshape(batch_size, num_tokens, 3)
#     out_g = out_g.tensor.reshape(batch_size, num_tokens, 3)
#     g_out = model(pos_g)
#
#     assert g_out.shape == out_g.shape, "Shape mismatch"
#     assert torch.allclose(out_g, g_out, atol=1e-5), "Projection model not equivariant"
#
#


# def test_flags():
#     FLAGS, UNPARSED_ARGV = get_flags()
#     assert True
#
#
# def test_train_epoch():
#
#     FLAGS, UNPARSED_ARGV = get_flags()
#
#     model = SE3HyenaGrasp(
#         sequence_length=FLAGS.resolution,  # Max number of points
#         positional_encoding_dimension=8,
#         input_dimension_1=8,
#         input_dimension_2=8,
#         input_dimension_3=8,
#         positional_encoding_type="pos_only",  ### pos_only, none
#     ).to(FLAGS.device)
#
#     criterion = nn.MSELoss()
#     criterion = criterion.to(FLAGS.device)
#     task_loss = criterion
#
#     # Load full dataset
#     dataset = GraspingDataset(
#         root_dir=FLAGS.data_dir,
#         resolution="pts_" + str(FLAGS.resolution),
#         transform=None,
#     )
#
#     # Create DataLoaders
#     dataloader = DataLoader(
#         dataset,
#         batch_size=FLAGS.batch_size,
#         shuffle=True,
#         num_workers=FLAGS.num_workers,
#     )
#
#     # Optimizer settings
#     optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#         optimizer, FLAGS.num_epochs, eta_min=1e-4
#     )
#
#     ### Log all args to wandb
#     wandb.init(
#         project="SE3-Hyena", name=FLAGS.name + str(FLAGS.resolution), config=vars(FLAGS)
#     )
#     wandb.save("*.txt")
#
#     epoch = 1
#     train_epoch(epoch, model, task_loss, dataloader, optimizer, scheduler, FLAGS)
#
#
# def test_run_test_epoch():
#
#     FLAGS, UNPARSED_ARGV = get_flags()
#
#     model = SE3HyenaGrasp(
#         sequence_length=FLAGS.resolution,  # Max number of points
#         positional_encoding_dimension=8,
#         input_dimension_1=8,
#         input_dimension_2=8,
#         input_dimension_3=8,
#         positional_encoding_type="pos_only",  ### pos_only, none
#     ).to(FLAGS.device)
#
#     criterion = nn.MSELoss()
#     criterion = criterion.to(FLAGS.device)
#     task_loss = criterion
#
#     # Load full dataset
#     dataset = GraspingDataset(
#         root_dir=FLAGS.data_dir,
#         resolution="pts_" + str(FLAGS.resolution),
#         transform=None,
#     )
#
#     # Create DataLoaders
#     dataloader = DataLoader(
#         dataset,
#         batch_size=FLAGS.batch_size,
#         shuffle=True,
#         num_workers=FLAGS.num_workers,
#     )
#
#     ### Log all args to wandb
#     wandb.init(
#         project="SE3-Hyena", name=FLAGS.name + str(FLAGS.resolution), config=vars(FLAGS)
#     )
#     wandb.save("*.txt")
#
#     epoch = 1
#     run_test_epoch(epoch, model, task_loss, dataloader, FLAGS)
