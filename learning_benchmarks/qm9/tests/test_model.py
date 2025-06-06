from fixture import mock_qm9_batch, mock_qm9_dataloader, mock_so3_group

from qm9.qm9_models import qm9_SE3Hyena
from qm9.qm9_flags import get_flags
import escnn
import escnn.nn
from torch_geometric.utils import to_dense_batch
import torch


def test_baseline_model(mock_qm9_dataloader):
    FLAGS, UNPARSED_ARGV = get_flags()

    pe_dim = 32
    output_dimension = 1

    model = qm9_SE3Hyena(
        positional_encoding_dimension=pe_dim,
        input_dimension_1=32,
        input_dimension_2=32,
        input_dimension_3=16,
        output_dimension=output_dimension,
        node_feature_dimension=11,
        edge_feature_dimension=4,
    ).to(FLAGS.device)

    for batch in mock_qm9_dataloader:
        batch = batch.to(FLAGS.device)
        targets = batch.y  # Shape: [B, output_dim]
        pred_f = model(batch)
        assert pred_f.shape == targets[:, 0].shape


### test that model outputs are invariant to translation
def test_model_trans_invariant(mock_qm9_dataloader):
    FLAGS, UNPARSED_ARGV = get_flags()

    pe_dim = 12
    output_dimension = 19

    model = qm9_SE3Hyena(
        positional_encoding_dimension=pe_dim,
        input_dimension_1=12,
        input_dimension_2=12,
        input_dimension_3=12,
        output_dimension=output_dimension,
        node_feature_dimension=11,
        edge_feature_dimension=4,
    ).to(FLAGS.device)

    for batch in mock_qm9_dataloader:
        batch = batch.to(FLAGS.device)
        pred_f = model(batch)

        ### get positions
        x, mask = to_dense_batch(batch.pos, batch.batch)

        batch_size = x.shape[0]
        num_tokens = x.shape[1]

        ### apply uniform shift to each token
        t_shift = (
            torch.randn(batch_size, 3, device=FLAGS.device)
            .unsqueeze(1)
            .repeat(1, num_tokens, 1)
        )

        x_trans = x + t_shift  # (BN,3)

        assert x_trans.shape == x.shape
        batch.pos = x_trans[mask.bool()]
        pred_trans_f = model(batch)

        assert pred_f.shape == pred_trans_f.shape
        assert torch.allclose(pred_f, pred_trans_f, atol=1e-5)


#
### test that model outputs are invariant to geometric transformation
def test_model_rot_invariant(mock_qm9_dataloader, mock_so3_group):
    FLAGS, UNPARSED_ARGV = get_flags()

    # Type-0 (invariant) and Type-1 (vector) representations
    type1_representation = mock_so3_group.irrep(1)

    pe_dim = 12
    output_dimension = 19

    model = qm9_SE3Hyena(
        positional_encoding_dimension=pe_dim,
        input_dimension_1=12,
        input_dimension_2=12,
        input_dimension_3=12,
        output_dimension=output_dimension,
        node_feature_dimension=11,
        edge_feature_dimension=4,
    ).to(FLAGS.device)

    for batch in mock_qm9_dataloader:
        batch = batch.to(FLAGS.device)
        pred_f = model(batch)

        ### get positions
        x, mask = to_dense_batch(batch.pos, batch.batch)

        batch_size = x.shape[0]
        num_tokens = x.shape[1]

        x_g = x.reshape(batch_size * num_tokens, 3)  # (BN,3)
        vector_type = escnn.nn.FieldType(mock_so3_group, [type1_representation])
        # Wrap tensors
        x_g = escnn.nn.GeometricTensor(x_g, vector_type)

        # random G transformation
        g = mock_so3_group.fibergroup.sample()

        # Apply the transformation to the vector features (x)
        x_g = x_g.transform(g)

        ### now replace x with x_g in batch
        x_g = x_g.tensor.reshape(batch_size, num_tokens, 3)

        assert x_g.shape == x.shape

        batch.pos = x_g[mask.bool()]

        pred_g_f = model(batch)

        assert pred_f.shape == pred_g_f.shape
        assert torch.allclose(pred_f, pred_g_f, atol=1e-5)


def test_model(mock_qm9_dataloader):
    FLAGS, UNPARSED_ARGV = get_flags()

    pe_dim = 12
    output_dimension = 1

    model = qm9_SE3Hyena(
        positional_encoding_dimension=pe_dim,
        input_dimension_1=12,
        input_dimension_2=12,
        input_dimension_3=12,
        output_dimension=output_dimension,
        node_feature_dimension=11,
        edge_feature_dimension=4,
    ).to(FLAGS.device)

    for batch in mock_qm9_dataloader:
        batch = batch.to(FLAGS.device)
        targets = batch.y  # Shape: [B, output_dim]
        pred_f = model(batch)
        assert pred_f.shape == targets[:, 0].shape
