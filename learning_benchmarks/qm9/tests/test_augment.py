from fixture import mock_qm9_batch, mock_qm9_dataloader, mock_so3_group
from qm9_models import qm9_SE3Hyenea
from qm9_flags import get_flags
from torch_geometric.utils import to_dense_batch
import torch
from qm9_utils import random_rotation


def test_augment_data(mock_qm9_dataloader):
    FLAGS, UNPARSED_ARGV = get_flags()

    pe_dim = 12
    output_dimension = 1

    model = qm9_SE3Hyenea(
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

    # Get positions and mask
    x, mask = to_dense_batch(batch.pos, batch.batch)

    b = batch.num_graphs  # Number of graphs in the batch
    N = x.shape[1]  # Number of nodes per graph (after padding)

    # Generate b random 3D translation vectors, one for each graph in the batch
    translations = torch.randn(b, 3).to(x.device)  # Shape: (b, 3)

    # Expand translations to shape (b, N, 3) so that each graph's nodes get the same translation
    expanded_translations = translations.unsqueeze(1).expand(-1, N, -1)

    # Apply translation to the positions
    x_new = x + expanded_translations

    # Apply translation only to non-padded nodes (using the mask)
    x_new = x_new * mask.unsqueeze(-1) + x * (~mask).unsqueeze(-1)

    # Now, reconstruct the batch with the new positions, while keeping padding intact
    batch_pos_flat = x_new.view(
        -1, 3
    )  # Flatten the positions to match original batch.pos shape

    # Make sure batch.pos is properly updated, while respecting padding
    batch_pos = torch.zeros_like(
        batch.pos
    )  # Initialize a tensor with the same shape as batch.pos
    batch_pos[mask.view(-1)] = batch_pos_flat[
        mask.view(-1)
    ]  # Assign new positions only to non-padded nodes

    # Set batch.pos to the updated positions
    batch.pos = batch_pos

    # Forward pass
    targets = batch.y  # Shape: [B, output_dim]
    pred_f = model(batch)
    assert pred_f.shape == targets[:, 0].shape
