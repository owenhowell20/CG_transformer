import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the parent directory to the Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from graspbench.dataset import GraspingDataset
from graspbench.flags import get_flags
from graspbench.models import SE3HyenaGrasp, StandardGrasp, GraspDGCNN
import wandb

import tempfile
import os
import wandb


def plot_point_clouds_with_normals(
    point_clouds,
    normals=None,
    sample_every=50,
    show_normals=True,
    save_path="gt_normals.png",
):
    assert point_clouds.shape[0] == 16, "Expected 16 point clouds"

    if show_normals:
        assert (
            normals is not None and normals.shape == point_clouds.shape
        ), "Normals must be provided and match point cloud shape if show_normals is True"

    fig = plt.figure(figsize=(12, 12))

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, projection="3d")
        pts = point_clouds[i].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, alpha=0.6)

        if show_normals:
            nrm = normals[i].cpu().numpy()
            sampled = slice(0, None, sample_every)
            ax.quiver(
                pts[sampled, 0],
                pts[sampled, 1],
                pts[sampled, 2],
                nrm[sampled, 0],
                nrm[sampled, 1],
                nrm[sampled, 2],
                length=0.02,
                normalize=True,
                color="r",
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_predicted_point_clouds_with_normals(
    point_clouds,
    predicted_normals,
    normals=None,
    sample_every=50,
    show_normals=True,
    save_path="pred_vs_gt_normals.png",
):
    B = point_clouds.shape[0]
    assert B == 16, "Currently only supports 16 point clouds for 4x4 grid"
    assert predicted_normals.shape == point_clouds.shape
    if normals is not None:
        assert normals.shape == point_clouds.shape

    fig = plt.figure(figsize=(12, 12))

    for i in range(B):
        ax = fig.add_subplot(4, 4, i + 1, projection="3d")
        pts = point_clouds[i].cpu().numpy()
        preds = predicted_normals[i].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, alpha=0.6)

        if show_normals:
            sampled = slice(0, None, sample_every)

            ax.quiver(
                pts[sampled, 0],
                pts[sampled, 1],
                pts[sampled, 2],
                preds[sampled, 0],
                preds[sampled, 1],
                preds[sampled, 2],
                length=0.02,
                normalize=True,
                color="r",
            )

            if normals is not None:
                gt = normals[i].cpu().numpy()
                ax.quiver(
                    pts[sampled, 0],
                    pts[sampled, 1],
                    pts[sampled, 2],
                    gt[sampled, 0],
                    gt[sampled, 1],
                    gt[sampled, 2],
                    length=0.02,
                    normalize=True,
                    color="b",
                    alpha=0.5,
                )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f"Cloud {i + 1}", fontsize=8)
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_normal_comparisons(
    point_cloud: torch.Tensor,
    true_normals: torch.Tensor,
    predicted_normals: torch.Tensor,
    sample_every: int = 200,  # Increased sampling interval to reduce noise
    save_path: str = "normal_comparison.png",
):
    """
    Plot point cloud with normals in four identical views.

    Args:
        point_cloud: Point cloud tensor of shape [N, 3]
        true_normals: Ground truth normals tensor of shape [N, 3]
        predicted_normals: Model predicted normals tensor of shape [N, 3]
        sample_every: Sample rate for normal vectors to avoid overcrowding
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(20, 20))

    pts = point_cloud.cpu().numpy()
    true_nrm = true_normals.cpu().numpy()
    sampled = slice(0, None, sample_every)

    # Use the same view for all four panels
    elev, azim = 30, 45

    for i in range(4):
        ax = fig.add_subplot(221 + i, projection="3d")

        # Plot full point cloud with larger points and higher alpha
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, alpha=0.6, color="gray")

        # Plot sampled points with normals
        ax.scatter(
            pts[sampled, 0],
            pts[sampled, 1],
            pts[sampled, 2],
            s=2.0,
            alpha=1.0,
            color="blue",
        )

        # Plot normals at sampled points
        for j in range(0, len(pts), sample_every):
            ax.quiver(
                pts[j, 0],
                pts[j, 1],
                pts[j, 2],
                true_nrm[j, 0],
                true_nrm[j, 1],
                true_nrm[j, 2],
                length=0.1,  # Increased length for better visibility
                normalize=True,
                color="b",
                alpha=0.8,
            )

        ax.set_title("")  # Empty title
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_rotated_normals(
    normals: torch.Tensor, angle_degrees: float = 45
) -> torch.Tensor:
    """
    Create rotated versions of the input normals by rotating around the z-axis.

    Args:
        normals: Input normals tensor of shape [N, 3]
        angle_degrees: Rotation angle in degrees

    Returns:
        Rotated normals tensor of same shape as input
    """
    angle_rad = torch.tensor(angle_degrees * torch.pi / 180.0)
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)

    # Rotation matrix around z-axis
    R = torch.tensor([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

    # Apply rotation to each normal
    rotated_normals = torch.matmul(normals, R.T)
    return rotated_normals


if __name__ == "__main__":
    FLAGS, UNPARSED_ARGV = get_flags()

    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "fine_data")

    print(f"Looking for data in: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")

    data = GraspingDataset(root_dir=data_dir, resolution="pts_4096", transform=None)

    num_samples = 16
    point_clouds = []
    normals = []
    for i, sample in enumerate(data):
        pos = sample["pos"]
        normal = sample["normals"]

        point_clouds.append(pos)
        normals.append(normal)

        if i >= num_samples - 1:
            break

    point_clouds = torch.stack(point_clouds, dim=0)
    normals = torch.stack(normals, dim=0)

    # Create output directory if it doesn't exist
    output_dir = "visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)

    plot_point_clouds_with_normals(
        point_clouds,
        normals,
        sample_every=1000,
        show_normals=True,
        save_path=os.path.join(output_dir, "gt_normals.png"),
    )

    # Create rotated normals instead of using model predictions
    predicted_normals = torch.stack([create_rotated_normals(n) for n in normals])

    plot_predicted_point_clouds_with_normals(
        point_clouds,
        predicted_normals,
        normals=normals,
        sample_every=50,
        show_normals=True,
        save_path=os.path.join(output_dir, "pred_vs_gt_normals.png"),
    )

    # Add normal comparison plot for first object using true normals
    plot_normal_comparisons(
        point_clouds[0],
        normals[0],  # Use true normals
        normals[
            0
        ],  # Use true normals again (since we're just showing the same view 4 times)
        sample_every=200,
        save_path=os.path.join(output_dir, "normal_comparison.png"),
    )

    wandb.finish()
