import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def GraphLaplicianEmbeddings(pc, knn: int = 5):
    """
    Compute the Graph Laplacian Embedding of a point cloud.

    Args:
        pc (Tensor): A tensor of shape [b, N, 3] representing point clouds in 3D.
        knn (int): The number of nearest neighbors to use when constructing the adjacency matrix.

    Returns:
        Tensor: The Laplacian matrix of shape [b, N, N], where each graph is represented by a Laplacian matrix.
    """
    batch_size, num_points, _ = pc.shape
    laplacians = []

    for i in range(batch_size):
        # Get the point cloud for the current batch item
        point_cloud = (
            pc[i].cpu().numpy()
        )  # Convert to numpy for nearest neighbors search

        # Compute the KNN for the current point cloud
        knn_model = NearestNeighbors(n_neighbors=knn)
        knn_model.fit(point_cloud)
        distances, indices = knn_model.kneighbors(point_cloud)

        # Construct the adjacency matrix based on KNN
        adj_matrix = torch.zeros((num_points, num_points), device=pc.device)
        for j in range(num_points):
            adj_matrix[j, indices[j]] = torch.exp(
                -distances[j] ** 2
            )  # Gaussian kernel for weights

        # Construct the degree matrix (D)
        degree_matrix = torch.diag(adj_matrix.sum(dim=1))

        # Compute the graph Laplacian: L = D - A
        laplacian = degree_matrix - adj_matrix
        laplacians.append(laplacian)

    # Stack the laplacians for all batches
    laplacians = torch.stack(laplacians, dim=0)

    return laplacians
