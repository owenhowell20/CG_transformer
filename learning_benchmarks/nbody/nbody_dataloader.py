import os
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import dgl
import torch
import numpy as np
import os.path as osp
import pathlib

DTYPE = np.float32

import torch
import torch.nn.functional as F


def knn_graph(x, k):
    # Compute pairwise distances using cdist
    dist_matrix = torch.cdist(x, x, p=2)  # Euclidean distance matrix

    # Set the diagonal to a large value to avoid self-loops
    dist_matrix.fill_diagonal_(float("inf"))

    # Find the indices of the k smallest distances for each node
    _, indices = torch.topk(dist_matrix, k, largest=False, sorted=False)

    # Convert the indices to a long tensor
    indices_src = torch.arange(x.size(0)).unsqueeze(1).repeat(1, k)  # Source indices
    indices_dst = indices  # Destination indices

    return indices_src.flatten(), indices_dst.flatten()


class UnifiedDatasetWrapper(Dataset):
    def __init__(self, FLAGS, split=None):
        if FLAGS.model == "SE3Transformer":
            self.dataset = RIDataset(FLAGS, split)
        elif FLAGS.model in ["SE3Hyena", "GATr"]:
            self.dataset = RIDataset_PyG(FLAGS, split)
        elif FLAGS.model == "NBodyBaseline":
            self.dataset = NBodyDataset(
                partition=split or "train",
                max_samples=FLAGS.max_samples,
                dataset_name=FLAGS.dataset_name,
                graph_type=FLAGS.graph_type,
                k=FLAGS.k,
            )
        else:
            raise ValueError(f"Unknown model: {FLAGS.model}")

        # Forward key properties from the underlying dataset
        if hasattr(self.dataset, "n_points"):
            self.n_points = self.dataset.n_points
        if hasattr(self.dataset, "n_frames"):
            self.n_frames = self.dataset.n_frames
        if hasattr(self.dataset, "node_feature_size"):
            self.node_feature_size = self.dataset.node_feature_size

        # Expose .data if present (RIDataset and RIDataset_PyG)
        if hasattr(self.dataset, "data"):
            self.data = self.dataset.data
        else:
            self.data = None  # Optional fallback

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


### PYG object
class RIDataset_PyG(Dataset):
    node_feature_size = 1

    def __init__(self, FLAGS, split):
        self.FLAGS = FLAGS
        self.split = split

        # Choose dataset type
        if "charged" in FLAGS.ri_data_type:
            _data_type = "charged"
        else:
            assert "springs" in FLAGS.ri_data_type
            _data_type = "springs"

        assert split in ["test", "train"]
        filename = "ds_" + split + "_" + _data_type + "_3D_" + FLAGS.data_str
        filename = os.path.join(FLAGS.ri_data, filename + ".pkl")

        # Load dataset
        time_start = time.time()
        with open(filename, "rb") as file:
            data = pickle.load(file)

        data["points"] = np.swapaxes(data["points"], 2, 3)[:, FLAGS.ri_burn_in :]
        data["vel"] = np.swapaxes(data["vel"], 2, 3)[:, FLAGS.ri_burn_in :]

        if "sample_freq" not in data:
            data["sample_freq"] = 100
            data["delta_T"] = 0.001
            print("warning: sample_freq not found in dataset")

        self.data = data
        self.len = data["points"].shape[0]
        self.n_frames = data["points"].shape[1]
        self.n_points = data["points"].shape[2]

        if split == "train":
            print(data["points"][0, 0, 0])
            print(data["points"][-1, 30, 0])

    def __len__(self):
        return self.len

    def connect_fully(self, num_atoms):
        src = []
        dst = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index

    def __getitem__(self, idx):
        if self.FLAGS.ri_start_at == "zero":
            frame_0 = 0
        else:
            last_pssbl = self.n_frames - self.FLAGS.ri_delta_t
            if "vic" in self.FLAGS.data_str:
                frame_0 = 30
            elif self.split == "train":
                frame_0 = np.random.choice(range(last_pssbl))
            elif self.FLAGS.ri_start_at == "center":
                frame_0 = int(last_pssbl / 2)
            elif self.FLAGS.ri_start_at == "all":
                frame_0 = int(last_pssbl / self.len * idx)

        frame_T = frame_0 + self.FLAGS.ri_delta_t

        x_0 = torch.tensor(
            self.data["points"][idx, frame_0], dtype=torch.float32
        )  # [N, 3]
        assert x_0.shape[1] == 3, "point cloud not right dimension"

        x_T = torch.tensor(self.data["points"][idx, frame_T], dtype=torch.float32) - x_0
        v_0 = torch.tensor(self.data["vel"][idx, frame_0], dtype=torch.float32)
        v_T = torch.tensor(self.data["vel"][idx, frame_T], dtype=torch.float32) - v_0
        charges = torch.tensor(self.data["charges"][idx], dtype=torch.float32)

        edge_index = self.connect_fully(self.n_points)
        src, dst = edge_index[0], edge_index[1]

        # Edge features
        d = x_0[dst] - x_0[src]  # relative positions [E, 3]
        w = charges[dst] * charges[src]  # scalar charge product [E, 1]

        edge_attr = torch.cat([d, w], dim=1)  # [E, 4]

        # Node features (concatenated position, velocity, charge)
        node_feats = torch.cat(
            [x_0, v_0, charges], dim=1  # [N,  3]  # [N,  3]  # [N, 1]
        )  # [N, 7] after flattening

        # Generate a random permutation of integers from 0 to n-1
        perm = torch.randperm(node_feats.shape[0], device=node_feats.device)
        node_feats = node_feats[perm, :]

        assert node_feats.shape[1] == 7, "wrong dimension"
        data = Data(
            x=node_feats,  # [N, 7] (x, v, c)
            edge_index=edge_index,
            edge_attr=edge_attr,  # [E, 4]
            y_pos=x_T,  # target Δposition
            y_vel=v_T,  # target Δvelocity
        )
        return data


class RIDataset(torch.utils.data.Dataset):
    """Custom dgl dataset class taken from SE3-transformer.
    Modified to also generate knn graphs
    """

    node_feature_size = 1

    def __init__(self, FLAGS, split):
        """Create a dataset object"""

        self.FLAGS = FLAGS
        self.split = split

        if "charged" in FLAGS.ri_data_type:
            _data_type = "charged"
        else:
            assert "springs" in FLAGS.ri_data_type
            _data_type = "springs"

        assert split in ["test", "train"]
        filename = "ds_" + split + "_" + _data_type + "_3D_" + FLAGS.data_str
        filename = os.path.join(FLAGS.ri_data, filename + ".pkl")

        with open(filename, "rb") as file:
            data = pickle.load(file)

        data["points"] = np.swapaxes(data["points"], 2, 3)[:, FLAGS.ri_burn_in :]
        data["vel"] = np.swapaxes(data["vel"], 2, 3)[:, FLAGS.ri_burn_in :]

        if "sample_freq" not in data.keys():
            data["sample_freq"] = 100
            data["delta_T"] = 0.001
            print("warning: sample_freq not found in dataset")

        self.data = data
        self.len = data["points"].shape[0]
        self.n_frames = data["points"].shape[1]
        self.n_points = data["points"].shape[2]

    def __len__(self):
        return self.len

    def connect_fully(self, num_atoms):
        src, dst = [], []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array(src), np.array(dst)

    def connect_knn(self, x, k):
        edge_index = knn_graph(x, k=k, loop=False)
        return edge_index[0].numpy(), edge_index[1].numpy()

    def __getitem__(self, idx):
        if self.FLAGS.ri_start_at == "zero":
            frame_0 = 0
        else:
            last_pssbl = self.n_frames - self.FLAGS.ri_delta_t
            if "vic" in self.FLAGS.data_str:
                frame_0 = 30
            elif self.split == "train":
                frame_0 = np.random.choice(range(last_pssbl))
            elif self.FLAGS.ri_start_at == "center":
                frame_0 = int((last_pssbl) / 2)
            elif self.FLAGS.ri_start_at == "all":
                frame_0 = int(last_pssbl / self.len * idx)

        frame_T = frame_0 + self.FLAGS.ri_delta_t

        x_0 = torch.tensor(self.data["points"][idx, frame_0].astype(DTYPE))
        x_T = torch.tensor(self.data["points"][idx, frame_T].astype(DTYPE)) - x_0
        v_0 = torch.tensor(self.data["vel"][idx, frame_0].astype(DTYPE))
        v_T = torch.tensor(self.data["vel"][idx, frame_T].astype(DTYPE)) - v_0
        charges = torch.tensor(self.data["charges"][idx].astype(DTYPE))

        # Choose edge construction method
        if self.FLAGS.graph_type == "fully_connected":
            indices_src, indices_dst = self.connect_fully(self.n_points)
        elif self.FLAGS.graph_type == "knn":
            k = getattr(self.FLAGS, "k", None)
            assert k is not None, "FLAGS.k must be provided for knn graph"
            indices_src, indices_dst = self.connect_knn(x_0, k)
        else:
            raise ValueError(f"Unknown graph_type: {self.FLAGS.graph_type}")

        G = dgl.DGLGraph((indices_src, indices_dst))

        G.ndata["x"] = torch.unsqueeze(x_0, dim=1)
        G.ndata["v"] = torch.unsqueeze(v_0, dim=1)
        G.ndata["c"] = torch.unsqueeze(charges, dim=1)
        G.edata["d"] = x_0[indices_dst] - x_0[indices_src]
        G.edata["w"] = charges[indices_dst] * charges[indices_src]
        G.edata["r"] = torch.sqrt(torch.sum(G.edata["d"] ** 2, -1, keepdim=True))

        return G, x_T, v_T


class NBodyDataset:
    """
    NBodyDataset with support for dynamic KNN or fully connected graphs.
    """

    def __init__(
        self,
        partition="train",
        max_samples=1e8,
        dataset_name="nbody_small",
        graph_type="fully_connected",
        k=None,
    ):
        self.partition = partition
        self.graph_type = graph_type
        self.k = k

        if self.partition == "val":
            self.suffix = "valid"
        else:
            self.suffix = self.partition

        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += "_charged5_initvel1"
        elif dataset_name in ["nbody_small", "nbody_small_out_dist"]:
            self.suffix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def load(self):
        dir = pathlib.Path(__file__).parent.absolute()
        loc = np.load(osp.join(dir, "dataset", "loc_" + self.suffix + ".npy"))
        vel = np.load(osp.join(dir, "dataset", "vel_" + self.suffix + ".npy"))
        edges = np.load(osp.join(dir, "dataset", "edges_" + self.suffix + ".npy"))
        charges = np.load(osp.join(dir, "dataset", "charges_" + self.suffix + ".npy"))

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        loc = loc[: self.max_samples]
        vel = vel[: self.max_samples]
        charges = torch.Tensor(charges[: self.max_samples])
        edge_attr = []

        # Default edge construction
        n_nodes = loc.size(2)
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)

        return loc, vel, edge_attr, edges, charges

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def connect_knn(self, positions, k):
        # `positions` is (n_nodes, 3), output edges is [2, num_edges]
        edge_index = knn_graph(positions, k=k, loop=False)
        return edge_index[0], edge_index[1]

    def get_edges(self, batch_size, n_nodes, positions=None):
        if self.graph_type == "knn":
            assert positions is not None, "Must provide node positions for knn graph"
            rows, cols = [], []
            for i in range(batch_size):
                pos_i = positions[i]  # [n_nodes, 3]
                ei_src, ei_dst = self.connect_knn(pos_i, self.k)
                ei_src += i * n_nodes
                ei_dst += i * n_nodes
                rows.append(ei_src)
                cols.append(ei_dst)
            return [torch.cat(rows), torch.cat(cols)]

        elif self.graph_type == "fully_connected":
            edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
            if batch_size == 1:
                return edges
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            return [torch.cat(rows), torch.cat(cols)]

        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]

    def __len__(self):
        return len(self.data[0])
