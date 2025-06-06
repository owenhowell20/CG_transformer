from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.loader import DataLoader
from modelnet_se3hyena import ModelNetSE3Hyena
from modelnet_dgcnn import ModelNetDGCNN, ModelNetDGCNNLoss

# Download datasets
train_dataset_40 = ModelNet(
    root="data/ModelNet40", name="40", train=True, transform=SamplePoints(1024)
)
test_dataset_40 = ModelNet(
    root="data/ModelNet40", name="40", train=False, transform=SamplePoints(1024)
)

train_dataset_10 = ModelNet(
    root="data/ModelNet10", name="10", train=True, transform=SamplePoints(1024)
)
test_dataset_10 = ModelNet(
    root="data/ModelNet10", name="10", train=False, transform=SamplePoints(1024)
)

# Example of model initialization
# For ModelNet40
se3hyena_model_40 = ModelNetSE3Hyena(num_classes=40)
dgcnn_model_40 = ModelNetDGCNN(num_classes=40)

# For ModelNet10
se3hyena_model_10 = ModelNetSE3Hyena(num_classes=10)
dgcnn_model_10 = ModelNetDGCNN(num_classes=10)

# Example of data loaders
batch_size = 32
train_loader_40 = DataLoader(train_dataset_40, batch_size=batch_size, shuffle=True)
test_loader_40 = DataLoader(test_dataset_40, batch_size=batch_size)

train_loader_10 = DataLoader(train_dataset_10, batch_size=batch_size, shuffle=True)
test_loader_10 = DataLoader(test_dataset_10, batch_size=batch_size)

### torch geometric --> PyG graph

### dgl --> SE(3) transformer
