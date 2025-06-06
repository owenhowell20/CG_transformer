# import torch
# from torch_geometric.data import Data
# from torch_geometric.nn import radius_graph
# from src.graph_layers import (
#     SE3GNN,
#     GraphConvLayer,
#     AdaptiveSpectralConvLayer,
#     HyenaGraphConv,
#     PyG_GraphFourierTransform,
#     PEGLayer,
# )
# from fixtures import (
#     mock_graph_batch,
#     mock_qm9_dataloader,
#     mock_big_graph_batch,
#     permute_batch,
# )
# from torch_geometric.utils import to_dense_adj
#
#
# def test_PyG_GraphFourierTransform(mock_big_graph_batch):
#     rank = 3
#     U = PyG_GraphFourierTransform(mock_big_graph_batch, rank=rank)
#
#     # The number of nodes in both graphs is 7 + 10 = 17
#     # So the expected output shape should be [17, 3]
#     assert U.shape[1] == rank, f"Expected rank to be {rank}, but got {U.shape[1]}"
#     assert U.shape[0] == 17, f"Expected number of nodes to be 17, but got {U.shape[0]}"
#
#
# def test_PEGLayer(mock_big_graph_batch):
#
#     model = PEGLayer(node_input_dimension=11, node_output_dimension=16)
#
#     out = model(mock_graph_batch)
#     # Assert the number of nodes in the graph matches the first dimension of the output
#     assert out.shape[0] == mock_graph_batch.num_nodes, "Number of nodes incorrect"
#     assert out.shape[1] == 16, "Output channels incorrect"
#
#
# #
# #
# # def test_HyenaGraphConv(mock_graph_batch):
# #
# #     ### get adj matrix
# #     adj = to_dense_adj(mock_graph_batch.edge_index, batch=mock_graph_batch.batch)
# #
# #     ### process edge features
# #     hidden_channels = 20
# #     conv_model = GraphConvLayer(node_dimension=11, edge_dimension=4, out_channels=hidden_channels)
# #     out = conv_model(mock_graph_batch)
# #
# #     # Assert the number of nodes in the graph matches the first dimension of the output
# #     assert out.shape[0] == mock_graph_batch.num_nodes, "Number of nodes incorrect"
# #     assert out.shape[1] == hidden_channels, "Output channels incorrect"
# #
# #     graph_hyena  = HyenaGraphConv(node_dimension=hidden_channels, order=3, low_rank=1)
# #     out = graph_hyena ( out , adj, batch=mock_graph_batch.batch )
# #
# #     ### [b,N,d]
# #     assert out.shape[2] == hidden_channels, "output dim wrong"
# #
# #
# #
# # def test_permute_batch(mock_big_graph_batch):
# #     adj = to_dense_adj(mock_big_graph_batch.edge_index, batch=mock_big_graph_batch.batch)
# #
# #     perm_mock_big_graph_batch, per_graph_perms = permute_batch(mock_big_graph_batch)
# #
# #     perm_adj = to_dense_adj(perm_mock_big_graph_batch.edge_index, batch=perm_mock_big_graph_batch.batch)
# #
# #     node_features = mock_big_graph_batch.x
# #     perm_node_features = perm_mock_big_graph_batch.x
# #
# #     ### check node features
# #     assert node_features.shape == perm_node_features.shape
# #
# #     batch = mock_big_graph_batch.batch
# #     perm_batch = perm_mock_big_graph_batch.batch
# #     num_graphs = batch.max().item() + 1
# #
# #     offset = 0  # keeps track of node positions across graphs
# #     for g in range(num_graphs):
# #         idx = (batch == g).nonzero(as_tuple=True)[0]
# #         num_nodes = idx.size(0)
# #
# #         perm = per_graph_perms[g]
# #         inv_perm = torch.argsort(perm)
# #
# #         # Compare node features
# #         orig_feat = node_features[idx]
# #         perm_feat = perm_node_features[offset:offset + num_nodes]
# #
# #         assert torch.allclose(orig_feat, perm_feat[inv_perm, :], atol=1e-5)
# #
# #         ### check adjacency matrices
# #         adj_g = adj[g][:num_nodes, :num_nodes]
# #         perm_adj_g = perm_adj[g][:num_nodes, :num_nodes]
# #
# #         # Apply the local permutation for row and column reordering
# #         permuted_adj_g = perm_adj_g[inv_perm][:, inv_perm]
# #
# #         print(f"Graph {g}:")
# #         print(f"Original adj_g:\n{adj_g}")
# #         print(f"Permuted perm_adj_g:\n{perm_adj_g}")
# #         print(f"Inverse perm:\n{inv_perm}")
# #
# #         assert torch.allclose(
# #             adj_g,
# #             permuted_adj_g,
# #             atol=1e-5
# #         )
# #
# #         offset += num_nodes  # update offset for next graph
# #
# # def test_permutation_SpectralConvLayer(mock_big_graph_batch):
# #     hidden_channels = 20
# #
# #     # Get adjacency matrix
# #     adj = to_dense_adj(mock_big_graph_batch.edge_index, batch=mock_big_graph_batch.batch)
# #
# #     conv_model = GraphConvLayer(node_dimension=11, edge_dimension=4, out_channels=hidden_channels)
# #     out = conv_model(mock_big_graph_batch)
# #
# #     spectral_layer = AdaptiveSpectralConvLayer(node_dimension=hidden_channels, output_dimension=5, low_rank=3)
# #     out = spectral_layer(out, adj, batch=mock_big_graph_batch.batch)
# #
# #     assert out.shape[2] == 5, "output dim wrong"
# #
# #     # Permute batch correctly
# #     perm_mock_graph_batch, per_graph_perms = permute_batch(mock_big_graph_batch)
# #
# #     # Get permuted adjacency
# #     perm_adj = to_dense_adj(perm_mock_graph_batch.edge_index, batch=perm_mock_graph_batch.batch)
# #
# #     perm_out = conv_model(perm_mock_graph_batch)
# #     perm_out = spectral_layer(perm_out, perm_adj, batch=perm_mock_graph_batch.batch)
# #
# #     assert out.shape == perm_out.shape, "wrong shapes"
# #
# #
# #
# #
# # def test_SpectralConvLayer(mock_big_graph_batch):
# #
# #     ### get adj matrix
# #     adj = to_dense_adj(mock_big_graph_batch.edge_index, batch=mock_big_graph_batch.batch)
# #
# #     hidden_channels = 20
# #     conv_model = GraphConvLayer(node_dimension=11, edge_dimension=4, out_channels=hidden_channels)
# #     out = conv_model(mock_big_graph_batch)
# #
# #     # Assert the number of nodes in the graph matches the first dimension of the output
# #     assert out.shape[0] == mock_big_graph_batch.num_nodes, "Number of nodes incorrect"
# #     assert out.shape[1] == hidden_channels, "Output channels incorrect"
# #
# #     spectral_layer  =  AdaptiveSpectralConvLayer(node_dimension=hidden_channels, output_dimension=5, low_rank=3)
# #     out = spectral_layer( out , adj, batch=mock_big_graph_batch.batch )
# #
# #     ### [b,N,d]
# #     assert out.shape[2] == 5, "output dim wrong"
# #
# #
# #
# #
# # # # def test_graph_conv(mock_graph_batch):
# # # #     model = GraphConvLayer(node_dimension=11, edge_dimension=4, out_channels=4)
# # # #     out = model(mock_graph_batch)
# # # #     # Assert the number of nodes in the graph matches the first dimension of the output
# # # #     assert out.shape[0] == mock_graph_batch.num_nodes, "Number of nodes incorrect"
# # # #     assert out.shape[1] == 4, "Output channels incorrect"
# # # #
# # # #
# # # # def test_graph_conv_batched(mock_qm9_dataloader):
# # # #     model = GraphConvLayer(node_dimension=11, edge_dimension=4, out_channels=4)
# # # #
# # # #     for G in mock_qm9_dataloader:
# # # #         out = model(G)
# # # #
# # # #         ### Assert the number of nodes in the graph matches the first dimension of the output
# # # #         assert out.shape[0] == G.num_nodes, "Number of nodes incorrect"
# # # #         assert out.shape[1] == 4, "Output channels incorrect"
