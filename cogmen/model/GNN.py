import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        # print(f'GNN: g_dim={g_dim}, h1_dim={h1_dim}, h2_dim={h2_dim}')

        super(GNN, self).__init__()
        # modify num_relations to include self-loop and global edge
        self.num_relations = 2 * args.n_speakers ** 2 + 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=args.gnn_nheads, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * args.gnn_nheads)

    def forward(self, node_features, edge_index, edge_type):
        # print(f'GNN: node_features={node_features.shape}, edge_index={edge_index.shape}, edge_type={edge_type.shape}')
        x = self.conv1(node_features, edge_index, edge_type)
        # print(f'GNN: conv1={x.shape}')
        # print(f'first 10 edges: {edge_index[:, :10]}')
        # print(f'last 10 edges: {edge_index[:, -10:]}')
        x = self.conv2(x, edge_index)
        # print(f'GNN: conv2={x.shape}')
        x = self.bn(x)
        # print(f'GNN: bn={x.shape}')
        x = nn.functional.leaky_relu(x)
        # print(f'GNN: leaky_relu={x.shape}')

        return x
