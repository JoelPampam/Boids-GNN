"""
A small multi-relational GNN for next-step velocity prediction.

Rather than collapsing separation/cohesion/alignment into one generic
graph, this keeps them as three distinct message-passing channels (one
GCNConv per relation, per layer) and sums their contributions -- so the
model can, in principle, learn different behavior for "boid I'm too close
to" vs "boid I'm flocking toward" vs "boid I'm matching velocity with".
This is a simplified relational GCN (similar in spirit to R-GCN).
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MultiRelGCNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_sep = GCNConv(in_channels, out_channels)
        self.conv_coh = GCNConv(in_channels, out_channels)
        self.conv_ali = GCNConv(in_channels, out_channels)
        self.self_loop = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index_sep, edge_index_coh, edge_index_ali):
        out = self.self_loop(x)
        if edge_index_sep.numel() > 0:
            out = out + self.conv_sep(x, edge_index_sep)
        if edge_index_coh.numel() > 0:
            out = out + self.conv_coh(x, edge_index_coh)
        if edge_index_ali.numel() > 0:
            out = out + self.conv_ali(x, edge_index_ali)
        return out


class BoidsGNN(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=32, out_channels=2):
        super().__init__()
        self.layer1 = MultiRelGCNLayer(in_channels, hidden_channels)
        self.layer2 = MultiRelGCNLayer(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        args = (data.edge_index_sep, data.edge_index_coh, data.edge_index_ali)
        x = F.relu(self.layer1(x, *args))
        x = F.relu(self.layer2(x, *args))
        return self.out(x)  # predicted [x_vel, y_vel], normalized