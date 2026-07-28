"""Baseline models: MLP, GCN, GraphSAGE, GAT, GPS.

Unlike the earlier stages of this project (which fixed every baseline at
2 layers), every model here supports a configurable num_layers, since
depth is part of this benchmark's hyperparameter search.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GPSConv


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x, edge_index=None, **kwargs):
        for i, lin in enumerate(self.lins):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, heads=8):
        super().__init__()
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, **kwargs):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x


class GPS(nn.Module):
    """GraphGPS (Rampasek et al., 2022): local MPNN + global transformer
    attention per layer, via PyG's GPSConv wrapping a GCNConv as the local
    message-passing operator (GPSConv's local conv only needs to accept
    (x, edge_index); GCNConv fits without requiring edge features, which
    none of this benchmark's datasets have).

    Node features are augmented with a fixed positional encoding (RWPE,
    see aria_augmentations.compute_rwpe -- the same construction ARIA-PH/
    ARIA-PHG use) before the input projection, since global attention has
    no innate notion of node position without one.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                 heads=4, pe_dim=8):
        super().__init__()
        self.pe_dim = pe_dim
        self.input_proj = nn.Linear(in_dim + pe_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GPSConv(hidden_dim, GCNConv(hidden_dim, hidden_dim), heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, pe=None, **kwargs):
        if pe is None:
            pe = torch.zeros(x.size(0), self.pe_dim, device=x.device)
        h = torch.cat([x, pe], dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.input_proj(h))
        for conv in self.convs:
            h = conv(h, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out_lin(h)


def build_baseline(name, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
                    heads=8, pe_dim=8):
    name = name.lower()
    if name == "mlp":
        return MLP(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name == "gcn":
        return GCN(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name in ("graphsage", "sage"):
        return GraphSAGE(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name == "gat":
        return GAT(in_dim, hidden_dim, out_dim, num_layers, dropout, heads)
    if name == "gps":
        return GPS(in_dim, hidden_dim, out_dim, num_layers, dropout, heads, pe_dim)
    raise ValueError(f"Unknown baseline '{name}'")


BASELINE_NAMES = ["mlp", "gcn", "graphsage", "gat", "gps"]
