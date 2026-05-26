"""
MGTAB-compatible GNN models matching the official MGTAB paper implementation.

Architectures match exactly: /tmp/MGTAB/models.py
- GCN: Linear+LeakyReLU -> GCNConv -> Dropout -> GCNConv -> Linear+LeakyReLU -> Linear
- GAT: Linear+LeakyReLU -> GATConv(8 heads) -> Dropout -> GATConv -> Linear+LeakyReLU -> Linear
- SAGE: Linear+LeakyReLU -> SAGEConv -> Dropout -> SAGEConv -> Linear+LeakyReLU -> Linear
- RGCN: Linear+LeakyReLU -> RGCNConv -> Dropout -> RGCNConv -> Linear+LeakyReLU -> Linear

All use kaiming_uniform init, dropout=0.3, hidden=256, lr=1e-3 (AdamW).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv


def init_weights(m):
    """Kaiming uniform initialization matching MGTAB paper."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class MGTAB_GCN(nn.Module):
    """GCN model matching MGTAB paper architecture.

    Architecture: Linear(emb->256)+LeakyReLU -> GCNConv(256,256) -> Dropout -> GCNConv(256,256) -> Linear(256,256)+LeakyReLU -> Linear(256,out)
    """

    def __init__(self, embedding_dim, hidden_dim=256, out_dim=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.linear_relu_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.linear_relu_input(x)
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.linear_relu_output(x)
        x = self.linear_output(x)
        return x


class MGTAB_GAT(nn.Module):
    """GAT model matching MGTAB paper architecture."""

    def __init__(self, embedding_dim, hidden_dim=256, out_dim=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU()
        )
        # 8 heads of size hidden_dim/8 = 32, concatenated back to hidden_dim
        self.gat1 = GATConv(hidden_dim, hidden_dim // 8, heads=8)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.linear_relu_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output(x)
        x = self.linear_output(x)
        return x


class MGTAB_GraphSAGE(nn.Module):
    """GraphSAGE model matching MGTAB paper architecture."""

    def __init__(self, embedding_dim, hidden_dim=256, out_dim=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.linear_relu_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.linear_relu_input(x)
        x = self.sage1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        x = self.linear_relu_output(x)
        x = self.linear_output(x)
        return x


class MGTAB_RGCN(nn.Module):
    """RGCN model matching MGTAB paper architecture.

    Uses RGCNConv for relation-aware message passing.
    """

    def __init__(self, embedding_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.linear_relu_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        x = self.linear_relu_input(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.linear_relu_output(x)
        x = self.linear_output(x)
        return x


class MGTAB_MLP(nn.Module):
    """Feature-only MLP baseline. No graph structure used.

    Same projection layers as GNN models to isolate the feature-only signal:
    Linear(emb->256)+LeakyReLU -> Linear(256,256)+LeakyReLU -> Linear(256,2).
    """

    def __init__(self, embedding_dim, hidden_dim=256, out_dim=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.linear_relu_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index=None):
        x = self.linear_relu_input(x)
        x = self.fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_relu_output(x)
        x = self.linear_output(x)
        return x


class MGTAB_RelationGCN(nn.Module):
    """Relation-aware GCN using RGCNConv for multi-relational message passing.

    Same architecture as MGTAB_GCN but uses RGCNConv instead of GCNConv,
    so it leverages the 7 relation types in MGTAB.
    """

    def __init__(self, embedding_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.linear_relu_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type=None):
        x = self.linear_relu_input(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = self.linear_relu_output(x)
        x = self.linear_output(x)
        return x


class RelationGATConv(nn.Module):
    """Relation-aware GAT convolution: separate GATConv per relation type.

    Applies independent GAT attention for each relation, then sums outputs.
    """

    def __init__(self, in_channels, out_channels, num_relations=7, heads=8, dropout=0.3):
        super().__init__()
        self.num_relations = num_relations
        self.heads = heads
        # One GATConv per relation type
        self.relation_convs = nn.ModuleList([
            GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout)
            for _ in range(num_relations)
        ])

    def forward(self, x, edge_index, edge_type):
        # Sum contributions from each relation
        out = None
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            rel_edge = edge_index[:, mask]
            rel_out = self.relation_convs[r](x, rel_edge)
            if out is None:
                out = rel_out
            else:
                out = out + rel_out
        return out


class MGTAB_RelationGAT(nn.Module):
    """Relation-aware GAT using separate attention per relation type.

    Same architecture as MGTAB_GAT but relation-aware.
    """

    def __init__(self, embedding_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.rgat1 = RelationGATConv(hidden_dim, hidden_dim, num_relations, heads=8, dropout=dropout)
        # Second layer: standard GATConv (relation split for RGATConv is expensive)
        # Use a single GATConv on the merged graph for layer 2, since the first layer
        # already separated relations
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        self.linear_relu_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type=None):
        x = self.linear_relu_input(x)
        x = self.rgat1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output(x)
        x = self.linear_output(x)
        return x
