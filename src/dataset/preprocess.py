"""
Data preprocessing utilities for social network anomaly detection.
"""

import torch
import numpy as np
from torch_geometric.transforms import NormalizeFeatures, ToUndirected


def normalize_features(data):
    """Normalize node features to zero mean, unit variance."""
    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True) + 1e-8
    data.x = (data.x - mean) / std
    return data


def compute_graph_statistics(data):
    """Compute basic graph statistics."""
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2  # undirected
    degree = torch.bincount(data.edge_index[0], minlength=num_nodes).float()
    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": degree.mean().item(),
        "max_degree": degree.max().item(),
        "min_degree": degree.min().item(),
        "std_degree": degree.std().item(),
    }
    return stats


def create_train_val_test_split(
    data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
):
    """Create train/val/test split.

    When stratify=True, ensures each split has proportional
    representation of each class.
    """
    rng = np.random.RandomState(seed)
    n = data.num_nodes

    if stratify:
        train_idx, val_idx, test_idx = [], [], []
        for label in data.y.unique():
            mask = (data.y == label).nonzero(as_tuple=True)[0].cpu().numpy()
            rng.shuffle(mask)
            n_total = len(mask)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            train_idx.extend(mask[:n_train])
            val_idx.extend(mask[n_train : n_train + n_val])
            test_idx.extend(mask[n_train + n_val :])
    else:
        indices = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train : n_train + n_val].tolist()
        test_idx = indices[n_train + n_val :].tolist()

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )
