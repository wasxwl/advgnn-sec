"""
Dataset loaders for social network anomaly detection.

Supports:
- Synthetic dataset generation (for immediate experimentation)
- TwiBot-22 style dataset (Twitter bot detection benchmark)
- Custom edge list + node feature formats
"""

import os
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple

import torch
from torch_geometric.data import Data

from .twibot_loader import TwiBot22StyleDataset


class SyntheticSocialNetwork:
    """Generate a synthetic social network with anomalous users.

    Creates a heterogeneous graph with:
    - Normal users: high clustering, diverse features
    - Anomalous users (bots): star-like patterns, repetitive features
    - Attackers: users who try to mimic normal behavior
    """

    def __init__(
        self,
        n_normal: int = 800,
        n_anomaly: int = 200,
        n_attacker: int = 50,
        feature_dim: int = 16,
        seed: int = 42,
    ):
        self.n_normal = n_normal
        self.n_anomaly = n_anomaly
        self.n_attacker = n_attacker
        self.feature_dim = feature_dim
        self.seed = seed
        self.n_total = n_normal + n_anomaly + n_attacker

    def generate(self) -> Tuple[Data, torch.Tensor, str]:
        """Generate synthetic social network data.

        Returns:
            data: PyG Data object with edge_index, x, y
            labels: node labels (0=normal, 1=anomaly, 2=attacker)
            description: str description of the dataset
        """
        rng = np.random.RandomState(self.seed)

        # === Node features ===
        # Normal users: features from a Gaussian with moderate variance
        x_normal = rng.randn(self.n_normal, self.feature_dim) * 0.8

        # Anomalous users: shifted mean, lower variance (repetitive behavior)
        x_anomaly = rng.randn(self.n_anomaly, self.feature_dim) * 0.3 + 2.0

        # Attacker users: features close to normal but with subtle differences
        x_attacker = rng.randn(self.n_attacker, self.feature_dim) * 0.6 + 0.5

        x = np.vstack([x_normal, x_anomaly, x_attacker])
        x = torch.tensor(x, dtype=torch.float32)

        # === Labels ===
        # 0 = normal, 1 = anomaly, 2 = attacker (treated as anomaly in evaluation)
        y = torch.cat([
            torch.zeros(self.n_normal, dtype=torch.long),
            torch.ones(self.n_anomaly, dtype=torch.long),
            torch.ones(self.n_attacker, dtype=torch.long) * 2,
        ])

        # === Edge construction ===
        edge_list = []

        # Normal user connections: high clustering (clique-like neighborhoods)
        for i in range(self.n_normal):
            # Each normal user connects to 5-15 other normal users
            n_edges = rng.randint(5, 16)
            candidates = list(range(0, self.n_normal))
            candidates.remove(i)
            targets = rng.choice(candidates, size=min(n_edges, len(candidates)), replace=False)
            for t in targets:
                edge_list.append((i, t))
                edge_list.append((t, i))

        # Anomalous users: star-like patterns (follow many, few follow back)
        for i in range(self.n_anomaly):
            node_id = self.n_normal + i
            # Follow many normal users
            n_follow = rng.randint(20, 50)
            targets = rng.choice(self.n_normal, size=min(n_follow, self.n_normal), replace=False)
            for t in targets:
                edge_list.append((node_id, t))
            # Very few reciprocal edges
            n_recip = rng.randint(0, 3)
            recip_targets = rng.choice(self.n_normal, size=min(n_recip, self.n_normal), replace=False)
            for t in recip_targets:
                edge_list.append((t, node_id))

        # Attacker users: try to mimic normal users
        for i in range(self.n_attacker):
            node_id = self.n_normal + self.n_anomaly + i
            # Connect to normal users and other attackers
            n_edges = rng.randint(8, 15)
            targets = rng.choice(self.n_normal, size=min(n_edges, self.n_normal), replace=False)
            for t in targets:
                edge_list.append((node_id, t))
                edge_list.append((t, node_id))
            # Also connect to other attackers (hidden community)
            n_secret = rng.randint(2, 5)
            attacker_candidates = list(range(self.n_normal + self.n_anomaly, self.n_total))
            attacker_candidates.remove(node_id)
            secret_targets = rng.choice(attacker_candidates, size=min(n_secret, len(attacker_candidates)), replace=False)
            for t in secret_targets:
                edge_list.append((node_id, t))
                edge_list.append((t, node_id))

        # Deduplicate edges
        edge_set = set()
        for u, v in edge_list:
            if u != v:
                edge_set.add((min(u, v), max(u, v)))
        edge_list = [(u, v) for u, v in edge_set]
        # Make undirected
        edge_list_doubled = []
        for u, v in edge_list:
            edge_list_doubled.append((u, v))
            edge_list_doubled.append((v, u))

        edge_index = torch.tensor(edge_list_doubled, dtype=torch.long).T

        # Remove any edges with invalid node indices (shouldn't happen, but safety)
        mask = (edge_index[0] < self.n_total) & (edge_index[1] < self.n_total)
        edge_index = edge_index[:, mask]

        data = Data(x=x, edge_index=edge_index, y=y)

        description = (
            f"Synthetic social network: {self.n_total} nodes, "
            f"{edge_index.shape[1]} edges, "
            f"feature_dim={self.feature_dim}\n"
            f"Normal: {self.n_normal}, Anomaly: {self.n_anomaly}, "
            f"Attacker: {self.n_attacker}"
        )

        return data, y, description


class CustomEdgeListLoader:
    """Load a social network from edge list and node feature files.

    Expected format:
    - edges.txt: "src dst" per line (0-indexed)
    - features.npy: numpy array of shape (n_nodes, feature_dim)
    - labels.txt: "node_id label" per line
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self) -> Tuple[Data, torch.Tensor, str]:
        edges_path = os.path.join(self.data_dir, "edges.txt")
        features_path = os.path.join(self.data_dir, "features.npy")
        labels_path = os.path.join(self.data_dir, "labels.txt")

        # Load edges
        edges = np.loadtxt(edges_path, dtype=np.int64)
        if edges.ndim == 1:
            edges = edges.reshape(-1, 2)
        # Make undirected
        edges_rev = edges[:, [1, 0]]
        all_edges = np.vstack([edges, edges_rev])
        edge_index = torch.tensor(all_edges.T, dtype=torch.long)

        # Load features
        x = torch.tensor(np.load(features_path), dtype=torch.float32)

        # Load labels
        label_data = np.loadtxt(labels_path, dtype=np.int64)
        if label_data.ndim == 1:
            y = torch.tensor(label_data, dtype=torch.long)
        else:
            # Format: node_id label
            y = torch.zeros(len(x), dtype=torch.long)
            for row in label_data:
                y[row[0]] = row[1]

        data = Data(x=x, edge_index=edge_index, y=y)
        description = f"Custom dataset from {self.data_dir}: {len(x)} nodes, {edge_index.shape[1]} edges"

        return data, y, description


def get_dataset(
    dataset_name: str = "synthetic",
    data_dir: str = "./data",
    **kwargs,
) -> Tuple[Data, torch.Tensor, str]:
    """Factory function to get datasets.

    Args:
        dataset_name: "synthetic", "twibot22", "twibot22_full", or "custom"
        data_dir: path to data directory
        **kwargs: passed to dataset constructor
    """
    if dataset_name == "synthetic":
        gen = SyntheticSocialNetwork(**kwargs)
        return gen.generate()
    elif dataset_name == "twibot22":
        gen = TwiBot22StyleDataset(**kwargs)
        return gen.generate()
    elif dataset_name == "custom":
        loader = CustomEdgeListLoader(data_dir)
        return loader.load()
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' not implemented. "
            f"Use 'synthetic', 'twibot22', or 'custom'."
        )
