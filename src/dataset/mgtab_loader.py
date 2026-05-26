"""
MGTAB dataset loader for bot detection.

MGTAB: A Multi-Relational Graph-Based Twitter Bot Detection Benchmark.
Contains 10,199 expert-annotated users with 7 relationship types
(followers, friends, mention, reply, quoted, URL, hashtag) and
788-dimensional node features.

Labels: bot (human/bot) and stance (neutral/against/support).
We use the bot detection task for adversarial robustness evaluation.

Reference: "MGTAB: A Multi-Relational Graph-Based Twitter Bot Detection Benchmark"
"""

import os
import torch
from torch_geometric.data import Data
from typing import Tuple


class MGTABDataset:
    """Load MGTAB bot detection dataset.

    Args:
        data_dir: path to directory containing MGTAB .pt files
        use_bot_label: if True, use bot labels (human/bot); else use stance labels
        relation_types: list of edge type indices to include (default: all 7)
    """

    # Edge type descriptions from MGTAB paper
    EDGE_TYPES = [
        "followers",     # 0
        "friends",       # 1
        "mention",       # 2
        "reply",         # 3
        "quoted",        # 4
        "url",           # 5
        "hashtag",       # 6
    ]

    def __init__(
        self,
        data_dir: str,
        use_bot_label: bool = True,
        relation_types: list = None,
    ):
        self.data_dir = data_dir
        self.use_bot_label = use_bot_label
        self.relation_types = relation_types  # None = all

    def load(self) -> Tuple[Data, torch.Tensor, str]:
        """Load MGTAB dataset.

        Returns:
            data: PyG Data object with x, edge_index, y, and optional edge_type
            labels: node labels (0=human, 1=bot)
            description: str description of the dataset
        """
        features = torch.load(
            os.path.join(self.data_dir, "features.pt"),
            weights_only=True,
        )
        edge_index = torch.load(
            os.path.join(self.data_dir, "edge_index.pt"),
            weights_only=True,
        )
        edge_type = torch.load(
            os.path.join(self.data_dir, "edge_type.pt"),
            weights_only=True,
        )
        edge_weight = torch.load(
            os.path.join(self.data_dir, "edge_weight.pt"),
            weights_only=True,
        )

        if self.use_bot_label:
            labels = torch.load(
                os.path.join(self.data_dir, "labels_bot.pt"),
                weights_only=True,
            )
        else:
            labels = torch.load(
                os.path.join(self.data_dir, "labels_stance.pt"),
                weights_only=True,
            )

        # Filter edges by relation type if specified
        if self.relation_types is not None:
            mask = torch.isin(edge_type, torch.tensor(self.relation_types))
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]
            edge_weight = edge_weight[mask]

        # Ensure features are float32
        features = features.to(torch.float32)

        data = Data(
            x=features,
            edge_index=edge_index,
            y=labels,
        )
        data.edge_type = edge_type
        data.edge_weight = edge_weight

        num_nodes = features.shape[0]
        num_edges = edge_index.shape[1] // 2  # undirected
        bot_ratio = labels.sum().item() / num_nodes

        relation_str = "all" if self.relation_types is None else str(self.relation_types)
        label_type = "bot" if self.use_bot_label else "stance"

        description = (
            f"MGTAB dataset ({label_type} labels): {num_nodes} nodes, "
            f"{num_edges} edges, "
            f"feature_dim={features.shape[1]}\n"
            f"Bot rate: {bot_ratio:.2%}, "
            f"Relation types: {relation_str}"
        )

        return data, labels, description


def load_mgtab(
    data_dir: str,
    use_bot_label: bool = True,
    relation_types: list = None,
    seed: int = 42,
) -> Tuple[Data, torch.Tensor, str]:
    """Factory function to load MGTAB dataset."""
    dataset = MGTABDataset(data_dir, use_bot_label, relation_types)
    return dataset.load()
