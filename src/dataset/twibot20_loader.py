"""
TwiBot-20 dataset loader for bot detection.

TwiBot-20: A Twitter bot detection benchmark with ~12K users,
multi-relational graph, and text-based features.

Reference: Yang et al., "TwiBot-20: A Comprehensive Benchmark
for Twitter Bot Detection", KDD 2021.
"""

import os
import re
import json
import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple
from datetime import datetime


class TwiBot20Dataset:
    """Load TwiBot-20 bot detection dataset.

    Extracts numerical profile features from text data.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def _extract_features(self, user_texts: list) -> np.ndarray:
        """Extract numerical features from TwiBot-20 text entries."""
        n = len(user_texts)
        features = np.zeros((n, 13))

        for i, text in enumerate(user_texts):
            # Split by </s> and parse metadata
            parts = text.split("</s>")
            if len(parts) < 5:
                continue

            # First part starts with "METADATA: "
            meta_raw = parts[0].replace("METADATA:", "").strip()
            # Remaining metadata fields from parts[1] onwards
            meta_fields = [meta_raw] + [p.strip() for p in parts[1:]]

            # Find where DESCRIPTION or TWEET starts
            desc_pos = None
            for pi, p in enumerate(meta_fields):
                if "DESCRIPTION:" in p or "TWEET:" in p:
                    desc_pos = pi
                    break

            if desc_pos is not None and desc_pos >= 9:
                # Numerical fields: indices 4-7 (favourites, followers, friends, statuses)
                try:
                    features[i, 0] = np.log1p(float(meta_fields[4]))  # favourites
                except:
                    pass
                try:
                    features[i, 1] = np.log1p(float(meta_fields[5]))  # followers
                except:
                    pass
                try:
                    features[i, 2] = np.log1p(float(meta_fields[6]))  # friends
                except:
                    pass
                try:
                    features[i, 3] = np.log1p(float(meta_fields[7]))  # statuses
                except:
                    pass

                # Verified (index 3)
                features[i, 10] = 1.0 if meta_fields[3].lower() == "true" else 0.0

                # Location (index 1) - check if non-empty
                features[i, 11] = 1.0 if meta_fields[1].strip() else 0.0

                # Parse date (index 0)
                try:
                    dt = datetime.strptime(meta_fields[0], "%a %b %d %H:%M:%S %Y")
                    ref = datetime(2021, 1, 1)
                    age = (ref - dt).days
                    features[i, 5] = np.log1p(max(age, 0))
                except:
                    pass

                # Parse description text
                desc_full = meta_fields[desc_pos]
                if "DESCRIPTION:" in desc_full:
                    desc_part = desc_full.replace("DESCRIPTION:", "").strip()
                    if "TWEET:" in desc_part:
                        desc_part = desc_part.split("TWEET:")[0].strip()
                    features[i, 6] = len(desc_part)  # description length
                else:
                    features[i, 6] = 0

                # Parse tweets after DESCRIPTION
                # Collect all parts from desc_pos onwards for tweet analysis
                tweet_text = ""
                for p in meta_fields[desc_pos:]:
                    tweet_text += " " + p
                if "TWEET:" in tweet_text:
                    tweet_text = tweet_text.split("TWEET:")[1].strip()
                    tweets = tweet_text.split("</s>")
                    valid_tweets = [t.strip() for t in tweets if len(t.strip()) > 0]
                    features[i, 7] = len(valid_tweets)  # tweet count
                    if valid_tweets:
                        features[i, 8] = np.mean([len(t) for t in valid_tweets])  # avg length
                    features[i, 12] = float(tweet_text.count("HTTPURL"))  # URL count
                else:
                    features[i, 7] = 0
                    features[i, 12] = float(text.count("HTTPURL"))

                # Has profile image: check for SHAQ/True pattern after numerical fields
                # It's at index 8 in the metadata
                features[i, 9] = 1.0 if meta_fields[8].lower() == "true" else 0.0
            else:
                # Fallback: try to extract what we can
                try:
                    features[i, 1] = np.log1p(float(meta_fields[1]))
                except:
                    pass
                try:
                    features[i, 2] = np.log1p(float(meta_fields[2]))
                except:
                    pass

        # Normalize features to [0, 1]
        for j in range(features.shape[1]):
            col = features[:, j]
            min_val = col.min()
            max_val = col.max()
            if max_val > min_val:
                features[:, j] = (col - min_val) / (max_val - min_val)
            else:
                features[:, j] = 0.0

        # Add small noise for numerical stability
        features += np.random.randn(*features.shape) * 0.01

        return features

    def load(self) -> Tuple[Data, torch.Tensor, str]:
        """Load TwiBot-20 dataset.

        Returns:
            data: PyG Data object with x, edge_index, y
            labels: node labels (0=human, 1=bot)
            description: str description of the dataset
        """
        # Load edge data
        edge_index = torch.load(
            os.path.join(self.data_dir, "edge_index.pt"),
            weights_only=False,
        )
        edge_type = torch.load(
            os.path.join(self.data_dir, "edge_type.pt"),
            weights_only=False,
        )

        # Load labels (one-hot -> scalar)
        labels_onehot = torch.load(
            os.path.join(self.data_dir, "labels.pt"),
            weights_only=False,
        )
        # labels are [human, bot] one-hot -> take argmax
        labels = labels_onehot.argmax(dim=1).long()

        # Load splits
        train_idx = torch.load(
            os.path.join(self.data_dir, "train_idx.pt"),
            weights_only=False,
        )
        valid_idx = torch.load(
            os.path.join(self.data_dir, "valid_idx.pt"),
            weights_only=False,
        )
        test_idx = torch.load(
            os.path.join(self.data_dir, "test_idx.pt"),
            weights_only=False,
        )

        # Load and extract features
        with open(os.path.join(self.data_dir, "norm_user_text.json"), "r") as f:
            user_texts = json.load(f)

        np.random.seed(42)
        features = self._extract_features(user_texts)
        x = torch.tensor(features, dtype=torch.float32)

        # Build PyG Data object
        data = Data(x=x, edge_index=edge_index, y=labels)
        data.edge_type = edge_type
        data.train_idx = train_idx
        data.valid_idx = valid_idx
        data.test_idx = test_idx

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        bot_rate = labels.sum().item() / num_nodes

        description = (
            f"TwiBot-20 dataset: {num_nodes} nodes, "
            f"{num_edges} edges, feature_dim={x.shape[1]}\n"
            f"Bot rate: {bot_rate:.2%}, "
            f"Train/Valid/Test: {len(train_idx)}/{len(valid_idx)}/{len(test_idx)}"
        )

        return data, labels, description
