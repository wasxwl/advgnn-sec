"""
TwiBot-22 style dataset generator.

Generates a synthetic dataset that mimics TwiBot-22 properties:
- Twitter-like user profiles (description length, followers/following ratio, tweet frequency)
- Power-law degree distribution
- Bot accounts with characteristic patterns
- Bot accounts that mimic human behavior (advanced bots)

Reference: Feng et al., "TwiBot-22: A New Benchmark for Bot Detection", arXiv 2022.

The real TwiBot-22 dataset has ~450K users and 16.8M edges.
For single-GPU training, we generate a manageable subset with similar properties.
"""

import os
import numpy as np
from typing import Tuple
import torch
from torch_geometric.data import Data


class TwiBot22StyleDataset:
    """Generate TwiBot-22 style bot detection dataset.

    Mimics the real TwiBot-22 dataset properties:
    - 10K-50K nodes with power-law degree distribution
    - 13-dimensional Twitter profile features
    - Human/bot labels with realistic bot behavior patterns
    - Advanced bots that mimic human profiles
    """

    # TwiBot-22 feature descriptions
    FEATURE_NAMES = [
        "profile_complete",    # 0: profile completeness score
        "description_len",     # 1: description length
        "followers_count",     # 2: number of followers
        "friends_count",       # 3: number of friends (following)
        "favourites_count",    # 4: number of favorites
        "listed_count",        # 5: number of lists
        "statuses_count",      # 6: number of tweets
        "account_age_days",    # 7: account age in days
        "tweet_freq",          # 8: tweets per day
        "follower_friend_ratio",# 9: follower/friend ratio
        "has_profile_image",   # 10: has profile image (binary)
        "has_location",        # 11: has location (binary)
        "url_count",           # 12: number of URLs in profile
    ]

    def __init__(
        self,
        n_human: int = 8000,
        n_bot: int = 1500,
        n_advanced_bot: int = 500,
        feature_dim: int = 13,
        edge_density: float = 0.001,
        seed: int = 42,
    ):
        self.n_human = n_human
        self.n_bot = n_bot
        self.n_advanced_bot = n_advanced_bot
        self.feature_dim = min(feature_dim, len(self.FEATURE_NAMES))
        self.edge_density = edge_density
        self.seed = seed
        self.n_total = n_human + n_bot + n_advanced_bot

    def generate(self) -> Tuple[Data, torch.Tensor, str]:
        """Generate TwiBot-22 style dataset.

        Returns:
            data: PyG Data object
            labels: node labels (0=human, 1=bot)
            description: dataset description
        """
        rng = np.random.RandomState(self.seed)

        # === Node features: realistic Twitter profile features ===
        features = np.zeros((self.n_total, self.feature_dim))

        # Human users: diverse profiles
        features[:self.n_human, 0] = np.clip(rng.beta(3, 1, self.n_human), 0, 1)  # profile complete
        features[:self.n_human, 1] = np.clip(rng.exponential(30, self.n_human), 0, 150)  # description len
        features[:self.n_human, 2] = np.log1p(rng.pareto(1.5, self.n_human))  # followers (power law)
        features[:self.n_human, 3] = np.log1p(rng.pareto(1.8, self.n_human))  # friends
        features[:self.n_human, 4] = np.log1p(rng.pareto(1.2, self.n_human))  # favourites
        features[:self.n_human, 5] = np.log1p(rng.exponential(5, self.n_human))  # listed
        features[:self.n_human, 6] = np.log1p(rng.pareto(1.0, self.n_human))  # statuses
        features[:self.n_human, 7] = np.clip(rng.exponential(1000, self.n_human), 0, 5000)  # account age
        features[:self.n_human, 8] = np.clip(rng.gamma(2, 2, self.n_human), 0, 50)  # tweet freq
        features[:self.n_human, 9] = rng.lognormal(0, 1, self.n_human)  # follower/friend ratio
        features[:self.n_human, 10] = rng.binomial(1, 0.85, self.n_human)  # has profile image
        features[:self.n_human, 11] = rng.binomial(1, 0.5, self.n_human)  # has location
        features[:self.n_human, 12] = rng.poisson(1.5, self.n_human)  # url count

        # Basic bots: repetitive, low-activity profiles
        bot_start = self.n_human
        features[bot_start:bot_start + self.n_bot, 0] = np.clip(rng.beta(1, 3, self.n_bot), 0, 1)
        features[bot_start:bot_start + self.n_bot, 1] = np.clip(rng.exponential(5, self.n_bot), 0, 30)
        features[bot_start:bot_start + self.n_bot, 2] = np.log1p(rng.exponential(10, self.n_bot))  # few followers
        features[bot_start:bot_start + self.n_bot, 3] = np.log1p(rng.exponential(5000, self.n_bot))  # follow many
        features[bot_start:bot_start + self.n_bot, 4] = np.log1p(rng.exponential(2, self.n_bot))
        features[bot_start:bot_start + self.n_bot, 5] = 0  # not listed
        features[bot_start:bot_start + self.n_bot, 6] = np.log1p(rng.exponential(100, self.n_bot))
        features[bot_start:bot_start + self.n_bot, 7] = np.clip(rng.exponential(100, self.n_bot), 0, 500)  # new accounts
        features[bot_start:bot_start + self.n_bot, 8] = np.clip(rng.gamma(5, 2, self.n_bot), 0, 100)  # high tweet freq
        features[bot_start:bot_start + self.n_bot, 9] = rng.exponential(0.1, self.n_bot)  # low follower ratio
        features[bot_start:bot_start + self.n_bot, 10] = rng.binomial(1, 0.3, self.n_bot)  # no profile image
        features[bot_start:bot_start + self.n_bot, 11] = rng.binomial(1, 0.1, self.n_bot)  # no location
        features[bot_start:bot_start + self.n_bot, 12] = 0  # no URLs

        # Advanced bots: mimic human profiles but with subtle differences
        adv_start = self.n_human + self.n_bot
        # Copy human-like distributions but with slight shifts
        features[adv_start:adv_start + self.n_advanced_bot, 0] = np.clip(rng.beta(3, 1, self.n_advanced_bot), 0, 1)
        features[adv_start:adv_start + self.n_advanced_bot, 1] = np.clip(rng.exponential(25, self.n_advanced_bot), 0, 120)
        features[adv_start:adv_start + self.n_advanced_bot, 2] = np.log1p(rng.pareto(1.5, self.n_advanced_bot))
        features[adv_start:adv_start + self.n_advanced_bot, 3] = np.log1p(rng.pareto(2.0, self.n_advanced_bot))
        features[adv_start:adv_start + self.n_advanced_bot, 4] = np.log1p(rng.pareto(1.5, self.n_advanced_bot))
        features[adv_start:adv_start + self.n_advanced_bot, 5] = np.log1p(rng.exponential(3, self.n_advanced_bot))
        features[adv_start:adv_start + self.n_advanced_bot, 6] = np.log1p(rng.pareto(1.2, self.n_advanced_bot))
        features[adv_start:adv_start + self.n_advanced_bot, 7] = np.clip(rng.exponential(800, self.n_advanced_bot), 0, 4000)
        features[adv_start:adv_start + self.n_advanced_bot, 8] = np.clip(rng.gamma(3, 3, self.n_advanced_bot), 0, 60)
        features[adv_start:adv_start + self.n_advanced_bot, 9] = rng.lognormal(-0.2, 1.2, self.n_advanced_bot)
        features[adv_start:adv_start + self.n_advanced_bot, 10] = rng.binomial(1, 0.8, self.n_advanced_bot)
        features[adv_start:adv_start + self.n_advanced_bot, 11] = rng.binomial(1, 0.45, self.n_advanced_bot)
        features[adv_start:adv_start + self.n_advanced_bot, 12] = rng.poisson(1.0, self.n_advanced_bot)

        # Normalize features to [0, 1] range for GNN
        for i in range(self.feature_dim):
            col = features[:, i]
            min_val, max_val = col.min(), col.max()
            if max_val > min_val:
                features[:, i] = (col - min_val) / (max_val - min_val)

        # Add small Gaussian noise
        features += rng.randn(*features.shape) * 0.02

        x = torch.tensor(features[:, :self.feature_dim], dtype=torch.float32)

        # === Labels ===
        # 0 = human, 1 = bot (both basic and advanced)
        y = torch.cat([
            torch.zeros(self.n_human, dtype=torch.long),
            torch.ones(self.n_bot, dtype=torch.long),
            torch.ones(self.n_advanced_bot, dtype=torch.long),
        ])

        # === Edge construction: realistic Twitter-like graph ===
        # Power-law degree distribution for human users
        # Use Barabasi-Albert-like preferential attachment
        edge_set = set()

        # Human user connections: preferential attachment
        # Each new user connects to existing users with probability proportional to degree
        degrees = np.ones(self.n_human)
        for i in range(1, self.n_human):
            # Connect to m existing users
            m = min(rng.randint(3, 10), i)
            probs = degrees[:i] / degrees[:i].sum()
            targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
            for t in targets:
                edge_set.add((min(i, t), max(i, t)))
                degrees[i] += 1
                degrees[t] += 1

        # Bot connections: bots follow many humans, few reciprocal
        for i in range(self.n_bot):
            bot_id = self.n_human + i
            # Basic bots follow many random humans
            n_follow = rng.randint(20, 100)
            targets = rng.choice(self.n_human, size=min(n_follow, self.n_human), replace=False)
            for t in targets:
                edge_set.add((min(bot_id, t), max(bot_id, t)))

        # Advanced bots: selective following, more human-like
        for i in range(self.n_advanced_bot):
            bot_id = self.n_human + self.n_bot + i
            n_follow = rng.randint(10, 50)
            # Prefer humans with higher degree (follow popular accounts)
            probs = degrees / degrees.sum()
            targets = rng.choice(self.n_human, size=min(n_follow, self.n_human), replace=False, p=probs)
            for t in targets:
                edge_set.add((min(bot_id, t), max(bot_id, t)))

        # Bot-to-bot connections (bot networks)
        # Basic bots: sparse connections to other bots
        for i in range(self.n_bot):
            bot_id = self.n_human + i
            n_bot_edges = rng.randint(0, 5)
            other_bots = list(range(self.n_human, self.n_human + self.n_bot))
            other_bots.remove(bot_id)
            targets = rng.choice(other_bots, size=min(n_bot_edges, len(other_bots)), replace=False)
            for t in targets:
                edge_set.add((min(bot_id, t), max(bot_id, t)))

        # Advanced bots: more connections to other advanced bots
        for i in range(self.n_advanced_bot):
            bot_id = self.n_human + self.n_bot + i
            n_bot_edges = rng.randint(3, 10)
            other_adv = list(range(self.n_human + self.n_bot, self.n_total))
            other_adv.remove(bot_id)
            targets = rng.choice(other_adv, size=min(n_bot_edges, len(other_adv)), replace=False)
            for t in targets:
                edge_set.add((min(bot_id, t), max(bot_id, t)))

        # Convert to edge_index (bidirectional)
        edge_list = []
        for u, v in edge_set:
            if u != v and u < self.n_total and v < self.n_total:
                edge_list.append((u, v))
                edge_list.append((v, u))

        edge_index = torch.tensor(edge_list, dtype=torch.long).T

        data = Data(x=x, edge_index=edge_index, y=y)

        description = (
            f"TwiBot-22 style dataset: {self.n_total} nodes, "
            f"{edge_index.shape[1]} edges, "
            f"feature_dim={self.feature_dim}\n"
            f"Human: {self.n_human}, Bot: {self.n_bot}, "
            f"Advanced Bot: {self.n_advanced_bot}\n"
            f"Features: {', '.join(self.FEATURE_NAMES[:self.feature_dim])}"
        )

        return data, y, description


def generate_twibot22_subset(seed=42):
    """Generate a smaller TwiBot-22 style subset for quick testing."""
    gen = TwiBot22StyleDataset(
        n_human=2000,
        n_bot=400,
        n_advanced_bot=100,
        feature_dim=13,
        seed=seed,
    )
    return gen.generate()


def generate_twibot22_full(seed=42):
    """Generate a full-scale TwiBot-22 style dataset."""
    gen = TwiBot22StyleDataset(
        n_human=8000,
        n_bot=1500,
        n_advanced_bot=500,
        feature_dim=13,
        seed=seed,
    )
    return gen.generate()
