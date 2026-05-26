"""
Adversarial defense mechanisms for GNN-based anomaly detectors.

Defense strategies:
1. AdversarialTraining: train the detector with adversarial examples
2. FeatureSmoothing: apply feature smoothing to reduce feature attack impact
3. EdgeRegularization: penalize suspicious edge patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .attack import MetaGradGraphAttack, FeatureAttack


class AdversarialTraining:
    """Adversarial training defense for GNN anomaly detectors.

    During training, injects adversarial examples to make the model
    robust against both graph and feature perturbations.
    """

    def __init__(
        self,
        model: nn.Module,
        attack_budget: float = 0.05,
        feature_bound: float = 1.0,
        attack_freq: int = 3,  # perform attack every N epochs
        alpha: float = 0.5,    # weight of adversarial loss
    ):
        self.model = model
        self.graph_attack = MetaGradGraphAttack(model, budget_ratio=attack_budget)
        self.feature_attack = FeatureAttack(model, perturbation_bound=feature_bound, n_steps=30)
        self.attack_freq = attack_freq
        self.alpha = alpha

    def train_step(
        self,
        data,
        train_mask,
        optimizer,
        epoch,
        device="cpu",
    ):
        """Single training step with adversarial examples.

        Args:
            data: PyG Data object
            train_mask: boolean mask of training nodes
            optimizer: optimizer
            epoch: current epoch number
            device: compute device

        Returns:
            loss: combined loss value
            clean_loss: loss on clean data
            adv_loss: loss on adversarial data
        """
        model = self.model.to(device)
        data = data.to(device)

        optimizer.zero_grad()

        # Clean forward pass
        logits_clean = model(data.x, data.edge_index)
        clean_loss = F.cross_entropy(logits_clean[train_mask], data.y[train_mask])

        total_loss = clean_loss

        # Add adversarial loss periodically
        if epoch % self.attack_freq == 0:
            # Generate adversarial examples for anomaly nodes in training set
            # Convert train_mask from indices to boolean mask
            if train_mask.dtype == torch.bool:
                train_bool = train_mask
            else:
                train_bool = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
                train_bool[train_mask] = True
            adv_target_mask = (data.y > 0) & train_bool

            if adv_target_mask.sum() > 0:
                # Graph adversarial example
                graph_data, _ = self.graph_attack.attack(data, adv_target_mask, device)

                # Feature adversarial example
                feature_data, _ = self.feature_attack.attack(data, adv_target_mask, device)

                # Combine both attacks
                from torch_geometric.data import Data
                combined_data = Data(
                    x=feature_data.x,
                    edge_index=graph_data.edge_index,
                    y=data.y,
                )

                # Adversarial forward pass
                logits_adv = model(combined_data.x, combined_data.edge_index)
                adv_loss = F.cross_entropy(logits_adv[train_mask], data.y[train_mask])

                total_loss = total_loss + self.alpha * adv_loss

        total_loss.backward()
        optimizer.step()

        return total_loss.item(), clean_loss.item()


class FeatureSmoothing(nn.Module):
    """Feature smoothing defense layer.

    Applies diffusion-based smoothing to node features,
    making it harder for feature-space attacks to create
    distinguishable perturbations.
    """

    def __init__(self, alpha: float = 0.1, n_steps: int = 3):
        super().__init__()
        self.alpha = alpha
        self.n_steps = n_steps

    def forward(self, x, edge_index):
        """Apply feature smoothing.

        Args:
            x: node features [n_nodes, feature_dim]
            edge_index: graph edges [2, n_edges]

        Returns:
            smoothed features [n_nodes, feature_dim]
        """
        n = x.size(0)
        row, col = edge_index

        # Compute degree normalization
        deg = torch.bincount(row, minlength=n).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Message passing with self-loop
        x_smooth = x.clone()
        for _ in range(self.n_steps):
            # Aggregate neighbors (symmetric normalization)
            x_msg = x_smooth[col] * deg_inv_sqrt[col].unsqueeze(1)
            x_neighbor = torch.zeros_like(x_smooth)
            x_neighbor.index_add_(0, row, x_msg)
            x_neighbor = x_neighbor * deg_inv_sqrt.unsqueeze(1)

            # Blend with original
            x_smooth = (1 - self.alpha) * x_smooth + self.alpha * x_neighbor

        return x_smooth


class RobustDetector(nn.Module):
    """Robust detector that applies feature denoising before classification.

    Wraps a base GNN detector with a feature smoothing layer that
    removes adversarial perturbations from input features.
    """

    def __init__(self, base_model: nn.Module, smoothing_alpha: float = 0.3, smoothing_steps: int = 5):
        super().__init__()
        self.base_model = base_model
        self.smoothing = FeatureSmoothing(alpha=smoothing_alpha, n_steps=smoothing_steps)

    def forward(self, x, edge_index):
        x_smooth = self.smoothing(x, edge_index)
        return self.base_model(x_smooth, edge_index)

    def get_embedding(self, x, edge_index):
        x_smooth = self.smoothing(x, edge_index)
        return self.base_model.get_embedding(x_smooth, edge_index)


class EdgeRegularization:
    """Edge-level regularization defense.

    Adds a penalty for suspicious edge patterns during training:
    - Anomalous nodes having edges to other anomalous nodes
    - Encourages the model to be less sensitive to individual edges
    """

    def __init__(self, lambda_edge: float = 0.01):
        self.lambda_edge = lambda_edge

    def compute_loss(self, edge_index, predictions, labels):
        """Compute edge regularization loss.

        Penalizes confident predictions on anomalous nodes
        that are connected to other anomalous nodes.
        """
        row, col = edge_index

        # Find edges between anomalous nodes
        anomaly_mask = labels > 0
        anomaly_edges = anomaly_mask[row] & anomaly_mask[col]

        if anomaly_edges.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        # For these edges, penalize confident predictions
        anomalous_node_preds = predictions[anomaly_mask]
        # High confidence on anomaly class is good, but we penalize
        # confident predictions on nodes that form dense anomaly clusters
        conf = torch.softmax(anomalous_node_preds, dim=1)
        anomaly_conf = conf[:, 1:].max(dim=1)[0]  # confidence on anomaly classes

        # Penalize high confidence (encourage uncertainty for potential attacks)
        reg_loss = self.lambda_edge * anomaly_conf.mean()

        return reg_loss
