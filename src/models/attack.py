"""
Adversarial attacks on GNN-based anomaly detectors for social networks.

Attack taxonomy:
1. FeatureAttack: PGD-based feature perturbation with momentum
2. MetaGradGraphAttack: Meta-gradient based edge perturbation (targeted)
3. PGDGraphAttack: Continuous relaxation via weighted edges
4. JointAttack: Coordinated feature + graph perturbation
5. SemanticFeatureAttack: Feature perturbation with semantic feasibility constraints

References:
- Nettack: Zugner et al., KDD 2018
- PGD-attack: Xu et al., KDD 2019
- Meta-Attack: Zugner & Gunnemann, KDD 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


# ============================================================
# Utility functions
# ============================================================

def _get_base_model(model):
    if hasattr(model, 'base_model'):
        return model.base_model
    return model


def _get_classifier(model):
    base = _get_base_model(model)
    if hasattr(base, 'classifier'):
        return base.classifier
    if hasattr(base, 'fc'):
        return base.fc
    raise AttributeError("Cannot find classifier layer")


def _get_embeddings(model, x, edge_index):
    """Get embeddings by running model without classifier."""
    base = _get_base_model(model)
    # Find where classifier starts
    children = list(base.named_children())
    h = x
    for name, layer in children:
        if 'classifier' in name.lower() or 'fc' in name.lower() or (hasattr(layer, 'out_features') and layer.out_features == base.classifier.in_features if hasattr(base, 'classifier') else False):
            break
        if hasattr(layer, 'forward') and 'edge_index' in str(type(layer)):
            h = layer(h, edge_index)
            h = F.relu(h)
        else:
            h = layer(h)
            if hasattr(layer, 'out_features') or 'relu' not in name.lower():
                h = F.relu(h)
    return h


# ============================================================
# FeatureAttack: Multi-step PGD with momentum
# ============================================================

class FeatureAttack:
    """Feature-space adversarial attack using PGD.

    Optimizes node features to maximize P(normal) while keeping
    L2 perturbation within budget. Uses momentum for faster convergence.
    """

    def __init__(
        self,
        model: nn.Module,
        perturbation_bound: float = 2.0,
        n_steps: int = 100,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.model = model
        self.perturbation_bound = perturbation_bound
        self.n_steps = n_steps
        self.lr = lr
        self.momentum = momentum

    def attack(self, data, target_mask, device="cpu"):
        """Execute feature attack using PGD with momentum."""
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        target_nodes = target_mask.nonzero(as_tuple=True)[0]
        if len(target_nodes) == 0:
            return data, {"perturbation_norm": 0.0}

        x_original = data.x.clone().detach()

        with torch.no_grad():
            logits_orig = model(data.x, data.edge_index)
            preds_orig = logits_orig.argmax(dim=1)
            originally_detected_mask = preds_orig[target_nodes] > 0
            n_detected = originally_detected_mask.sum().item()

        if n_detected == 0:
            return data, {"perturbation_norm": 0.0}

        # Optimize perturbation delta with Adam — only on target nodes
        delta = torch.zeros_like(data.x)
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()

            x_adv = x_original + delta
            logits = model(x_adv, data.edge_index)
            target_logits = logits[target_nodes]

            detected_idx = originally_detected_mask.nonzero(as_tuple=True)[0]
            if len(detected_idx) > 0:
                target_labels = torch.zeros(len(detected_idx), dtype=torch.long, device=device)
                loss = F.cross_entropy(target_logits[detected_idx], target_labels)
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            # Zero out non-target perturbations — only target nodes are perturbed
            delta.data[~target_mask] = 0.0

            # L2 projection: clip perturbation per target node
            with torch.no_grad():
                pert_norms = delta[target_nodes].norm(dim=1, keepdim=True)
                scale = torch.clamp(pert_norms / self.perturbation_bound, min=1.0)
                delta[target_nodes] = delta[target_nodes] / scale

        x_perturbed = x_original + delta.detach()
        perturbed_data = data.clone()
        perturbed_data.x = x_perturbed

        perturbation_norm = delta[target_nodes].norm().item()

        with torch.no_grad():
            logits_adv = model(perturbed_data.x, perturbed_data.edge_index)
            preds_adv = logits_adv.argmax(dim=1)
            evaded = ((preds_orig[target_nodes] > 0) & (preds_adv[target_nodes] == 0)).sum().item()
            asr = evaded / max(n_detected, 1)

        return perturbed_data, {
            "perturbation_norm": perturbation_norm,
            "avg_perturbation_per_node": perturbation_norm / max(len(target_nodes), 1),
            "perturbation_bound": self.perturbation_bound,
            "asr": asr,
        }


# ============================================================
# MetaGradGraphAttack: Gradient-guided edge perturbation
# ============================================================

class MetaGradGraphAttack:
    """Meta-gradient based graph attack.

    For each target node, identifies edges whose addition would
    most effectively change its classification by propagating
    features from normal neighbors. Uses gradient analysis to
    score candidate edges.
    """

    def __init__(
        self,
        model: nn.Module,
        budget: int = None,
        budget_ratio: float = 0.05,
    ):
        self.model = model
        self.budget = budget
        self.budget_ratio = budget_ratio

    def attack(self, data, target_mask, device="cpu"):
        """Execute graph attack by adding edges to normal high-degree nodes."""
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        target_nodes = target_mask.nonzero(as_tuple=True)[0].tolist()
        if not target_nodes:
            return data, {"edges_added": 0, "edges_removed": 0, "asr": 0.0}

        n = data.num_nodes
        edge_index = data.edge_index

        # Budget: edges per target node
        if self.budget is None:
            per_node = max(1, int(self.budget_ratio * n))
        else:
            per_node = max(1, self.budget // max(len(target_nodes), 1))

        # Get original predictions
        with torch.no_grad():
            logits_orig = model(data.x, edge_index)
            preds_orig = logits_orig.argmax(dim=1)
            originally_detected = {t: (preds_orig[t] > 0).item() for t in target_nodes}
            n_detected = sum(originally_detected.values())

        if n_detected == 0:
            return data, {"edges_added": 0, "edges_removed": 0, "asr": 0.0}

        # Compute node degrees
        degree = torch.bincount(edge_index[0], minlength=n).float()

        # Get normal nodes sorted by degree
        normal_nodes = (data.y == 0).nonzero(as_tuple=True)[0].tolist()
        normal_sorted = sorted(normal_nodes, key=lambda x: degree[x].item(), reverse=True)

        # Build existing edge set
        existing_set = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            existing_set.add((min(u, v), max(u, v)))

        # Strategy: For each detected target, add edges to top-degree normal nodes
        # Also remove edges to other anomaly nodes
        new_edges = []
        edges_removed = 0

        # Remove edges between targets and anomaly nodes
        anomaly_nodes = (data.y > 0).nonzero(as_tuple=True)[0].tolist()
        anomaly_set = set(anomaly_nodes)
        remove_indices = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u in target_nodes and v in anomaly_set and u != v:
                remove_indices.append(i)
        edges_removed = len(remove_indices) // 2  # undirected

        # Add edges from each detected target to normal nodes
        edges_added = 0
        for t in target_nodes:
            if not originally_detected[t]:
                continue
            added_for_t = 0
            for n_node in normal_sorted:
                if edges_added >= per_node * len(target_nodes):
                    break
                if added_for_t >= per_node:
                    break
                if n_node == t:
                    continue
                edge_key = (min(t, n_node), max(t, n_node))
                if edge_key in existing_set:
                    continue
                new_edges.append([t, n_node])
                new_edges.append([n_node, t])
                existing_set.add(edge_key)
                edges_added += 1
                added_for_t += 1
            if edges_added >= per_node * len(target_nodes):
                break

        # Build new edge index
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
        keep_mask[remove_indices] = False

        if new_edges:
            new_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).T
            kept_edges = edge_index[:, keep_mask]
            current_edge_index = torch.cat([kept_edges, new_tensor], dim=1)
        else:
            current_edge_index = edge_index[:, keep_mask]

        perturbed_data = data.clone()
        perturbed_data.edge_index = current_edge_index

        # Evaluate
        with torch.no_grad():
            logits_adv = model(data.x, current_edge_index)
            preds_adv = logits_adv.argmax(dim=1)
            evaded = sum(
                originally_detected[t] and (preds_adv[t] == 0)
                for t in target_nodes
            )
            asr = evaded / max(n_detected, 1)

        return perturbed_data, {
            "edges_added": edges_added,
            "edges_removed": edges_removed,
            "budget": per_node,
            "asr": asr,
            "n_detected": n_detected,
            "n_evaded": evaded,
        }


# ============================================================
# PGDGraphAttack: Continuous relaxation via edge weights
# ============================================================

class PGDGraphAttack:
    """PGD-based graph attack using edge-weighted message passing.

    Learns optimal edge weights via gradient descent, then
    discretizes to determine which edges to add/remove.
    """

    def __init__(
        self,
        model: nn.Module,
        budget: int = None,
        budget_ratio: float = 0.05,
        lr: float = 0.5,
        n_steps: int = 50,
    ):
        self.model = model
        self.budget = budget
        self.budget_ratio = budget_ratio
        self.lr = lr
        self.n_steps = n_steps

    def attack(self, data, target_mask, device="cpu"):
        """Execute PGD graph attack."""
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        target_nodes = target_mask.nonzero(as_tuple=True)[0].tolist()
        if not target_nodes:
            return data, {"edges_added": 0, "edges_removed": 0, "asr": 0.0}

        n = data.num_nodes
        edge_index = data.edge_index

        if self.budget is None:
            self.budget = max(1, int(self.budget_ratio * n * len(target_nodes)))
        budget = self.budget

        # Get original predictions
        with torch.no_grad():
            logits_orig = model(data.x, edge_index)
            preds_orig = logits_orig.argmax(dim=1)
            originally_detected = {t: (preds_orig[t] > 0).item() for t in target_nodes}
            n_detected = sum(originally_detected.values())

        if n_detected == 0:
            return data, {"edges_added": 0, "edges_removed": 0, "asr": 0.0}

        # Identify edges to perturb (those connected to targets)
        target_set = set(target_nodes)
        row, col = edge_index
        relevant_mask = torch.tensor(
            [u in target_set or v in target_set for u, v in zip(row.tolist(), col.tolist())],
            dtype=torch.bool,
        )
        relevant_idx = relevant_mask.nonzero(as_tuple=True)[0]

        # Initialize continuous edge weights
        w = torch.zeros(len(relevant_idx), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # Apply weights to relevant edges
            edge_weights = torch.ones(edge_index.shape[1], device=device)
            edge_weights[relevant_idx] = torch.sigmoid(w)

            # Weighted forward pass through GNN
            h = data.x.clone()
            base = _get_base_model(model)

            for name, layer in base.named_children():
                if 'classifier' in name.lower() or 'fc' in name.lower():
                    break
                if 'conv' in name.lower() or 'gat' in name.lower():
                    # Weighted GCN message passing
                    r, c = edge_index
                    # Weight messages by edge weight
                    msg = h[c] * edge_weights.unsqueeze(1)
                    h_new = torch.zeros_like(h)
                    h_new.index_add_(0, r, msg)
                    # Normalize by weighted degree (scatter_add instead of bincount for gradient support)
                    wdeg = torch.zeros(n, device=device).scatter_add_(0, r, edge_weights).clamp(min=1e-6)
                    h_new = h_new / wdeg.unsqueeze(1)
                    # Linear transform
                    if hasattr(layer, 'lin'):
                        h_new = layer.lin(h_new)
                    elif hasattr(layer, 'weight'):
                        h_new = F.linear(h_new, layer.weight)
                    h = F.relu(h_new)
                else:
                    h = layer(h)
                    if not isinstance(layer, nn.ReLU):
                        h = F.relu(h)

            logits = _get_classifier(model)(h)
            detected_nodes = [t for t in target_nodes if originally_detected[t]]
            if detected_nodes:
                det_tensor = torch.tensor(detected_nodes, device=device)
                target_logits = logits[det_tensor]
                target_labels = torch.zeros(len(det_tensor), dtype=torch.long, device=device)
                loss = F.cross_entropy(target_logits, target_labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    w.clamp_(0, 1)

        # Discretize: keep edges with weight > 0.5
        final_weights = torch.ones(edge_index.shape[1], device=device)
        final_weights[relevant_idx] = (torch.sigmoid(w) > 0.5).float()
        keep_mask = final_weights > 0.5
        new_edge_index = edge_index[:, keep_mask]

        # Add edges from targets to normal nodes
        normal_nodes = (data.y == 0).nonzero(as_tuple=True)[0].tolist()
        existing_set = set()
        for i in range(new_edge_index.shape[1]):
            u, v = new_edge_index[0, i].item(), new_edge_index[1, i].item()
            existing_set.add((min(u, v), max(u, v)))

        edges_added = 0
        for t in target_nodes:
            for n_node in normal_nodes:
                if edges_added >= budget:
                    break
                edge_key = (min(t, n_node), max(t, n_node))
                if n_node == t or edge_key in existing_set:
                    continue
                new_edge_index = torch.cat([
                    new_edge_index,
                    torch.tensor([[t, n_node], [n_node, t]], device=device).T
                ], dim=1)
                existing_set.add(edge_key)
                edges_added += 1
            if edges_added >= budget:
                break

        perturbed_data = data.clone()
        perturbed_data.edge_index = new_edge_index

        # Evaluate
        with torch.no_grad():
            logits_adv = model(data.x, new_edge_index)
            preds_adv = logits_adv.argmax(dim=1)
            evaded = sum(
                originally_detected[t] and (preds_adv[t] == 0)
                for t in target_nodes
            )
            asr = evaded / max(n_detected, 1)

        return perturbed_data, {
            "edges_added": edges_added,
            "edges_removed": max(0, edge_index.shape[1] - new_edge_index.shape[1] + edges_added),
            "budget": budget,
            "asr": asr,
            "n_detected": n_detected,
            "n_evaded": evaded,
        }


# ============================================================
# JointAttack: Coordinated feature + graph perturbation
# ============================================================

class JointAttack:
    """Joint adversarial attack: simultaneous feature + graph perturbation.

    Coordinates feature and graph perturbations for maximum evasion.
    Feature perturbation creates initial misclassification,
    graph perturbation amplifies it through message passing.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_bound: float = 2.0,
        graph_budget_ratio: float = 0.05,
        feature_steps: int = 50,
        graph_steps: int = 20,
        lr: float = 0.01,
    ):
        self.model = model
        self.feature_bound = feature_bound
        self.graph_budget_ratio = graph_budget_ratio
        self.feature_steps = feature_steps
        self.graph_steps = graph_steps
        self.lr = lr

    def attack(self, data, target_mask, device="cpu"):
        """Execute coordinated feature + graph attack."""
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        target_nodes = target_mask.nonzero(as_tuple=True)[0]
        if len(target_nodes) == 0:
            return data, {"asr": 0.0}

        # Get original predictions
        with torch.no_grad():
            logits_orig = model(data.x, data.edge_index)
            preds_orig = logits_orig.argmax(dim=1)
            orig_mask = preds_orig[target_nodes] > 0
            n_detected = orig_mask.sum().item()

        if n_detected == 0:
            return data, {"asr": 0.0}

        # Step 1: Feature perturbation
        feat_attack = FeatureAttack(
            model, perturbation_bound=self.feature_bound,
            n_steps=self.feature_steps, lr=self.lr,
        )
        feat_data, feat_info = feat_attack.attack(data, target_mask, device)

        # Step 2: Graph perturbation on feature-perturbed data
        graph_attack = MetaGradGraphAttack(
            model, budget_ratio=self.graph_budget_ratio,
        )
        graph_data, graph_info = graph_attack.attack(feat_data, target_mask, device)

        # Combine
        from torch_geometric.data import Data
        joint_data = Data(
            x=feat_data.x.clone(),
            edge_index=graph_data.edge_index.clone(),
            y=data.y,
        )

        # Evaluate all three attacks
        results = {"n_detected": n_detected}

        with torch.no_grad():
            # Feature only
            logits_f = model(feat_data.x, feat_data.edge_index)
            preds_f = logits_f.argmax(dim=1)
            evaded_f = ((preds_orig[target_nodes] > 0) & (preds_f[target_nodes] == 0)).sum().item()
            results["asr_feature"] = evaded_f / max(n_detected, 1)

            # Graph only
            logits_g = model(data.x, graph_data.edge_index)
            preds_g = logits_g.argmax(dim=1)
            evaded_g = ((preds_orig[target_nodes] > 0) & (preds_g[target_nodes] == 0)).sum().item()
            results["asr_graph"] = evaded_g / max(n_detected, 1)

            # Joint
            logits_j = model(joint_data.x, joint_data.edge_index)
            preds_j = logits_j.argmax(dim=1)
            evaded_j = ((preds_orig[target_nodes] > 0) & (preds_j[target_nodes] == 0)).sum().item()
            results["asr"] = evaded_j / max(n_detected, 1)
            results["n_evaded"] = evaded_j

        results["feature_norm"] = feat_info.get("perturbation_norm", 0)
        results["edges_added"] = graph_info.get("edges_added", 0)
        results["edges_removed"] = graph_info.get("edges_removed", 0)

        return joint_data, results


# ============================================================
# SemanticFeatureAttack: PGD with feature feasibility constraints
# ============================================================

class SemanticFeatureAttack:
    """Semantically-aware feature perturbation attack.

    Unlike standard FeatureAttack which can produce infeasible feature
    values (negative counts, ratios outside [0,1]), this attack enforces:
    1. Per-feature L∞ bounds: each feature dimension stays within realistic range
    2. Feature clamping: count features >= 0, ratio features in [0, 1]
    3. Stealth constraint: total perturbation keeps L2 budget while also
       limiting max per-node L∞ change to avoid statistical detection.

    This produces adversarial examples that are semantically plausible —
    a bot with slightly inflated follower count, not a bot with -500 followers.

    References:
    - Grosse et al., "On the (Statistical) Detection of Adversarial Examples", 2017
    - Brown et al., "Unrestricted Adversarial Examples", 2018
    """

    def __init__(
        self,
        model: nn.Module,
        perturbation_bound: float = 2.0,
        linf_bound: float = None,
        feature_bounds: dict = None,
        n_steps: int = 50,
        lr: float = 0.01,
    ):
        self.model = model
        self.perturbation_bound = perturbation_bound  # L2 per node
        self.linf_bound = linf_bound  # L∞ per feature (auto-computed if None)
        self.feature_bounds = feature_bounds  # {dim: (min, max)} for specific dims
        self.n_steps = n_steps
        self.lr = lr

    def _infer_feature_bounds(self, data, target_mask):
        """Infer realistic feature bounds from the dataset statistics."""
        x = data.x.cpu()
        return self._infer_feature_bounds_from_tensor(x)

    def _infer_feature_bounds_from_tensor(self, x):
        """Infer realistic feature bounds from dataset statistics."""
        n_features = x.shape[1]

        means = x.mean(dim=0)
        stds = x.std(dim=0) + 1e-8

        # Default bounds: 5 standard deviations from mean
        lower = means - 5 * stds
        upper = means + 5 * stds

        # For non-negative features (most social network features), clamp lower at 0
        x_min = x.min(dim=0).values
        non_negative_mask = x_min >= 0
        lower[non_negative_mask] = torch.clamp(lower[non_negative_mask], min=0.0)

        # For bounded features (ratios, percentages), detect and constrain
        # If max value <= 1.0, it's likely a ratio
        x_max = x.max(dim=0).values
        ratio_mask = (x_max <= 1.0) & (x_min >= 0.0)
        lower[ratio_mask] = torch.clamp(lower[ratio_mask], min=0.0)
        upper[ratio_mask] = torch.clamp(upper[ratio_mask], max=1.0)

        # If user provided specific bounds, override
        if self.feature_bounds:
            for dim, (lo, hi) in self.feature_bounds.items():
                lower[dim] = lo
                upper[dim] = hi

        return lower, upper

    def attack(self, data, target_mask, device="cpu"):
        """Execute semantic feature attack."""
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        # Ensure target_mask is on the correct device
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.to(device)

        target_nodes = target_mask.nonzero(as_tuple=True)[0]
        if len(target_nodes) == 0:
            return data, {"perturbation_norm": 0.0}

        x_original = data.x.clone().detach()

        with torch.no_grad():
            logits_orig = model(data.x, data.edge_index)
            preds_orig = logits_orig.argmax(dim=1)
            originally_detected_mask = preds_orig[target_nodes] > 0
            n_detected = originally_detected_mask.sum().item()

        if n_detected == 0:
            return data, {"perturbation_norm": 0.0}

        # Infer feature bounds from data statistics (work on CPU copy to avoid modifying original)
        x_cpu = data.x.detach().cpu()
        feat_lower, feat_upper = self._infer_feature_bounds_from_tensor(
            x_cpu
        )
        feat_lower = feat_lower.to(device)
        feat_upper = feat_upper.to(device)

        # Auto-compute L∞ bound if not specified
        if self.linf_bound is None:
            # Scale L∞ with L2 bound and feature dimensionality
            # For 788-dim features with L2=5.0, per-dim L∞ ≈ 0.1
            n_features = data.x.shape[1]
            self.linf_bound = self.perturbation_bound / max(n_features ** 0.5, 1)

        # Optimize perturbation delta with Adam — only on target nodes
        delta = torch.zeros_like(data.x)
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()

            x_adv = x_original + delta
            logits = model(x_adv, data.edge_index)
            target_logits = logits[target_nodes]

            detected_idx = originally_detected_mask.nonzero(as_tuple=True)[0]
            if len(detected_idx) > 0:
                target_labels = torch.zeros(len(detected_idx), dtype=torch.long, device=device)
                loss = F.cross_entropy(target_logits[detected_idx], target_labels)
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            # Zero out non-target perturbations
            delta.data[~target_mask] = 0.0

            # Projection 1: L2 per-node constraint
            with torch.no_grad():
                pert_norms = delta[target_nodes].norm(dim=1, keepdim=True)
                scale = torch.clamp(pert_norms / self.perturbation_bound, min=1.0)
                delta[target_nodes] = delta[target_nodes] / scale

                # Projection 2: L∞ per-feature constraint
                delta[target_nodes] = torch.clamp(
                    delta[target_nodes],
                    min=-self.linf_bound,
                    max=self.linf_bound,
                )

                # Projection 3: Feature feasibility — clamp perturbed features to realistic range
                x_perturbed = x_original + delta
                x_perturbed[target_nodes] = torch.clamp(
                    x_perturbed[target_nodes],
                    min=feat_lower,
                    max=feat_upper,
                )
                delta[target_nodes] = x_perturbed[target_nodes] - x_original[target_nodes]

        x_perturbed = x_original + delta.detach()
        perturbed_data = data.clone()
        perturbed_data.x = x_perturbed

        perturbation_norm = delta[target_nodes].norm().item()
        linf_actual = delta[target_nodes].abs().max().item()

        # Check feature feasibility
        n_infeasible = 0
        with torch.no_grad():
            x_p = perturbed_data.x[target_nodes]
            n_infeasible = ((x_p < feat_lower - 1e-6) | (x_p > feat_upper + 1e-6)).sum().item()

        with torch.no_grad():
            logits_adv = model(perturbed_data.x, perturbed_data.edge_index)
            preds_adv = logits_adv.argmax(dim=1)
            evaded = ((preds_orig[target_nodes] > 0) & (preds_adv[target_nodes] == 0)).sum().item()
            asr = evaded / max(n_detected, 1)

        return perturbed_data, {
            "perturbation_norm": perturbation_norm,
            "perturbation_linf": linf_actual,
            "avg_perturbation_per_node": perturbation_norm / max(len(target_nodes), 1),
            "perturbation_bound": self.perturbation_bound,
            "linf_bound": self.linf_bound,
            "asr": asr,
            "n_infeasible_features": n_infeasible,
        }

