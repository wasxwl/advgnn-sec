"""
Temporal adversarial attack for GNN-based anomaly detection.

Models an attack that evolves over time across graph snapshots.
The attacker gradually modifies features/structure to avoid
detection by temporal anomaly detection systems.

Key insight: gradual perturbation is harder to detect than sudden changes,
as temporal detectors look for abrupt behavioral shifts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TemporalFeatureAttack:
    """Temporal feature attack that evolves over graph snapshots.

    Instead of applying a sudden perturbation, the attacker
    gradually modifies features over T time steps, staying
    below the temporal detector's threshold per step.

    Args:
        model: trained GNN detector
        total_budget: total L2 perturbation budget across all steps
        n_steps: number of attack optimization steps per snapshot
        n_snapshots: number of temporal snapshots
        per_step_budget: maximum perturbation per step (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        total_budget: float = 2.0,
        n_steps: int = 20,
        n_snapshots: int = 5,
        per_step_budget: float = None,
    ):
        self.model = model
        self.total_budget = total_budget
        self.n_steps = n_steps
        self.n_snapshots = n_snapshots
        self.per_step_budget = per_step_budget or (total_budget / n_snapshots)

    def attack(self, data, target_mask, device="cpu") -> Tuple:
        """Execute temporal feature attack.

        Args:
            data: PyG Data with features x
            target_mask: boolean mask of target nodes
            device: compute device

        Returns:
            temporal_data: list of Data objects per snapshot
            history: attack metrics per snapshot
        """
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        target_nodes = target_mask.nonzero(as_tuple=True)[0]
        if len(target_nodes) == 0:
            return [data], [{"asr": 0.0}]

        x_original = data.x.clone().detach()

        # Get original predictions
        with torch.no_grad():
            logits_orig = model(data.x, data.edge_index)
            preds_orig = logits_orig.argmax(dim=1)
            originally_detected = preds_orig[target_nodes] > 0
            n_detected = originally_detected.sum().item()

        if n_detected == 0:
            return [data], [{"asr": 0.0}]

        # Gradual perturbation over snapshots
        x_current = x_original.clone()
        temporal_data = []
        history = []

        for snapshot in range(self.n_snapshots):
            # Optimize perturbation for this snapshot
            delta = torch.zeros_like(data.x)
            delta.requires_grad_(True)
            optimizer = torch.optim.Adam([delta], lr=0.01)

            for step in range(self.n_steps):
                optimizer.zero_grad()

                x_adv = x_current + delta
                logits = model(x_adv, data.edge_index)
                target_logits = logits[target_nodes]

                detected_idx = originally_detected.nonzero(as_tuple=True)[0]
                if len(detected_idx) > 0:
                    target_labels = torch.zeros(len(detected_idx), dtype=torch.long, device=device)
                    loss = F.cross_entropy(target_logits[detected_idx], target_labels)
                else:
                    loss = torch.tensor(0.0, device=device)

                loss.backward()
                optimizer.step()

                # Project: per-snapshot budget
                with torch.no_grad():
                    pert_norms = delta[target_nodes].norm(dim=1, keepdim=True)
                    scale = torch.clamp(pert_norms / self.per_step_budget, min=1.0)
                    delta[target_nodes] = delta[target_nodes] / scale

            # Apply accumulated perturbation
            x_current = x_current + delta.detach()

            # Evaluate at this snapshot
            with torch.no_grad():
                logits_snap = model(x_current, data.edge_index)
                preds_snap = logits_snap.argmax(dim=1)
                evaded = ((preds_orig[target_nodes] > 0) & (preds_snap[target_nodes] == 0)).sum().item()
                asr = evaded / max(n_detected, 1)

            pert_norm = (x_current[target_nodes] - x_original[target_nodes]).norm().item()
            total_norm_ratio = pert_norm / (self.total_budget * len(target_nodes))

            snap_data = data.clone()
            snap_data.x = x_current.clone()
            temporal_data.append(snap_data)

            history.append({
                "snapshot": snapshot + 1,
                "asr": asr,
                "cumulative_perturbation_norm": pert_norm,
                "budget_used_ratio": total_norm_ratio,
                "n_evaded": evaded,
            })

            print(f"  Snapshot {snapshot+1}/{self.n_snapshots}: "
                  f"ASR={asr:.4f}, pert_norm={pert_norm:.4f}, "
                  f"budget_used={total_norm_ratio:.2%}")

        return temporal_data, history


class TemporalGraphAttack:
    """Temporal graph attack that gradually modifies structure.

    Adds/removes edges incrementally over time to avoid
    detection by temporal graph anomaly detectors that
    monitor for sudden structural changes.
    """

    def __init__(
        self,
        model: nn.Module,
        total_edge_budget: int = 100,
        n_snapshots: int = 5,
    ):
        self.model = model
        self.total_edge_budget = total_edge_budget
        self.n_snapshots = n_snapshots
        self.per_snapshot_budget = max(1, total_edge_budget // n_snapshots)

    def attack(self, data, target_mask, device="cpu") -> Tuple:
        """Execute temporal graph attack."""
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        target_nodes = target_mask.nonzero(as_tuple=True)[0].tolist()
        if not target_nodes:
            return [data], [{"asr": 0.0}]

        n = data.num_nodes
        edge_index = data.edge_index.clone()

        with torch.no_grad():
            logits_orig = model(data.x, edge_index)
            preds_orig = logits_orig.argmax(dim=1)
            originally_detected = {t: (preds_orig[t] > 0).item() for t in target_nodes}
            n_detected = sum(originally_detected.values())

        if n_detected == 0:
            return [data], [{"asr": 0.0}]

        # Identify normal nodes to connect to
        normal_nodes = (data.y == 0).nonzero(as_tuple=True)[0].tolist()
        degree = torch.bincount(edge_index[0], minlength=n).float()
        normal_sorted = sorted(normal_nodes, key=lambda x: degree[x].item(), reverse=True)

        # Track existing edges
        existing_set = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            existing_set.add((min(u, v), max(u, v)))

        edge_index_history = []
        history = []
        edges_added_total = 0

        for snapshot in range(self.n_snapshots):
            # Add edges for this snapshot
            edges_added_snap = 0
            for t in target_nodes:
                if not originally_detected.get(t, False):
                    continue
                if edges_added_snap >= self.per_snapshot_budget:
                    break

                for n_node in normal_sorted:
                    if edges_added_snap >= self.per_snapshot_budget:
                        break
                    if n_node == t:
                        continue
                    edge_key = (min(t, n_node), max(t, n_node))
                    if edge_key in existing_set:
                        continue

                    edge_index = torch.cat([
                        edge_index,
                        torch.tensor([[t, n_node], [n_node, t]], device=device).T
                    ], dim=1)
                    existing_set.add(edge_key)
                    edges_added_total += 1
                    edges_added_snap += 1

            # Evaluate
            with torch.no_grad():
                logits_snap = model(data.x, edge_index)
                preds_snap = logits_snap.argmax(dim=1)
                evaded = sum(
                    originally_detected.get(t, False) and (preds_snap[t] == 0)
                    for t in target_nodes
                )
                asr = evaded / max(n_detected, 1)

            snap_data = data.clone()
            snap_data.edge_index = edge_index
            edge_index_history.append(snap_data)

            history.append({
                "snapshot": snapshot + 1,
                "asr": asr,
                "edges_added_cumulative": edges_added_total,
                "edges_added_this_snapshot": edges_added_snap,
                "n_evaded": evaded,
            })

            print(f"  Snapshot {snapshot+1}/{self.n_snapshots}: "
                  f"ASR={asr:.4f}, edges_added={edges_added_total}")

        return edge_index_history, history
