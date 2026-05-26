"""
Evaluation metrics for adversarial robustness experiments.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, average="macro"):
    """Compute classification metrics.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        average: averaging strategy for multi-class

    Returns:
        dict of metrics
    """
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)

    # Binary: combine anomaly (1) and attacker (2) into positive class
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "binary_accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "binary_precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "binary_recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "binary_f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }

    return metrics


def compute_attack_success_rate(model, data, target_mask, perturbed_data=None, device="cpu"):
    """Compute attack success rate.

    Attack is successful if:
    1. Original: model correctly identifies the node as anomalous
    2. After attack: model misclassifies the node as normal

    Args:
        model: trained detector
        data: original data
        target_mask: mask of attacked nodes
        perturbed_data: data after attack (if None, use data)
        device: compute device

    Returns:
        asr: attack success rate (float)
        details: dict with detailed breakdown
    """
    if perturbed_data is None:
        perturbed_data = data

    model.eval()
    model = model.to(device)
    data = data.to(device)
    perturbed_data = perturbed_data.to(device)

    target_nodes = target_mask.nonzero(as_tuple=True)[0]
    if len(target_nodes) == 0:
        return 0.0, {}

    # Get predictions on original data
    with torch.no_grad():
        logits_orig = model(data.x, data.edge_index)
        logits_adv = model(perturbed_data.x, perturbed_data.edge_index)

    preds_orig = logits_orig.argmax(dim=1)
    preds_adv = logits_adv.argmax(dim=1)

    target_preds_orig = preds_orig[target_nodes]
    target_preds_adv = preds_adv[target_nodes]
    target_labels = data.y[target_nodes]

    # Original correct predictions (detected as anomaly)
    originally_detected = target_preds_orig > 0
    n_originally_detected = originally_detected.sum().item()

    # Among those originally detected, how many are now misclassified as normal?
    evaded = originally_detected & (target_preds_adv == 0)
    n_evaded = evaded.sum().item()

    asr = n_evaded / n_originally_detected if n_originally_detected > 0 else 0.0

    details = {
        "n_targets": len(target_nodes),
        "n_originally_detected": n_originally_detected,
        "n_evaded": n_evaded,
        "asr": asr,
    }

    return asr, details


def compute_robustness_curve(model, data, target_mask, attack_fn, budget_range, device="cpu"):
    """Compute robustness curve across different attack budgets.

    Returns:
        budgets: list of budget values
        asr_values: list of corresponding attack success rates
    """
    budgets = []
    asr_values = []

    for budget in budget_range:
        attack_fn.budget_ratio = budget
        perturbed_data, _ = attack_fn.attack(data, target_mask, device)
        asr, _ = compute_attack_success_rate(model, data, target_mask, perturbed_data, device)
        budgets.append(budget)
        asr_values.append(asr)

    return budgets, asr_values


def format_results(results: dict, title: str = "") -> str:
    """Format results dictionary as a readable string."""
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def compute_stealth_score(original_data, perturbed_data, target_mask=None):
    """Compute statistical distinguishability of perturbed graph.

    Measures how well an defender could detect the attack by comparing:
    1. Degree distribution (KS test)
    2. Clustering coefficient change
    3. Eigenvalue spectrum shift (Laplacian)
    4. Feature distribution shift (for feature attacks)

    Lower scores = more stealthy (harder to detect).
    Score range: 0.0 (identical) to 1.0 (maximally different).

    Args:
        original_data: original PyG Data object
        perturbed_data: perturbed PyG Data object
        target_mask: optional mask of attacked nodes

    Returns:
        dict with stealth metrics
    """
    from scipy.stats import ks_2samp

    n = original_data.num_nodes
    device = original_data.x.device

    # 1. Degree distribution comparison
    deg_orig = torch.bincount(original_data.edge_index[0], minlength=n).float().cpu().numpy()
    deg_pert = torch.bincount(perturbed_data.edge_index[0], minlength=n).float().cpu().numpy()
    # Filter zero-degree nodes for KS test
    nonzero_mask = (deg_orig > 0) | (deg_pert > 0)
    if nonzero_mask.sum() > 1:
        ks_stat, ks_pvalue = ks_2samp(deg_orig[nonzero_mask], deg_pert[nonzero_mask])
    else:
        ks_stat, ks_pvalue = 0.0, 1.0

    # 2. Average clustering coefficient change
    def avg_clustering(edge_index, num_nodes):
        row, col = edge_index.cpu()
        adj = {i: set() for i in range(num_nodes)}
        for r, c in zip(row.tolist(), col.tolist()):
            if r != c:
                adj[r].add(c)
        cc_sum = 0.0
        count = 0
        for node in range(num_nodes):
            neighbors = list(adj[node])
            k = len(neighbors)
            if k < 2:
                continue
            edges_between = sum(1 for i in range(len(neighbors))
                               for j in range(i + 1, len(neighbors))
                               if neighbors[j] in adj[neighbors[i]])
            cc_sum += 2.0 * edges_between / (k * (k - 1))
            count += 1
        return cc_sum / max(count, 1)

    cc_orig = avg_clustering(original_data.edge_index, n)
    cc_pert = avg_clustering(perturbed_data.edge_index, n)
    cc_change = abs(cc_pert - cc_orig) / max(cc_orig, 1e-8)

    # 3. Feature distribution shift (if features changed)
    feat_shift = 0.0
    if target_mask is not None and perturbed_data.x.shape == original_data.x.shape:
        target_nodes = target_mask.nonzero(as_tuple=True)[0]
        if len(target_nodes) > 0:
            feat_diff = (perturbed_data.x[target_nodes] - original_data.x[target_nodes]).norm(dim=1)
            feat_shift = feat_diff.mean().item()

    # Overall stealth score: weighted combination
    # Normalize each component to [0, 1] range
    stealth_score = 0.0
    if ks_stat > 0.01:  # Only count if statistically significant
        stealth_score += 0.3 * min(ks_stat * 10, 1.0)
    stealth_score += 0.3 * min(cc_change * 5, 1.0)
    if feat_shift > 0:
        stealth_score += 0.4 * min(feat_shift / 5.0, 1.0)  # Normalize by typical perturbation bound

    stealth_score = min(stealth_score, 1.0)

    return {
        "stealth_score": stealth_score,
        "degree_ks_statistic": ks_stat,
        "degree_ks_pvalue": ks_pvalue,
        "clustering_coefficient_original": cc_orig,
        "clustering_coefficient_perturbed": cc_pert,
        "clustering_change_relative": cc_change,
        "feature_shift_avg": feat_shift,
    }


def compute_transferability(
    source_model,
    target_models,
    data,
    target_mask,
    perturbed_data,
    device="cpu",
):
    """Compute attack transferability across different model architectures.

    Measures how well adversarial examples crafted for one model
    transfer to other models (black-box attack scenario).

    Args:
        source_model: model the attack was crafted against
        target_models: dict of {name: model} for transfer targets
        data: original PyG Data
        target_mask: mask of attacked nodes
        perturbed_data: perturbed data
        device: compute device

    Returns:
        dict with per-model transfer ASR and white-box baseline
    """
    source_asr, _ = compute_attack_success_rate(
        source_model, data, target_mask, perturbed_data, device
    )

    results = {
        "source_model_asr": source_asr,
        "transfer_rates": {},
    }

    for name, model in target_models.items():
        model = model.to(device).eval()
        transfer_asr, _ = compute_attack_success_rate(
            model, data, target_mask, perturbed_data, device
        )
        results["transfer_rates"][name] = transfer_asr

    # Average transfer rate
    if results["transfer_rates"]:
        results["avg_transfer_rate"] = np.mean(list(results["transfer_rates"].values()))
    else:
        results["avg_transfer_rate"] = 0.0

    return results


def compute_perturbation_stats(original_data, perturbed_data, target_mask=None):
    """Compute detailed perturbation statistics.

    Args:
        original_data: original PyG Data
        perturbed_data: perturbed PyG Data
        target_mask: optional mask of attacked nodes

    Returns:
        dict with perturbation statistics
    """
    n = original_data.num_nodes
    stats = {}

    # Feature perturbation
    if not torch.allclose(original_data.x, perturbed_data.x):
        if target_mask is not None:
            target_nodes = target_mask.nonzero(as_tuple=True)[0]
            feat_diff = perturbed_data.x[target_nodes] - original_data.x[target_nodes]
            stats["feature_l2_per_node"] = feat_diff.norm(dim=1).mean().item()
            stats["feature_l2_total"] = feat_diff.norm().item()
            stats["feature_linf_max"] = feat_diff.abs().max().item()
        else:
            feat_diff = perturbed_data.x - original_data.x
            stats["feature_l2_total"] = feat_diff.norm().item()
            stats["feature_linf_max"] = feat_diff.abs().max().item()

    # Graph perturbation
    orig_edges = set()
    for i in range(original_data.edge_index.shape[1]):
        u, v = original_data.edge_index[0, i].item(), original_data.edge_index[1, i].item()
        if u < v:
            orig_edges.add((u, v))

    pert_edges = set()
    for i in range(perturbed_data.edge_index.shape[1]):
        u, v = perturbed_data.edge_index[0, i].item(), perturbed_data.edge_index[1, i].item()
        if u < v:
            pert_edges.add((u, v))

    edges_added = len(pert_edges - orig_edges)
    edges_removed = len(orig_edges - pert_edges)
    stats["edges_added"] = edges_added
    stats["edges_removed"] = edges_removed
    stats["edges_modified"] = edges_added + edges_removed
    stats["edge_perturbation_ratio"] = (edges_added + edges_removed) / max(len(orig_edges), 1)

    return stats
