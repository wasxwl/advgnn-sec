"""
MGTAB real-world dataset evaluation for adversarial robustness.

Trains a GNN-based bot detector on the MGTAB dataset (10,199 users,
788-dim features, 7 relation types) and evaluates adversarial attacks
and defenses on real-world Twitter bot detection data.
"""

import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.mgtab_loader import MGTABDataset
from src.dataset.preprocess import normalize_features
from src.models.detector import GCNDetector
from src.models.attack import FeatureAttack
from src.models.defense_advanced import SVDPurification
from src.utils.metrics import compute_attack_success_rate, compute_metrics

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")


def run_mgtab_eval(seed=42, out_dir="results"):
    """Run adversarial evaluation on MGTAB dataset."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    print(f"=== MGTAB Real-World Dataset Evaluation ===")
    print(f"Device: {DEVICE}, Seed: {seed}")

    # Load data
    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"\nDataset: {desc}")

    # Normalize features
    data = normalize_features(data).to(DEVICE)

    # Binary classification: 0=human, 1=bot
    target_mask = data.y == 1
    n_targets = target_mask.sum().item()
    print(f"  Humans: {(data.y == 0).sum().item()}, Bots: {n_targets}")

    # Train detector
    print(f"\nTraining GCN detector (in_channels=788)...")
    # Scale up model for high-dimensional features
    model = GCNDetector(in_channels=788, hidden_channels=256, out_channels=2)
    model = model.to(DEVICE)

    # Training with class-weighted loss
    class_counts = torch.bincount(data.y)
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_f1 = 0.0
    best_state = None
    patience = 0
    max_patience = 30

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)

        # Train on 60% of nodes
        rng = np.random.RandomState(seed + epoch)
        train_ratio = 0.6
        n_train = int(train_ratio * data.num_nodes)
        train_idx = torch.tensor(
            rng.choice(data.num_nodes, size=n_train, replace=False),
            dtype=torch.long, device=DEVICE
        )
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        train_mask[train_idx] = True

        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x, data.edge_index)
                val_preds = val_logits.argmax(dim=1)
                val_metrics = compute_metrics(data.y, val_preds)

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

        if patience >= max_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Evaluate clean performance
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        clean_metrics = compute_metrics(data.y, preds)

    print(f"\nClean Performance:")
    print(f"  Accuracy: {clean_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {clean_metrics['macro_f1']:.4f}")
    print(f"  Binary F1: {clean_metrics['binary_f1']:.4f}")

    # Feature attack
    print(f"\n=== Feature Attack ===")
    feat_attack = FeatureAttack(model, perturbation_bound=5.0, n_steps=50, lr=0.01)
    perturbed, feat_info = feat_attack.attack(data, target_mask, DEVICE)
    asr_feature, _ = compute_attack_success_rate(model, data, target_mask, perturbed, DEVICE)
    print(f"  Feature ASR: {asr_feature:.4f}")
    print(f"  Perturbation norm: {feat_info.get('perturbation_norm', 0):.4f}")
    print(f"  Avg perturbation per node: {feat_info.get('avg_perturbation_per_node', 0):.4f}")

    # SVD purification defense
    print(f"\n=== SVD Purification Defense ===")
    svd = SVDPurification(rank=128, symmetric=True)
    purified_edge, svd_info = svd.purify(data.edge_index, data.num_nodes, DEVICE)
    n_orig = data.edge_index.shape[1]
    n_purified = purified_edge.shape[1]
    print(f"  Edges: {n_orig} -> {n_purified} (removed {n_orig - n_purified})")

    svd_model = model  # Same model, purified graph
    svd_model.eval()

    with torch.no_grad():
        svd_logits = svd_model(data.x, purified_edge)
        svd_preds = svd_logits.argmax(dim=1)
        svd_clean_metrics = compute_metrics(data.y, svd_preds)
    print(f"  SVD Clean F1: {svd_clean_metrics['macro_f1']:.4f}")

    # Attack SVD-defended model
    print(f"\n=== Attack SVD-Defended Model ===")
    svd_data = data.clone()
    svd_data.edge_index = purified_edge
    svd_attack = FeatureAttack(svd_model, perturbation_bound=5.0, n_steps=50, lr=0.01)
    svd_perturbed, _ = svd_attack.attack(svd_data, target_mask, DEVICE)
    svd_asr, _ = compute_attack_success_rate(svd_model, data, target_mask, svd_perturbed, DEVICE)
    print(f"  SVD-defended Feature ASR: {svd_asr:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("MGTAB EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<40} {'Value':<15}")
    print(f"{'-'*55}")
    print(f"{'Clean Accuracy':<40} {clean_metrics['accuracy']:.4f}")
    print(f"{'Clean Macro F1':<40} {clean_metrics['macro_f1']:.4f}")
    print(f"{'Feature ASR (epsilon=5.0)':<40} {asr_feature:.4f}")
    print(f"{'SVD Clean Macro F1':<40} {svd_clean_metrics['macro_f1']:.4f}")
    print(f"{'SVD-defended ASR':<40} {svd_asr:.4f}")
    print(f"{'ASR Reduction (SVD)':<40} {asr_feature - svd_asr:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "dataset": "MGTAB",
        "timestamp": timestamp,
        "seed": seed,
        "clean_metrics": {k: v for k, v in clean_metrics.items()},
        "feature_asr": asr_feature,
        "perturbation_info": {
            "norm": feat_info.get("perturbation_norm", 0),
            "avg_per_node": feat_info.get("avg_perturbation_per_node", 0),
        },
        "svd_clean_f1": svd_clean_metrics["macro_f1"],
        "svd_asr": svd_asr,
        "svd_edge_reduction": (n_orig - n_purified) / n_orig,
    }

    results_path = os.path.join(out_dir, f"mgtab_eval_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_mgtab_eval()
