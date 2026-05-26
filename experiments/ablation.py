"""
Ablation studies: analyze the impact of individual components.
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.loader import get_dataset
from src.dataset.preprocess import normalize_features, create_train_val_test_split
from src.models.detector import GCNDetector, GATDetector
from src.models.attack import GraphAttack, FeatureAttack
from src.utils.metrics import compute_attack_success_rate, compute_metrics


def run_ablation(args):
    """Run ablation studies."""
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda:0"

    print(f"Device: {args.device}")

    # Load dataset
    data, labels, desc = get_dataset(
        dataset_name=args.dataset,
        seed=args.seed,
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly,
        n_attacker=args.n_attacker,
        feature_dim=args.feature_dim,
    )
    data = normalize_features(data)
    data = data.to(args.device)
    print(desc)

    train_mask, val_mask, test_mask = create_train_val_test_split(data)
    target_mask = labels > 0

    results = {}

    # === Ablation 1: Model Architecture ===
    print("\n=== Ablation: Model Architecture ===")
    for model_type in ["gcn", "gat"]:
        if model_type == "gcn":
            model = GCNDetector(in_channels=args.feature_dim, hidden_channels=args.hidden_dim, out_channels=3)
        else:
            model = GATDetector(in_channels=args.feature_dim, hidden_channels=args.hidden_dim, out_channels=3, num_heads=4)

        # Train
        import torch.nn as nn
        import torch.optim as optim
        model = model.to(args.device)

        class_counts = torch.bincount(data.y)
        class_weights = 1.0 / (class_counts.float() + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_f1 = 0.0
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(data.x, data.edge_index)
                val_preds = val_logits.argmax(dim=1)
                val_metrics = compute_metrics(data.y[val_mask], val_preds[val_mask])

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate under attack
        graph_attack = GraphAttack(model, budget_ratio=0.05)
        perturbed_data, _ = graph_attack.attack(data, target_mask, args.device)
        asr, _ = compute_attack_success_rate(model, data, target_mask, perturbed_data, args.device)

        # Clean accuracy
        model.eval()
        with torch.no_grad():
            test_logits = model(data.x, data.edge_index)
            test_preds = test_logits.argmax(dim=1)
            test_metrics = compute_metrics(data.y[test_mask], test_preds[test_mask])

        results[f"model_{model_type}"] = {
            "clean_f1": round(test_metrics["macro_f1"], 4),
            "clean_acc": round(test_metrics["accuracy"], 4),
            "asr_005": round(asr, 4),
        }
        print(f"  {model_type.upper()}: clean_f1={test_metrics['macro_f1']:.4f}, ASR(0.05)={asr:.4f}")

    # === Ablation 2: Feature Dimension ===
    print("\n=== Ablation: Feature Dimension ===")
    for feat_dim in [8, 16, 32, 64]:
        data_fd, _, _ = get_dataset(
            dataset_name=args.dataset, seed=args.seed,
            n_normal=args.n_normal, n_anomaly=args.n_anomaly,
            n_attacker=args.n_attacker, feature_dim=feat_dim,
        )
        data_fd = normalize_features(data_fd).to(args.device)

        model = GCNDetector(in_channels=feat_dim, hidden_channels=args.hidden_dim, out_channels=3)
        model = model.to(args.device)

        class_counts = torch.bincount(data_fd.y)
        class_weights = 1.0 / (class_counts.float() + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_f1 = 0.0
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data_fd.x, data_fd.edge_index)
            loss = criterion(logits[train_mask], data_fd.y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(data_fd.x, data_fd.edge_index)
                val_preds = val_logits.argmax(dim=1)
                val_metrics = compute_metrics(data_fd.y[val_mask], val_preds[val_mask])

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        graph_attack = GraphAttack(model, budget_ratio=0.05)
        perturbed_data, _ = graph_attack.attack(data_fd, target_mask, args.device)
        asr, _ = compute_attack_success_rate(model, data_fd, target_mask, perturbed_data, args.device)

        model.eval()
        with torch.no_grad():
            test_logits = model(data_fd.x, data_fd.edge_index)
            test_preds = test_logits.argmax(dim=1)
            test_metrics = compute_metrics(data_fd.y[test_mask], test_preds[test_mask])

        results[f"feat_dim_{feat_dim}"] = {
            "clean_f1": round(test_metrics["macro_f1"], 4),
            "asr_005": round(asr, 4),
        }
        print(f"  feat_dim={feat_dim}: clean_f1={test_metrics['macro_f1']:.4f}, ASR(0.05)={asr:.4f}")

    # Save ablation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.out_dir, f"ablation_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--n_normal", type=int, default=800)
    parser.add_argument("--n_anomaly", type=int, default=200)
    parser.add_argument("--n_attacker", type=int, default=50)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    run_ablation(args)
