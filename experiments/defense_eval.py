"""
Adversarial defense evaluation: train defended model and compare robustness.
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
from src.models.attack import MetaGradGraphAttack, FeatureAttack
from src.models.defense import AdversarialTraining, RobustDetector
from src.utils.metrics import compute_attack_success_rate, compute_metrics
from src.utils.visualization import plot_attack_results, plot_comparison_bar


def train_clean_model(model, data, train_mask, val_mask, args, device="cpu"):
    """Train model without adversarial training (baseline)."""
    import torch.nn as nn
    import torch.optim as optim

    model = model.to(device)
    data = data.to(device)

    class_counts = torch.bincount(data.y)
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
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
    return model


def train_defended_model(model, data, train_mask, val_mask, args, device="cpu"):
    """Train model with adversarial training defense."""
    import torch.nn as nn
    import torch.optim as optim

    model = model.to(device)
    data = data.to(device)

    class_counts = torch.bincount(data.y)
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    adv_trainer = AdversarialTraining(
        model,
        attack_budget=0.03,
        feature_bound=2.0,
        attack_freq=2,
        alpha=1.0,
    )

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss, clean_loss = adv_trainer.train_step(
            data, train_mask, optimizer, epoch, device
        )

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_preds = val_logits.argmax(dim=1)
            val_metrics = compute_metrics(data.y[val_mask], val_preds[val_mask])

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={total_loss:.4f}, clean_loss={clean_loss:.4f}, val_f1={val_metrics['macro_f1']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_feature_attack(model, data, target_mask, device="cpu"):
    """Evaluate model under feature attack with varying perturbation bounds."""
    bounds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    asr_values = []
    for bound in bounds:
        feat_attack = FeatureAttack(model, perturbation_bound=bound, n_steps=50)
        perturbed_data, feat_info = feat_attack.attack(data, target_mask, device)
        asr, _ = compute_attack_success_rate(model, data, target_mask, perturbed_data, device)
        asr_values.append(asr)
        print(f"  Bound={bound:.1f}: ASR={asr:.4f}, pert_norm={feat_info['perturbation_norm']:.4f}")
    return bounds, asr_values


def evaluate_graph_attack(model, data, target_mask, budgets, device="cpu"):
    """Evaluate model under graph attack with varying budgets."""
    graph_attack = MetaGradGraphAttack(model, budget_ratio=0.05)
    asr_values = []
    for budget in budgets:
        graph_attack.budget_ratio = budget
        perturbed_data, attack_info = graph_attack.attack(data, target_mask, device)
        asr, _ = compute_attack_success_rate(model, data, target_mask, perturbed_data, device)
        asr_values.append(asr)
        print(f"  Budget={budget:.2f}: ASR={asr:.4f}")
    return asr_values


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial defense")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
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
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10])
    args = parser.parse_args()

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
    train_mask, val_mask, test_mask = (
        train_mask.to(args.device),
        val_mask.to(args.device),
        test_mask.to(args.device),
    )
    target_mask = labels > 0

    # === Train baseline model ===
    print("\n=== Training Baseline Model ===")
    baseline_model = GCNDetector(
        in_channels=args.feature_dim,
        hidden_channels=args.hidden_dim,
        out_channels=3,
    )
    baseline_model = train_clean_model(baseline_model, data, train_mask, val_mask, args, args.device)

    baseline_model.eval()
    with torch.no_grad():
        test_logits = baseline_model(data.x, data.edge_index)
        test_preds = test_logits.argmax(dim=1)
        test_metrics = compute_metrics(data.y[test_mask], test_preds[test_mask])
    print(f"  Clean test F1: {test_metrics['macro_f1']:.4f}")

    # === Evaluate baseline under attack ===
    print("\n=== Baseline Under Feature Attack ===")
    feat_bounds, asr_baseline_feat = evaluate_feature_attack(baseline_model, data, target_mask, args.device)

    print("\n=== Baseline Under Graph Attack ===")
    asr_baseline_graph = evaluate_graph_attack(baseline_model, data, target_mask, args.budgets, args.device)

    # === Train defended model (RobustDetector with adversarial training) ===
    print("\n=== Training Defended Model ===")
    base_model = GCNDetector(
        in_channels=args.feature_dim,
        hidden_channels=args.hidden_dim,
        out_channels=3,
    )
    defended_model = RobustDetector(base_model, smoothing_alpha=0.3, smoothing_steps=5)
    defended_model = train_defended_model(defended_model, data, train_mask, val_mask, args, args.device)

    defended_model.eval()
    with torch.no_grad():
        test_logits = defended_model(data.x, data.edge_index)
        test_preds = test_logits.argmax(dim=1)
        test_metrics_def = compute_metrics(data.y[test_mask], test_preds[test_mask])
    print(f"  Clean test F1: {test_metrics_def['macro_f1']:.4f}")

    # === Evaluate defended under attack ===
    print("\n=== Defended Model Under Feature Attack ===")
    _, asr_defended_feat = evaluate_feature_attack(defended_model, data, target_mask, args.device)

    print("\n=== Defended Model Under Graph Attack ===")
    asr_defended_graph = evaluate_graph_attack(defended_model, data, target_mask, args.budgets, args.device)

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"{'':<20} {'Baseline F1':<15} {'Defended F1':<15}")
    print(f"{'Clean accuracy':<20} {test_metrics['macro_f1']:<15.4f} {test_metrics_def['macro_f1']:<15.4f}")

    print(f"\nFeature Attack (ASR):")
    print(f"{'Bound':<8} {'Baseline':<15} {'Defended':<15} {'Improvement':<15}")
    print("-" * 53)
    for b, base, def_ in zip(feat_bounds, asr_baseline_feat, asr_defended_feat):
        imp = (base - def_) / base * 100 if base > 0 else 0
        print(f"{b:<8.1f} {base:<15.4f} {def_:<15.4f} {imp:<15.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "baseline_clean_f1": round(test_metrics["macro_f1"], 4),
        "defended_clean_f1": round(test_metrics_def["macro_f1"], 4),
        "feature_attack": {
            "bounds": feat_bounds,
            "baseline_asr": asr_baseline_feat,
            "defended_asr": asr_defended_feat,
        },
        "graph_attack": {
            "budgets": args.budgets,
            "baseline_asr": asr_baseline_graph,
            "defended_asr": asr_defended_graph,
        },
    }

    results_path = os.path.join(args.out_dir, f"defense_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot feature attack comparison
    plot_attack_results(feat_bounds, asr_baseline_feat, asr_defended_feat, save_path=os.path.join(args.out_dir, "defense_comparison.png"))

    # Bar chart at bound=2.0
    idx = feat_bounds.index(2.0)
    plot_comparison_bar(
        categories=["Feature Attack (bound=2.0)"],
        values_before=[asr_baseline_feat[idx]],
        values_after=[asr_defended_feat[idx]],
        labels=["No Defense", "With Defense"],
        save_path=os.path.join(args.out_dir, "defense_bar.png"),
        metric_name="Attack Success Rate",
    )


if __name__ == "__main__":
    main()
