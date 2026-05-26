"""
Adversarial attack evaluation: measure attack effectiveness against trained detector.
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.loader import get_dataset
from src.dataset.preprocess import normalize_features
from src.models.detector import GCNDetector, GATDetector
from src.models.attack import MetaGradGraphAttack, FeatureAttack
from src.utils.metrics import compute_attack_success_rate
from src.utils.visualization import plot_attack_results


def tensor_to_python(obj):
    """Recursively convert PyTorch tensors in a dict/list to Python primitives."""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return [tensor_to_python(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial attacks")
    parser.add_argument("--model_path", type=str, default="results/baseline_model.pth")
    parser.add_argument("--dataset", type=str, default="synthetic")
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

    # Load model
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_args = checkpoint["args"]

    if model_args["model"] == "gcn":
        model = GCNDetector(
            in_channels=model_args["feature_dim"],
            hidden_channels=model_args["hidden_dim"],
            out_channels=3,
        )
    else:
        model = GATDetector(
            in_channels=model_args["feature_dim"],
            hidden_channels=model_args["hidden_dim"],
            out_channels=3,
        )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(args.device)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Load dataset
    data, labels, desc = get_dataset(
        dataset_name=args.dataset,
        seed=args.seed,
        n_normal=model_args.get("n_normal", 800),
        n_anomaly=model_args.get("n_anomaly", 200),
        n_attacker=model_args.get("n_attacker", 50),
        feature_dim=model_args.get("feature_dim", 16),
    )
    data = normalize_features(data)
    data = data.to(args.device)
    print(desc)

    # Target: anomaly and attacker nodes
    target_mask = labels > 0

    # === Feature Attack (primary threat) ===
    print("\n=== Feature Adversarial Attack ===")
    feature_attack = FeatureAttack(model, perturbation_bound=2.0, n_steps=20)
    perturbed_data_feat, feat_info = feature_attack.attack(data, target_mask, args.device)
    asr_feat, details_feat = compute_attack_success_rate(model, data, target_mask, perturbed_data_feat, args.device)
    print(f"  ASR={asr_feat:.4f}, perturbation_norm={feat_info['perturbation_norm']:.4f}")

    # === Graph Attack (secondary threat) ===
    print("\n=== Graph Adversarial Attack ===")
    graph_attack = MetaGradGraphAttack(model, budget_ratio=0.05)

    budgets = args.budgets
    asr_graph = []
    details_all = []

    for budget in budgets:
        graph_attack.budget_ratio = budget
        perturbed_data, attack_info = graph_attack.attack(data, target_mask, args.device)
        asr, details = compute_attack_success_rate(model, data, target_mask, perturbed_data, args.device)
        asr_graph.append(asr)
        details_all.append({"budget": budget, **details, **attack_info})
        print(f"  Budget={budget:.2f}: ASR={asr:.4f}, edges_removed={attack_info.get('edges_removed', 0)}, edges_added={attack_info.get('edges_added', 0)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "feature_attack": {
            "asr": asr_feat,
            "info": feat_info,
        },
        "graph_attack": {
            "budgets": budgets,
            "asr_values": asr_graph,
            "details": details_all,
        },
    }

    results_path = os.path.join(args.out_dir, f"attack_results_{timestamp}.json")
    results_serializable = tensor_to_python(results)
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot feature attack comparison
    plot_attack_results(budgets, asr_graph, save_path=os.path.join(args.out_dir, "attack_curve.png"))

    print(f"\n=== Attack Evaluation Complete ===")


if __name__ == "__main__":
    main()
