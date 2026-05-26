"""
Baseline experiment: Train and evaluate GNN anomaly detector.
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.loader import get_dataset
from src.dataset.preprocess import compute_graph_statistics
from src.models.detector import GCNDetector, GATDetector
from src.train import train_model
from src.utils.metrics import compute_metrics, format_results
from src.utils.visualization import plot_training_curve, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Baseline GNN anomaly detector")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--n_normal", type=int, default=800)
    parser.add_argument("--n_anomaly", type=int, default=200)
    parser.add_argument("--n_attacker", type=int, default=50)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-detect CUDA
    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda:0"

    print(f"Device: {args.device}")
    print(f"Model: {args.model.upper()}")
    print(f"Dataset: {args.dataset}")

    # Load dataset
    print("\n--- Loading Dataset ---")
    data, labels, desc = get_dataset(
        dataset_name=args.dataset,
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly,
        n_attacker=args.n_attacker,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )
    print(desc)
    stats = compute_graph_statistics(data)
    print(f"Graph stats: {stats}")

    # Initialize model
    if args.model == "gcn":
        model = GCNDetector(
            in_channels=args.feature_dim,
            hidden_channels=args.hidden_dim,
            out_channels=3,
        )
    else:
        model = GATDetector(
            in_channels=args.feature_dim,
            hidden_channels=args.hidden_dim,
            out_channels=3,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    print("\n--- Training ---")
    model, history, final_metrics = train_model(
        model,
        data,
        n_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "args": vars(args),
        "graph_stats": stats,
        "n_params": n_params,
        "final_metrics": {k: round(v, 4) for k, v in final_metrics.items()},
        "best_val_f1": round(max(history["val_f1"]), 4),
        "training_history": {
            "train_loss": [round(x, 4) for x in history["train_loss"]],
            "val_loss": [round(x, 4) for x in history["val_loss"]],
            "val_f1": [round(x, 4) for x in history["val_f1"]],
        },
    }

    results_path = os.path.join(args.out_dir, f"baseline_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plots
    plot_training_curve(
        history["train_loss"],
        history["val_loss"],
        history["val_acc"],
        save_path=os.path.join(args.out_dir, "training_curve.png"),
    )

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "metrics": final_metrics,
    }, os.path.join(args.out_dir, "baseline_model.pth"))

    print(f"\n=== Baseline Complete ===")
    print(format_results(final_metrics, "Final Test Results"))


if __name__ == "__main__":
    main()
