"""
Relation-aware GCN/GAT evaluation on MGTAB.

Tests whether using relation types (7 edge types) closes the ~5pp gap
between our GCN/GAT and the MGTAB paper results:
  GCN: 80.4% -> target 85.8%
  GAT: 81.9% -> target 87.0%

Models: MGTAB_RelationGCN (RGCNConv), MGTAB_RelationGAT (relation-split GAT)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.mgtab_loader import MGTABDataset
from src.models.mgtab_models import MGTAB_RelationGCN, MGTAB_RelationGAT, init_weights
from src.models.attack import FeatureAttack
from src.utils.metrics import compute_attack_success_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")
SEEDS = [42, 123, 456, 789, 1024]


class RelationAwareWrapper(nn.Module):
    """Wrapper that stores edge_type so forward takes (x, edge_index)."""
    def __init__(self, model, edge_type):
        super().__init__()
        self.model = model
        self.edge_type = edge_type

    def forward(self, x, edge_index):
        return self.model(x, edge_index, self.edge_type)


def train_model(model, data_d, train_mask, val_mask, epochs=200, lr=1e-3, weight_decay=5e-4):
    """Train model matching MGTAB paper protocol."""
    model = model.to(DEVICE)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data_d.x, data_d.edge_index)
        loss_train = criterion(output[train_mask], data_d.y[train_mask])
        loss_train.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(data_d.x, data_d.edge_index)
                val_preds = val_output.argmax(dim=1)
                val_acc = accuracy_score(
                    data_d.y[val_mask].cpu().numpy(),
                    val_preds[val_mask].cpu().numpy()
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_relation_aware_eval(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Relation-Aware GCN/GAT Evaluation on MGTAB ===")
    print(f"Device: {DEVICE}")

    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"Dataset: {desc}")

    data_d = data.to(DEVICE)
    edge_type = data.edge_type.to(DEVICE)
    emb_dim = data.x.shape[1]
    num_relations = int(edge_type.max().item()) + 1

    all_results = {}

    for model_name, model_cls in [("RelGCN", MGTAB_RelationGCN), ("RelGAT", MGTAB_RelationGAT)]:
        print(f"\n{'='*60}")
        print(f"=== {model_name} ===")
        print(f"{'='*60}")

        model_results = {"clean": [], "attacks": {}}

        for seed in SEEDS:
            print(f"\n--- {model_name} Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            rng = np.random.RandomState(seed)
            sample_number = data.num_nodes
            shuffled_idx = rng.permutation(sample_number)
            train_idx = shuffled_idx[:int(0.7 * sample_number)]
            val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
            test_idx = shuffled_idx[int(0.9 * sample_number):]

            train_mask = torch.zeros(sample_number, dtype=torch.bool, device=DEVICE)
            train_mask[train_idx] = True
            val_mask = torch.zeros(sample_number, dtype=torch.bool, device=DEVICE)
            val_mask[val_idx] = True
            test_mask = torch.zeros(sample_number, dtype=torch.bool, device=DEVICE)
            test_mask[test_idx] = True

            # Build relation-aware model
            if model_name == "RelGCN":
                model = model_cls(emb_dim, hidden_dim=256, out_dim=2, num_relations=num_relations, dropout=0.3)
            else:
                model = model_cls(emb_dim, hidden_dim=256, out_dim=2, num_relations=num_relations, dropout=0.3)

            wrapped = RelationAwareWrapper(model, edge_type)
            wrapped = train_model(wrapped, data_d, train_mask, val_mask, epochs=200)

            # Evaluate clean
            wrapped.eval()
            with torch.no_grad():
                output = wrapped(data_d.x, data_d.edge_index)
                preds = output.argmax(dim=1)

            labels_np = data_d.y[test_mask].cpu().numpy()
            preds_np = preds[test_mask].cpu().numpy()

            clean_metrics = {
                "accuracy": accuracy_score(labels_np, preds_np),
                "macro_f1": f1_score(labels_np, preds_np, average="macro"),
            }
            print(f"  Clean: Acc={clean_metrics['accuracy']:.4f}, F1={clean_metrics['macro_f1']:.4f}")
            model_results["clean"].append(clean_metrics)

            # Feature attacks
            for budget in [0.5, 1.0, 2.0, 5.0]:
                feat_attack = FeatureAttack(wrapped, perturbation_bound=budget, n_steps=50, lr=0.01)
                perturbed, feat_info = feat_attack.attack(data, (data.y == 1), DEVICE)

                asr, _ = compute_attack_success_rate(
                    wrapped, data, (data.y == 1) & test_mask, perturbed, DEVICE
                )
                print(f"  Attack eps={budget:.1f}: ASR={asr:.4f}")

                budget_key = f"eps_{budget:.0f}"
                if budget_key not in model_results["attacks"]:
                    model_results["attacks"][budget_key] = []
                model_results["attacks"][budget_key].append(asr)

        # Summary
        clean_accs = [r["accuracy"] for r in model_results["clean"]]
        clean_f1s = [r["macro_f1"] for r in model_results["clean"]]

        summary = {
            "accuracy": f"{np.mean(clean_accs)*100:.2f} +/- {np.std(clean_accs)*100:.2f}",
            "macro_f1": f"{np.mean(clean_f1s)*100:.2f} +/- {np.std(clean_f1s)*100:.2f}",
        }
        for budget_key, asr_list in model_results["attacks"].items():
            asr_arr = np.array(asr_list) * 100
            summary[f"attack_{budget_key}_asr"] = f"{np.mean(asr_arr):.2f} +/- {np.std(asr_arr):.2f}"

        model_results["summary"] = summary
        all_results[model_name] = model_results

        print(f"\n--- {model_name} Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"\n| Model       | Our Accuracy    | MGTAB Paper Acc | Gap   |")
    print(f"{'='*60}")
    paper_acc = {"RelGCN": 85.8, "RelGAT": 87.0}
    paper_f1 = {"RelGCN": 78.3, "RelGAT": 82.3}
    for name in ["RelGCN", "RelGAT"]:
        s = all_results[name]["summary"]
        our_acc = float(s["accuracy"].split()[0])
        gap = our_acc - paper_acc[name]
        sign = "+" if gap >= 0 else ""
        print(f"| {name:11s} | {s['accuracy']:<15s} | {paper_acc[name]:.1f}%           | {sign}{gap:.1f}% |")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(out_dir, f"mgtab_relation_aware_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_relation_aware_eval()
