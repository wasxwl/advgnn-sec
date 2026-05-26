"""
RGCN evaluation on MGTAB with relation-aware message passing.
Matches the MGTAB paper's RGCN baseline.

Uses a wrapper model that stores edge_type internally so that
FeatureAttack (which calls model(x, edge_index)) works correctly.
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
from src.models.mgtab_models import MGTAB_RGCN, init_weights
from src.models.attack import FeatureAttack
from src.utils.metrics import compute_attack_success_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")
SEEDS = [42, 123, 456, 789, 1024]


class RGCNWithEdgeType(nn.Module):
    """Wrapper that stores edge_type so forward takes (x, edge_index)."""
    def __init__(self, rgcn, edge_type):
        super().__init__()
        self.rgcn = rgcn
        self.edge_type = edge_type

    def forward(self, x, edge_index):
        return self.rgcn(x, edge_index, self.edge_type)


def train_model(model, data_d, train_mask, val_mask, epochs=200, lr=1e-3, weight_decay=5e-4):
    """Train model."""
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


def run_rgcn_eval(out_dir="results"):
    """Run RGCN evaluation on MGTAB."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== RGCN Evaluation on MGTAB ===")
    print(f"Device: {DEVICE}")

    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"Dataset: {desc}")

    data_d = data.to(DEVICE)
    edge_type = data.edge_type.to(DEVICE)
    emb_dim = data.x.shape[1]

    results = {"rgcn": {"clean": [], "attacks": {}}}

    for seed in SEEDS:
        print(f"\n--- RGCN Seed {seed} ---")
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

        rgcn = MGTAB_RGCN(emb_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3)
        model = RGCNWithEdgeType(rgcn, edge_type)
        model = train_model(model, data_d, train_mask, val_mask, epochs=200)

        # Evaluate clean
        model.eval()
        with torch.no_grad():
            output = model(data_d.x, data_d.edge_index)
            preds = output.argmax(dim=1)

        labels_np = data_d.y[test_mask].cpu().numpy()
        preds_np = preds[test_mask].cpu().numpy()

        clean_metrics = {
            "accuracy": accuracy_score(labels_np, preds_np),
            "macro_f1": f1_score(labels_np, preds_np, average="macro"),
            "precision": precision_score(labels_np, preds_np, average="macro"),
            "recall": recall_score(labels_np, preds_np, average="macro"),
        }
        print(f"  Clean: Acc={clean_metrics['accuracy']:.4f}, F1={clean_metrics['macro_f1']:.4f}")
        results["rgcn"]["clean"].append(clean_metrics)

        # Attack at different budgets
        for budget in [0.5, 1.0, 2.0, 5.0]:
            feat_attack = FeatureAttack(model, perturbation_bound=budget, n_steps=50, lr=0.01)
            perturbed, feat_info = feat_attack.attack(data, (data.y == 1), DEVICE)

            asr, _ = compute_attack_success_rate(
                model, data, (data.y == 1) & test_mask, perturbed, DEVICE
            )
            print(f"  Attack eps={budget:.1f}: ASR={asr:.4f}")

            budget_key = f"eps_{budget:.0f}"
            if budget_key not in results["rgcn"]["attacks"]:
                results["rgcn"]["attacks"][budget_key] = []
            results["rgcn"]["attacks"][budget_key].append(asr)

    # Summary
    clean_accs = [r["accuracy"] for r in results["rgcn"]["clean"]]
    clean_f1s = [r["macro_f1"] for r in results["rgcn"]["clean"]]

    print(f"\n{'='*60}")
    print("RGCN SUMMARY")
    print(f"{'='*60}")
    print(f"Accuracy: {np.mean(clean_accs)*100:.2f} +/- {np.std(clean_accs)*100:.2f}")
    print(f"Macro F1: {np.mean(clean_f1s)*100:.2f} +/- {np.std(clean_f1s)*100:.2f}")
    for budget_key, asr_list in results["rgcn"]["attacks"].items():
        asr_arr = np.array(asr_list) * 100
        print(f"Attack {budget_key} ASR: {np.mean(asr_arr):.2f} +/- {np.std(asr_arr):.2f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(out_dir, f"mgtab_rgcn_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_rgcn_eval()
