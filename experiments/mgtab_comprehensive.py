"""
Comprehensive MGTAB evaluation: SOTA baseline matching + adversarial attacks.

Tests GCN, GAT, GraphSAGE models across 5 seeds.
Attacks each model with feature perturbation at multiple budgets.
Evaluates SVD purification defense.
Reports mean ± std matching MGTAB paper format.
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
from src.models.mgtab_models import MGTAB_RelationGCN, MGTAB_RelationGAT, MGTAB_GraphSAGE, init_weights
from src.models.attack import FeatureAttack
from src.models.defense_advanced import SVDPurification
from src.utils.metrics import compute_attack_success_rate, compute_metrics

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")
SEEDS = [42, 123, 456, 789, 1024]
MODEL_NAMES = ["RelGCN", "RelGAT", "GraphSAGE"]
ATTACK_BUDGETS = [0.5, 1.0, 2.0, 5.0]


class RelationAwareWrapper(nn.Module):
    """Wrapper that stores edge_type so forward takes (x, edge_index)."""
    def __init__(self, model, edge_type):
        super().__init__()
        self.model = model
        self.edge_type = edge_type

    def forward(self, x, edge_index):
        return self.model(x, edge_index, self.edge_type)


def train_model(model, data, train_mask, val_mask, epochs=200, lr=1e-3, weight_decay=5e-4, device="cpu"):
    """Train model matching MGTAB paper protocol."""
    model = model.to(device)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    data_d = data.clone().to(device)
    train_mask_d = train_mask.clone().to(device)
    val_mask_d = val_mask.clone().to(device)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data_d.x, data_d.edge_index)
        loss_train = criterion(output[train_mask_d], data_d.y[train_mask_d])
        loss_train.backward()
        optimizer.step()

        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(data_d.x, data_d.edge_index)
                val_preds = val_output.argmax(dim=1)
                val_acc = accuracy_score(
                    data_d.y[val_mask_d].cpu().numpy(),
                    val_preds[val_mask_d].cpu().numpy()
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, data, mask, device="cpu"):
    """Evaluate model on given mask."""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        preds = output.argmax(dim=1)
    labels_np = data.y[mask].cpu().numpy()
    preds_np = preds[mask].cpu().numpy()
    return {
        "accuracy": accuracy_score(labels_np, preds_np),
        "macro_f1": f1_score(labels_np, preds_np, average="macro"),
        "precision": precision_score(labels_np, preds_np, average="macro"),
        "recall": recall_score(labels_np, preds_np, average="macro"),
    }


def run_mgtab_comprehensive_eval(out_dir="results"):
    """Run comprehensive MGTAB evaluation."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Comprehensive MGTAB Evaluation ===")
    print(f"Device: {DEVICE}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Seeds: {SEEDS}")

    # Load data once
    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"\nDataset: {desc}")
    print(f"  Humans: {(labels == 0).sum().item()}, Bots: {(labels == 1).sum().item()}")

    data_d = data.clone().to(DEVICE)
    edge_type = data.edge_type.to(DEVICE)
    num_relations = int(edge_type.max().item()) + 1
    n_targets = (labels == 1).sum().item()

    # Results storage
    all_results = {
        "dataset": "MGTAB",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "models": {},
        "summary": {},
    }

    for model_name in MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"=== {model_name} ===")
        print(f"{'='*60}")

        model_results = {"clean": [], "attacks": {}, "svd": []}

        for seed in SEEDS:
            print(f"\n--- {model_name} Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create train/val/test split matching MGTAB protocol
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

            # Build model
            emb_dim = data.x.shape[1]
            if model_name == "RelGCN":
                model = MGTAB_RelationGCN(emb_dim, hidden_dim=256, out_dim=2, num_relations=num_relations, dropout=0.3)
                model = RelationAwareWrapper(model, edge_type)
            elif model_name == "RelGAT":
                model = MGTAB_RelationGAT(emb_dim, hidden_dim=256, out_dim=2, num_relations=num_relations, dropout=0.3)
                model = RelationAwareWrapper(model, edge_type)
            elif model_name == "GraphSAGE":
                model = MGTAB_GraphSAGE(emb_dim, hidden_dim=256, out_dim=2, dropout=0.3)

            # Train
            model = train_model(model, data, train_mask, val_mask, epochs=200, device=DEVICE)

            # Evaluate clean
            clean_metrics = evaluate_model(model, data, test_mask, DEVICE)
            print(f"  Clean: Acc={clean_metrics['accuracy']:.4f}, F1={clean_metrics['macro_f1']:.4f}")
            model_results["clean"].append(clean_metrics)

            # Feature attacks at different budgets
            target_mask_full = (data.y == 1)
            for budget in ATTACK_BUDGETS:
                feat_attack = FeatureAttack(model, perturbation_bound=budget, n_steps=50, lr=0.01)

                # Attack uses full data for perturbation but evaluate on test targets
                perturbed, feat_info = feat_attack.attack(data, (data.y == 1), DEVICE)

                asr, _ = compute_attack_success_rate(
                    model, data, target_mask_full & test_mask, perturbed, DEVICE
                )
                print(f"  Attack eps={budget:.1f}: ASR={asr:.4f}")

                budget_key = f"eps_{budget:.0f}"
                if budget_key not in model_results["attacks"]:
                    model_results["attacks"][budget_key] = []
                model_results["attacks"][budget_key].append(asr)

            # SVD defense + attack
            svd = SVDPurification(rank=128, symmetric=True)
            purified_edge, _ = svd.purify(data.edge_index, data.num_nodes, DEVICE)

            svd_data = data.clone()
            svd_data.edge_index = purified_edge

            # Evaluate clean on purified graph
            with torch.no_grad():
                svd_output = model(svd_data.x, svd_data.edge_index)
                svd_preds = svd_output.argmax(dim=1)
            svd_clean_metrics = {
                "accuracy": accuracy_score(data.y[test_mask].cpu().numpy(), svd_preds[test_mask].cpu().numpy()),
                "macro_f1": f1_score(data.y[test_mask].cpu().numpy(), svd_preds[test_mask].cpu().numpy(), average="macro"),
            }

            # Attack SVD-defended model
            svd_attack = FeatureAttack(model, perturbation_bound=5.0, n_steps=50, lr=0.01)
            svd_perturbed, _ = svd_attack.attack(svd_data, target_mask_full, DEVICE)
            svd_asr, _ = compute_attack_success_rate(
                model, data, target_mask_full & test_mask, svd_perturbed, DEVICE
            )
            print(f"  SVD: Clean F1={svd_clean_metrics['macro_f1']:.4f}, ASR={svd_asr:.4f}")

            model_results["svd"].append({
                "clean_f1": svd_clean_metrics["macro_f1"],
                "asr": svd_asr,
            })

        # Compute summary statistics
        clean_accs = [r["accuracy"] for r in model_results["clean"]]
        clean_f1s = [r["macro_f1"] for r in model_results["clean"]]

        summary = {
            "accuracy": f"{np.mean(clean_accs)*100:.2f} +/- {np.std(clean_accs)*100:.2f}",
            "macro_f1": f"{np.mean(clean_f1s)*100:.2f} +/- {np.std(clean_f1s)*100:.2f}",
        }

        for budget_key, asr_list in model_results["attacks"].items():
            asr_arr = np.array(asr_list) * 100
            summary[f"attack_{budget_key}_asr"] = f"{np.mean(asr_arr):.2f} +/- {np.std(asr_arr):.2f}"

        svd_f1s = [r["clean_f1"] for r in model_results["svd"]]
        svd_asrs = [r["asr"] for r in model_results["svd"]]
        summary["svd_clean_f1"] = f"{np.mean(svd_f1s)*100:.2f} +/- {np.std(svd_f1s)*100:.2f}"
        summary["svd_asr"] = f"{np.mean(svd_asrs)*100:.2f} +/- {np.std(svd_asrs)*100:.2f}"

        model_results["summary"] = summary
        all_results["models"][model_name] = model_results

        # Print summary
        print(f"\n--- {model_name} Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    # Overall summary table
    print(f"\n{'='*60}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Accuracy':<20} {'Macro F1':<20}")
    print(f"{'-'*52}")
    for model_name in MODEL_NAMES:
        s = all_results["models"][model_name]["summary"]
        print(f"{model_name:<12} {s['accuracy']:<20} {s['macro_f1']:<20}")

    print(f"\n{'='*60}")
    print("ATTACK RESULTS")
    print(f"{'='*60}")
    for model_name in MODEL_NAMES:
        s = all_results["models"][model_name]["summary"]
        print(f"\n{model_name}:")
        for budget in ATTACK_BUDGETS:
            key = f"attack_eps_{int(budget)}_asr"
            print(f"  eps={budget:.1f}: {s.get(key, 'N/A')}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(out_dir, f"mgtab_comprehensive_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_mgtab_comprehensive_eval()
