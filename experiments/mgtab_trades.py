"""
TRADES-style adversarial training evaluation on MGTAB.

TRADES uses KL divergence between clean and adversarial predictions
to enforce smooth decision boundaries, rather than naive adversarial
loss injection. This is the strongest defense baseline for comparison.

Evaluates across all model architectures (GCN, GAT, GraphSAGE, RGCN)
on all 5 random seeds.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.mgtab_loader import MGTABDataset
from src.models.mgtab_models import MGTAB_RelationGCN, MGTAB_RelationGAT, MGTAB_GraphSAGE, MGTAB_RGCN, init_weights
from src.models.attack import FeatureAttack
from src.models.defense_advanced import TRADESAdversarialTrainer
from src.utils.metrics import compute_attack_success_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")
SEEDS = [42, 123, 456, 789, 1024]
MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "RGCN"]


class RGCNWithEdgeType(nn.Module):
    def __init__(self, rgcn, edge_type):
        super().__init__()
        self.rgcn = rgcn
        self.edge_type = edge_type
    def forward(self, x, edge_index):
        return self.rgcn(x, edge_index, self.edge_type)


def build_model(name, emb_dim, edge_type=None):
    if name == "GCN":
        rgcn = MGTAB_RelationGCN(emb_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3)
        return RGCNWithEdgeType(rgcn, edge_type)
    elif name == "GAT":
        rgat = MGTAB_RelationGAT(emb_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3)
        return RGCNWithEdgeType(rgat, edge_type)
    elif name == "GraphSAGE":
        return MGTAB_GraphSAGE(emb_dim, hidden_dim=256, out_dim=2, dropout=0.3)
    elif name == "RGCN":
        rgcn = MGTAB_RGCN(emb_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3)
        return RGCNWithEdgeType(rgcn, edge_type)
    raise ValueError(f"Unknown model: {name}")


def run_trades_eval(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== TRADES Adversarial Training Evaluation on MGTAB ===")
    print(f"Device: {DEVICE}")
    print(f"Betas: [1.0, 3.0, 6.0]")

    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"\nDataset: {desc}")

    data_d = data.to(DEVICE)
    edge_type = data.edge_type.to(DEVICE) if hasattr(data, 'edge_type') else None
    emb_dim = data.x.shape[1]

    # Results: {model: {beta: {clean_f1: [], asr: []}}}
    results = {m: {b: {"clean_f1": [], "asr_eps2": [], "asr_eps5": []} for b in [1.0, 3.0, 6.0]} for m in MODEL_NAMES}
    betas = [1.0, 3.0, 6.0]

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"=== Seed {seed} ===")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)
        shuffled_idx = rng.permutation(data.num_nodes)
        train_idx = shuffled_idx[:int(0.7 * data.num_nodes)]
        val_idx = shuffled_idx[int(0.7 * data.num_nodes):int(0.9 * data.num_nodes)]
        test_idx = shuffled_idx[int(0.9 * data.num_nodes):]

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        train_mask[train_idx] = True
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        val_mask[val_idx] = True
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        test_mask[test_idx] = True

        target_mask_full = (data.y == 1)
        test_targets = target_mask_full & test_mask

        for name in MODEL_NAMES:
            print(f"\n  {name}:")
            for beta in betas:
                # Train with TRADES
                model = build_model(name, emb_dim, edge_type)
                model.apply(init_weights)
                trades = TRADESAdversarialTrainer(
                    model, perturbation_bound=5.0, n_steps=20, lr=0.01, beta=beta
                )
                trained_model, history = trades.train(
                    data_d, train_mask, val_mask, DEVICE,
                    epochs=100, lr=1e-3, weight_decay=5e-4
                )
                trained_model.eval()

                # Clean F1
                with torch.no_grad():
                    out = trained_model(data_d.x, data_d.edge_index)
                    preds = out.argmax(dim=1)
                    clean_f1 = f1_score(data_d.y[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), average="macro")
                    clean_acc = accuracy_score(data_d.y[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy())

                # Attack with eps=2.0
                attack2 = FeatureAttack(trained_model, perturbation_bound=2.0, n_steps=50, lr=0.01)
                perturbed2, _ = attack2.attack(data, target_mask_full, DEVICE)
                asr2, _ = compute_attack_success_rate(trained_model, data, test_targets, perturbed2, DEVICE)

                # Attack with eps=5.0
                attack5 = FeatureAttack(trained_model, perturbation_bound=5.0, n_steps=50, lr=0.01)
                perturbed5, _ = attack5.attack(data, target_mask_full, DEVICE)
                asr5, _ = compute_attack_success_rate(trained_model, data, test_targets, perturbed5, DEVICE)

                results[name][beta]["clean_f1"].append(clean_f1)
                results[name][beta]["asr_eps2"].append(asr2)
                results[name][beta]["asr_eps5"].append(asr5)

                final_kl = history["kl_loss"][-1] if history["kl_loss"] else 0
                print(f"    beta={beta}: Acc={clean_acc:.4f}, F1={clean_f1:.4f}, ASR(eps=2)={asr2:.4f}, ASR(eps=5)={asr5:.4f}, KL={final_kl:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print("TRADES SUMMARY")
    print(f"{'='*60}")
    for name in MODEL_NAMES:
        print(f"\n  {name}:")
        for beta in betas:
            f1_arr = np.array(results[name][beta]["clean_f1"])
            asr2_arr = np.array(results[name][beta]["asr_eps2"])
            asr5_arr = np.array(results[name][beta]["asr_eps5"])
            print(f"    beta={beta}: F1={f1_arr.mean():.4f}+/-{f1_arr.std():.4f}, ASR(eps=2)={asr2_arr.mean():.4f}+/-{asr2_arr.std():.4f}, ASR(eps=5)={asr5_arr.mean():.4f}+/-{asr5_arr.std():.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(out_dir, f"mgtab_trades_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_trades_eval()
