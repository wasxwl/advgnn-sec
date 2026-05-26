"""
MGTAB defense evaluation across all 5 seeds.

Evaluates FeatureRandomizationDefense and RobustFeatureTrainer across
all model architectures (GCN, GAT, GraphSAGE, RGCN) on all 5 random seeds.
Produces mean +/- std for both clean F1 and defended ASR.
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
from src.models.defense_advanced import FeatureRandomizationDefense, RobustFeatureTrainer
from src.utils.metrics import compute_attack_success_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")
SEEDS = [42, 123, 456, 789, 1024]
MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "RGCN"]
NOISE_LEVELS = [0.05, 0.1, 0.2]


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


def train_model(model, data_d, train_mask, val_mask, device, epochs=200, lr=1e-3, weight_decay=5e-4):
    model = model.to(device)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data_d.x, data_d.edge_index)
        loss = criterion(output[train_mask], data_d.y[train_mask])
        loss.backward()
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


def train_robust_model(model, data_d, train_mask, val_mask, device, noise_std=0.1, epochs=200, lr=1e-3, weight_decay=5e-4):
    """Train with noise-injected features (RobustFeatureTrainer)."""
    model = model.to(device)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Inject noise into features during training
        x_noisy = data_d.x + torch.randn_like(data_d.x) * noise_std
        output = model(x_noisy, data_d.edge_index)
        loss = criterion(output[train_mask], data_d.y[train_mask])
        loss.backward()
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


def run_defense_eval(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== MGTAB Defense Evaluation Across 5 Seeds ===")
    print(f"Device: {DEVICE}")

    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"\nDataset: {desc}")

    data_d = data.to(DEVICE)
    edge_type = data.edge_type.to(DEVICE) if hasattr(data, 'edge_type') else None
    emb_dim = data.x.shape[1]

    # Results: {model: {defense: {noise_std: {f1: [], asr: []}}}}
    results = {
        "feature_randomization": {m: {n: {"def_f1": [], "def_asr": []} for n in NOISE_LEVELS} for m in MODEL_NAMES},
        "robust_training": {m: {n: {"rob_f1": [], "rob_asr": []} for n in NOISE_LEVELS} for m in MODEL_NAMES},
    }

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

            # Train clean model
            model = build_model(name, emb_dim, edge_type)
            model = train_model(model, data_d, train_mask, val_mask, DEVICE)
            model.eval()

            # --- FeatureRandomizationDefense (inference-only) ---
            for noise_std in NOISE_LEVELS:
                defense = FeatureRandomizationDefense(noise_std=noise_std, n_samples=5)
                _, def_preds = defense.defend_predict(model, data_d.x, data_d.edge_index)
                def_f1 = f1_score(data_d.y[test_mask].cpu().numpy(), def_preds[test_mask].cpu().numpy(), average="macro")

                # Craft attack on clean model
                feat_attack = FeatureAttack(model, perturbation_bound=5.0, n_steps=50, lr=0.01)
                perturbed, _ = feat_attack.attack(data, target_mask_full, DEVICE)

                # Evaluate with defense on perturbed data
                _, adv_def_preds = defense.defend_predict(model, perturbed.x, perturbed.edge_index)
                originally_detected = (def_preds[test_targets] > 0).sum().item()
                evaded = ((def_preds[test_targets] > 0) & (adv_def_preds[test_targets] == 0)).sum().item()
                def_asr = evaded / max(originally_detected, 1)

                results["feature_randomization"][name][noise_std]["def_f1"].append(def_f1)
                results["feature_randomization"][name][noise_std]["def_asr"].append(def_asr)
                print(f"    FR noise={noise_std}: F1={def_f1:.4f}, ASR={def_asr:.4f}")

            # --- RobustFeatureTrainer (noise-injected training) ---
            for noise_std in NOISE_LEVELS:
                robust_model = build_model(name, emb_dim, edge_type)
                robust_model = train_robust_model(
                    robust_model, data_d, train_mask, val_mask, DEVICE,
                    noise_std=noise_std
                )
                robust_model.eval()

                # Clean accuracy with robust model
                with torch.no_grad():
                    rob_out = robust_model(data_d.x, data_d.edge_index)
                    rob_preds = rob_out.argmax(dim=1)
                    rob_f1 = f1_score(data_d.y[test_mask].cpu().numpy(), rob_preds[test_mask].cpu().numpy(), average="macro")

                # Attack robust model
                feat_attack = FeatureAttack(robust_model, perturbation_bound=5.0, n_steps=50, lr=0.01)
                perturbed, _ = feat_attack.attack(data, target_mask_full, DEVICE)

                # Evaluate ASR on robust model
                rob_asr, _ = compute_attack_success_rate(
                    robust_model, data, test_targets, perturbed, DEVICE
                )

                results["robust_training"][name][noise_std]["rob_f1"].append(rob_f1)
                results["robust_training"][name][noise_std]["rob_asr"].append(rob_asr)
                print(f"    RT noise={noise_std}: F1={rob_f1:.4f}, ASR={rob_asr:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print("FEATURE RANDOMIZATION DEFENSE SUMMARY")
    print(f"{'='*60}")
    for name in MODEL_NAMES:
        print(f"\n  {name}:")
        for noise_std in NOISE_LEVELS:
            f1_arr = np.array(results["feature_randomization"][name][noise_std]["def_f1"])
            asr_arr = np.array(results["feature_randomization"][name][noise_std]["def_asr"])
            print(f"    noise={noise_std}: F1={f1_arr.mean():.4f}+/-{f1_arr.std():.4f}, ASR={asr_arr.mean():.4f}+/-{asr_arr.std():.4f}")

    print(f"\n{'='*60}")
    print("ROBUST TRAINING SUMMARY")
    print(f"{'='*60}")
    for name in MODEL_NAMES:
        print(f"\n  {name}:")
        for noise_std in NOISE_LEVELS:
            f1_arr = np.array(results["robust_training"][name][noise_std]["rob_f1"])
            asr_arr = np.array(results["robust_training"][name][noise_std]["rob_asr"])
            print(f"    noise={noise_std}: F1={f1_arr.mean():.4f}+/-{f1_arr.std():.4f}, ASR={asr_arr.mean():.4f}+/-{asr_arr.std():.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(out_dir, f"mgtab_defense_5seeds_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_defense_eval()
