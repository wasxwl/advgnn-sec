"""
Black-box transferability evaluation on MGTAB dataset.

Tests whether adversarial examples crafted for one model transfer to others,
simulating a realistic attacker who doesn't have white-box access to the
target detector. Also evaluates the new SemanticFeatureAttack and
FeatureRandomizationDefense.

For each (source_model -> target_model) pair including self-transfer:
  1. Train both models independently
  2. Craft attack on source model using SemanticFeatureAttack
  3. Evaluate transfer ASR on target model

Also evaluates FeatureRandomizationDefense against both white-box and
black-box attacks.
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
from src.models.mgtab_models import MGTAB_RelationGCN, MGTAB_RelationGAT, MGTAB_GraphSAGE, MGTAB_RGCN, MGTAB_MLP, init_weights
from src.models.attack import FeatureAttack, SemanticFeatureAttack
from src.models.defense_advanced import FeatureRandomizationDefense
from src.utils.metrics import compute_attack_success_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")
SEEDS = [42, 123, 456, 789, 1024]
MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "RGCN", "MLP"]


class RGCNWithEdgeType(nn.Module):
    """Wrapper that stores edge_type so forward takes (x, edge_index)."""
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
    elif name == "MLP":
        return MGTAB_MLP(emb_dim, hidden_dim=256, out_dim=2, dropout=0.3)
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


def run_blackbox_eval(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Black-Box Transferability Evaluation on MGTAB ===")
    print(f"Device: {DEVICE}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Seeds: {SEEDS}")

    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"\nDataset: {desc}")

    data_d = data.to(DEVICE)
    edge_type = data.edge_type.to(DEVICE) if hasattr(data, 'edge_type') else None
    emb_dim = data.x.shape[1]

    # Results storage
    all_results = {
        "transfer_matrix": {},  # {source: {target: [asr_values]}}
        "semantic_transfer_matrix": {},  # {source: {target: [asr_values]}} for SemanticFeatureAttack
        "semantic_attack": {},  # {model: {asr, linf}}
        "feature_randomization": {},  # {model: {clean_f1, defense_asr}}
        "summary": {},
    }

    for src_name in MODEL_NAMES:
        all_results["transfer_matrix"][src_name] = {}
        all_results["semantic_transfer_matrix"][src_name] = {}

    # Per-seed transfer matrix aggregation
    transfer_accum = {s: {t: [] for t in MODEL_NAMES} for s in MODEL_NAMES}
    sem_transfer_accum = {s: {t: [] for t in MODEL_NAMES} for s in MODEL_NAMES}

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

        # Train all models for this seed
        models = {}
        for name in MODEL_NAMES:
            print(f"  Training {name}...")
            m = build_model(name, emb_dim, edge_type)
            models[name] = train_model(m, data_d, train_mask, val_mask, DEVICE)
            models[name].eval()
            with torch.no_grad():
                out = models[name](data_d.x, data_d.edge_index)
                preds = out.argmax(dim=1)
                acc = accuracy_score(data_d.y[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy())
                f1 = f1_score(data_d.y[test_mask].cpu().numpy(), preds[test_mask].cpu().numpy(), average="macro")
            print(f"    {name}: Acc={acc:.4f}, F1={f1:.4f}")

        target_mask_full = (data.y == 1)
        test_targets = target_mask_full & test_mask

        # For each source model, craft attack and evaluate on all targets
        for src_name in MODEL_NAMES:
            src_model = models[src_name]

            # White-box attack on source model (standard FeatureAttack)
            feat_attack = FeatureAttack(src_model, perturbation_bound=5.0, n_steps=50, lr=0.01)
            perturbed, _ = feat_attack.attack(data, target_mask_full, DEVICE)

            # Evaluate on all target models
            for tgt_name in MODEL_NAMES:
                tgt_model = models[tgt_name]
                asr, _ = compute_attack_success_rate(
                    tgt_model, data, test_targets, perturbed, DEVICE
                )
                transfer_accum[src_name][tgt_name].append(asr)

            # SemanticFeatureAttack: evaluate on ALL target models (cross-architecture transfer)
            sem_attack = SemanticFeatureAttack(
                src_model, perturbation_bound=5.0, n_steps=50, lr=0.01
            )
            sem_perturbed, sem_info = sem_attack.attack(data, target_mask_full, DEVICE)
            for tgt_name in MODEL_NAMES:
                tgt_model = models[tgt_name]
                sem_asr, _ = compute_attack_success_rate(
                    tgt_model, data, test_targets, sem_perturbed, DEVICE
                )
                sem_transfer_accum[src_name][tgt_name].append(sem_asr)
            print(f"  {src_name}: Semantic self-ASR={sem_transfer_accum[src_name][src_name][-1]:.4f} (Linf={sem_info.get('perturbation_linf', 0):.4f})")

        # FeatureRandomizationDefense evaluation (on source models)
        if seed == SEEDS[0]:
            print(f"\n  Evaluating FeatureRandomizationDefense...")
            for name in MODEL_NAMES:
                model = models[name]
                # Evaluate defense at multiple noise levels
                for noise_std in [0.05, 0.1, 0.2]:
                    defense = FeatureRandomizationDefense(noise_std=noise_std, n_samples=5)
                    _, def_preds = defense.defend_predict(model, data_d.x, data_d.edge_index)

                    # Clean accuracy with defense
                    def_acc = accuracy_score(data_d.y[test_mask].cpu().numpy(), def_preds[test_mask].cpu().numpy())
                    def_f1 = f1_score(data_d.y[test_mask].cpu().numpy(), def_preds[test_mask].cpu().numpy(), average="macro")

                    # Attack defended model: craft attack on clean model, evaluate with defense
                    feat_attack = FeatureAttack(model, perturbation_bound=5.0, n_steps=50, lr=0.01)
                    perturbed, _ = feat_attack.attack(data, target_mask_full, DEVICE)

                    # Evaluate with defense: predict with noisy averaging on clean and perturbed data
                    with torch.no_grad():
                        _, clean_def_preds = defense.defend_predict(model, data_d.x, data_d.edge_index)
                        _, adv_def_preds = defense.defend_predict(model, perturbed.x, perturbed.edge_index)

                    originally_detected = (clean_def_preds[test_targets] > 0).sum().item()
                    evaded = ((clean_def_preds[test_targets] > 0) & (adv_def_preds[test_targets] == 0)).sum().item()
                    def_asr_manual = evaded / max(originally_detected, 1)

                    if name not in all_results["feature_randomization"]:
                        all_results["feature_randomization"][name] = {}
                    if noise_std not in all_results["feature_randomization"][name]:
                        all_results["feature_randomization"][name][noise_std] = {"def_f1": [], "def_asr": []}
                    all_results["feature_randomization"][name][noise_std]["def_f1"].append(def_f1)
                    all_results["feature_randomization"][name][noise_std]["def_asr"].append(def_asr_manual)

                    print(f"    {name} noise={noise_std}: Clean F1={def_f1:.4f}, Def ASR={def_asr_manual:.4f}")

    # Build transfer matrix with mean +/- std
    print(f"\n{'='*60}")
    print("TRANSFER MATRIX (ASR %)")
    print(f"{'='*60}")
    print(f"{'Source -> Target':<20}", end="")
    for tgt in MODEL_NAMES:
        print(f"{tgt:>12}", end="")
    print()
    print(f"{'-'*68}")

    for src in MODEL_NAMES:
        print(f"{src:<20}", end="")
        for tgt in MODEL_NAMES:
            vals = np.array(transfer_accum[src][tgt]) * 100
            mean = vals.mean()
            std = vals.std()
            all_results["transfer_matrix"][src][tgt] = f"{mean:.1f} +/- {std:.1f}"
            print(f"{mean:.1f}+/-{std:.1f}", end="")
            if tgt == "GCN":
                pass  # self-transfer, no marker needed
        print()

    # Build semantic transfer matrix
    for src in MODEL_NAMES:
        for tgt in MODEL_NAMES:
            vals = np.array(sem_transfer_accum[src][tgt]) * 100
            mean = vals.mean()
            std = vals.std()
            all_results["semantic_transfer_matrix"][src][tgt] = f"{mean:.1f} +/- {std:.1f}"

    # Semantic FeatureAttack transfer matrix
    print(f"\n{'='*60}")
    print("SEMANTIC FEATURE ATTACK TRANSFER MATRIX (ASR %)")
    print(f"{'='*60}")
    print(f"{'Source -> Target':<20}", end="")
    for tgt in MODEL_NAMES:
        print(f"{tgt:>12}", end="")
    print()
    print(f"{'-'*68}")

    for src in MODEL_NAMES:
        print(f"{src:<20}", end="")
        for tgt in MODEL_NAMES:
            vals = np.array(sem_transfer_accum[src][tgt]) * 100
            mean = vals.mean()
            std = vals.std()
            print(f"{mean:.1f}+/-{std:.1f}", end="")
        print()

    # Summary
    print(f"\n{'='*60}")
    print("SEMANTIC ATTACK SUMMARY (self-transfer)")
    print(f"{'='*60}")
    for name in MODEL_NAMES:
        vals = np.array(sem_transfer_accum[name][name]) * 100
        print(f"  {name}: ASR={vals.mean():.1f}+/-{vals.std():.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(out_dir, f"mgtab_blackbox_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_blackbox_eval()
