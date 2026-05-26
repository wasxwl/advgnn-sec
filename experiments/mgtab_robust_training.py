"""
RobustFeatureTrainer experiments on MGTAB: noise-injected training vs white-box feature attacks.

For each model (GCN, GAT, GraphSAGE, RGCN), trains with RobustFeatureTrainer at
noise_std=[0.05, 0.1, 0.2], then evaluates with FeatureAttack (PGD, eps=5.0).

Key: the attacker has full white-box access to the trained model but NOT to the
noise injection (which only happens at training time, not inference).
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.mgtab_loader import MGTABDataset
from src.models.mgtab_models import MGTAB_GCN, MGTAB_GAT, MGTAB_GraphSAGE, MGTAB_RGCN, init_weights
from src.models.attack import FeatureAttack
from src.models.defense_advanced import RobustFeatureTrainer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mgtab")
SEED = 42
NOISE_LEVELS = [0.05, 0.1, 0.2]
ATTACK_EPS = 5.0
TRAIN_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 5e-4


class RGCNWithEdgeType(nn.Module):
    """Wrapper that stores edge_type so forward takes (x, edge_index)."""
    def __init__(self, rgcn, edge_type):
        super().__init__()
        self.base_model = rgcn  # keep as base_model for _get_base_model in attack.py
        self.edge_type = edge_type

    def forward(self, x, edge_index):
        return self.base_model(x, edge_index, self.edge_type)


def make_model(name, emb_dim, edge_type=None):
    """Create model instance."""
    if name == "GCN":
        return MGTAB_GCN(emb_dim, hidden_dim=256, out_dim=2, dropout=0.3)
    elif name == "GAT":
        return MGTAB_GAT(emb_dim, hidden_dim=256, out_dim=2, dropout=0.3)
    elif name == "GraphSAGE":
        return MGTAB_GraphSAGE(emb_dim, hidden_dim=256, out_dim=2, dropout=0.3)
    elif name == "RGCN":
        rgcn = MGTAB_RGCN(emb_dim, hidden_dim=256, out_dim=2, num_relations=7, dropout=0.3)
        return RGCNWithEdgeType(rgcn, edge_type)
    else:
        raise ValueError(f"Unknown model: {name}")


def make_masks(data, rng):
    """Create 70/20/10 train/val/test masks."""
    n = data.num_nodes
    idx = rng.permutation(n)
    train_idx = idx[:int(0.7 * n)]
    val_idx = idx[int(0.7 * n):int(0.9 * n)]
    test_idx = idx[int(0.9 * n):]

    train_mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    val_mask[val_idx] = True
    test_mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def evaluate_clean(model, data, test_mask):
    """Evaluate clean F1 on test set."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
    test_f1 = f1_score(
        data.y[test_mask].cpu().numpy(),
        preds[test_mask].cpu().numpy(),
        average="macro"
    )
    return test_f1


def evaluate_attack(model, data, test_mask):
    """Run FeatureAttack on bots and compute ASR on test set."""
    # Target bots that are in the test set
    target_mask = (data.y == 1) & test_mask

    attack = FeatureAttack(
        model, perturbation_bound=ATTACK_EPS, n_steps=50, lr=0.01
    )
    perturbed, info = attack.attack(data, target_mask, DEVICE)

    # Compute ASR: among bots that were originally detected and are in test set,
    # what fraction are now misclassified?
    model.eval()
    with torch.no_grad():
        logits_orig = model(data.x, data.edge_index)
        preds_orig = logits_orig.argmax(dim=1)

        logits_adv = model(perturbed.x, perturbed.edge_index)
        preds_adv = logits_adv.argmax(dim=1)

    target_nodes = target_mask.nonzero(as_tuple=True)[0]
    if len(target_nodes) == 0:
        return 0.0, info

    detected_orig = ((preds_orig[target_nodes] > 0)).sum().item()
    evaded = ((preds_orig[target_nodes] > 0) & (preds_adv[target_nodes] == 0)).sum().item()
    asr = evaded / max(detected_orig, 1)

    return asr, info


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== RobustFeatureTrainer Experiments on MGTAB ===")
    print(f"Device: {DEVICE}, Seed: {SEED}")
    print(f"Attack: FeatureAttack (PGD), eps={ATTACK_EPS}, n_steps=50, lr=0.01")
    print()

    # Load data
    data, labels, desc = MGTABDataset(DATA_DIR).load()
    print(f"Dataset: {desc}")
    print(f"Humans: {(data.y == 0).sum().item()}, Bots: {(data.y == 1).sum().item()}")
    print()

    # Create masks
    rng = np.random.RandomState(SEED)
    train_mask, val_mask, test_mask = make_masks(data, rng)

    # Move data to device
    data_d = data.to(DEVICE)
    edge_type = data.edge_type.to(DEVICE) if hasattr(data, 'edge_type') else None

    emb_dim = data.x.shape[1]
    model_names = ["GCN", "GAT", "GraphSAGE", "RGCN"]

    # Results table
    results = []

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for noise_std in NOISE_LEVELS:
            print(f"\n  Training with noise_std={noise_std} ...")

            # Reset seeds for reproducibility
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            model = make_model(model_name, emb_dim, edge_type)

            trainer = RobustFeatureTrainer(
                noise_std=noise_std, noise_schedule="constant"
            )
            trained_model, history = trainer.train_robust(
                model, data_d, train_mask, val_mask,
                epochs=TRAIN_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE
            )

            # Clean evaluation
            clean_f1 = evaluate_clean(trained_model, data_d, test_mask)

            # Attack evaluation
            asr, attack_info = evaluate_attack(trained_model, data_d, test_mask)

            pert_norm = attack_info.get("perturbation_norm", 0)

            print(f"  Clean F1: {clean_f1:.4f}  |  ASR: {asr:.4f}  |  Pert norm: {pert_norm:.4f}")

            results.append({
                "model": model_name,
                "noise_std": noise_std,
                "clean_f1": clean_f1,
                "asr": asr,
                "perturbation_norm": pert_norm,
            })

    # Summary table
    print(f"\n\n{'='*70}")
    print("ROBUST TRAINING RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Noise Std':<12} {'Clean F1':<12} {'ASR (eps=5)':<14} {'Pert Norm':<12}")
    print(f"{'-'*70}")

    for r in results:
        print(
            f"{r['model']:<12} {r['noise_std']:<12.2f} "
            f"{r['clean_f1']:<12.4f} {r['asr']:<14.4f} {r['perturbation_norm']:<12.4f}"
        )

    # Also show clean baseline (no noise) for comparison
    print(f"\n{'='*70}")
    print("BASELINE (no noise injection, standard training)")
    print(f"{'='*70}")

    for model_name in model_names:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = make_model(model_name, emb_dim, edge_type)
        model = model.to(DEVICE)
        model.apply(init_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_acc = 0.0
        best_state = None
        for epoch in range(TRAIN_EPOCHS):
            model.train()
            optimizer.zero_grad()
            output = model(data_d.x, data_d.edge_index)
            loss = criterion(output[train_mask], data_d.y[train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_out = model(data_d.x, data_d.edge_index)
                    val_preds = val_out.argmax(dim=1)
                    from sklearn.metrics import accuracy_score
                    val_acc = accuracy_score(
                        data_d.y[val_mask].cpu().numpy(),
                        val_preds[val_mask].cpu().numpy()
                    )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        clean_f1 = evaluate_clean(model, data_d, test_mask)
        asr, _ = evaluate_attack(model, data_d, test_mask)

        print(
            f"{model_name:<12} {'0.00 (baseline)':<12} "
            f"{clean_f1:<12.4f} {asr:<14.4f}"
        )


if __name__ == "__main__":
    # Capture stdout to file as well
    import io
    out_file = os.path.join(os.environ.get("TEMP", "/tmp"), "robust_training_results.txt")
    with open(out_file, "w") as f:
        # Use a tee-like approach
        class Tee:
            def __init__(self, file, stdout):
                self.file = file
                self.stdout = stdout
            def write(self, msg):
                self.file.write(msg)
                self.stdout.write(msg)
            def flush(self):
                self.file.flush()
                self.stdout.flush()

        original_stdout = sys.stdout
        tee = Tee(f, original_stdout)
        sys.stdout = tee

        try:
            main()
        finally:
            sys.stdout = original_stdout

    print(f"\nResults also saved to {out_file}")
