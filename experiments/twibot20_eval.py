"""
TwiBot-20 validation: confirm AdvGNN-Sec findings on a second real-world dataset.

TwiBot-20: 11,826 Twitter users, 16,908 edges, 2 relation types,
13-dim profile features extracted from text metadata.
Pre-split train/val/test (8278/2365/1183).

This is a focused validation -- not a full re-evaluation.
Tests: (1) clean accuracy, (2) feature attack ASR, (3) one defense failure.
Results reported on single seed (matching the provided split).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.twibot20_loader import TwiBot20Dataset
from src.models.mgtab_models import init_weights
from src.models.attack import FeatureAttack
from src.models.defense_advanced import TRADESAdversarialTrainer
from src.utils.metrics import compute_attack_success_rate

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Twibot-20")
MODEL_NAMES = ["GCN", "GAT", "GraphSAGE", "RGCN"]

# TwiBot-20 has 13-dim features, use smaller hidden dim
HIDDEN_DIM = 64


class TwiBot20_GCN(nn.Module):
    def __init__(self, emb_dim, hidden=HIDDEN_DIM, out_dim=2, dropout=0.3):
        super().__init__()
        self.fc_in = nn.Linear(emb_dim, hidden)
        self.gcn1 = GCNConv(hidden, hidden)
        self.fc_mid = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.fc_in(x))
        x = F.leaky_relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.fc_mid(x))
        return self.fc_out(x)


class TwiBot20_GAT(nn.Module):
    def __init__(self, emb_dim, hidden=HIDDEN_DIM, out_dim=2, dropout=0.3):
        super().__init__()
        self.fc_in = nn.Linear(emb_dim, hidden)
        self.gat1 = GATConv(hidden, hidden // 4, heads=4, dropout=dropout)
        self.fc_mid = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.fc_in(x))
        x = F.leaky_relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.fc_mid(x))
        return self.fc_out(x)


class TwiBot20_SAGE(nn.Module):
    def __init__(self, emb_dim, hidden=HIDDEN_DIM, out_dim=2, dropout=0.3):
        super().__init__()
        self.fc_in = nn.Linear(emb_dim, hidden)
        self.sage1 = SAGEConv(hidden, hidden)
        self.fc_mid = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.fc_in(x))
        x = F.leaky_relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.fc_mid(x))
        return self.fc_out(x)


class TwiBot20_RGCN(nn.Module):
    def __init__(self, emb_dim, hidden=HIDDEN_DIM, out_dim=2, num_relations=2, dropout=0.3):
        super().__init__()
        self.fc_in = nn.Linear(emb_dim, hidden)
        self.rgcn1 = RGCNConv(hidden, hidden, num_relations)
        self.fc_mid = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = F.leaky_relu(self.fc_in(x))
        x = F.leaky_relu(self.rgcn1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.fc_mid(x))
        return self.fc_out(x)


class RGCNForAttack(nn.Module):
    """Wrap RGCN to accept (x, edge_index) from attack code."""
    def __init__(self, rgcn, edge_type):
        super().__init__()
        self.rgcn = rgcn
        self.edge_type = edge_type

    def forward(self, x, edge_index):
        return self.rgcn(x, edge_index, self.edge_type)


def build_model(name, emb_dim, edge_type=None, out_dim=2, hidden=HIDDEN_DIM):
    if name == "GCN":
        return TwiBot20_GCN(emb_dim, hidden, out_dim)
    elif name == "GAT":
        return TwiBot20_GAT(emb_dim, hidden, out_dim)
    elif name == "GraphSAGE":
        return TwiBot20_SAGE(emb_dim, hidden, out_dim)
    elif name == "RGCN":
        num_relations = int(edge_type.max().item()) + 1 if edge_type is not None else 2
        rgcn = TwiBot20_RGCN(emb_dim, hidden, out_dim, num_relations)
        return RGCNForAttack(rgcn, edge_type)
    raise ValueError(f"Unknown model: {name}")


def train_model(model, data, train_mask, val_mask, device, epochs=100, lr=1e-3, weight_decay=5e-4):
    """Standard training loop."""
    model = model.to(device)
    # Class-weighted loss for imbalanced labels
    class_counts = torch.bincount(data.y[train_mask]).float()
    class_weights = len(data.y[train_mask]) / (2 * class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_f1 = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = nn.CrossEntropyLoss()(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)
            val_f1 = f1_score(
                data.y[val_mask].cpu().numpy(),
                preds[val_mask].cpu().numpy(),
                average="macro",
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_f1


def run_twibot20_eval(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== TwiBot-20 Validation ===")
    print(f"Device: {DEVICE}")

    data, labels, desc = TwiBot20Dataset(DATA_DIR).load()
    print(f"\nDataset: {desc}")

    data_d = data.to(DEVICE)
    edge_type = data.edge_type.to(DEVICE) if hasattr(data, 'edge_type') else None
    emb_dim = data.x.shape[1]

    # Use provided split
    train_idx = data.train_idx
    val_idx = data.valid_idx
    test_idx = data.test_idx

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
    train_mask[train_idx] = True
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
    val_mask[val_idx] = True
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
    test_mask[test_idx] = True

    # Bot mask for attack targets
    bot_mask = (data.y == 1).to(DEVICE)
    test_bot_mask = bot_mask & test_mask

    results = {}

    for name in MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"=== {name} ===")
        print(f"{'='*60}")

        torch.manual_seed(42)
        np.random.seed(42)

        # Train clean model
        model = build_model(name, emb_dim, edge_type)
        model.apply(init_weights)
        model, val_f1 = train_model(model, data_d, train_mask, val_mask, DEVICE)
        model.eval()

        # Clean F1
        with torch.no_grad():
            out = model(data_d.x, data_d.edge_index)
            preds = out.argmax(dim=1)
            clean_acc = accuracy_score(
                data_d.y[test_mask].cpu().numpy(),
                preds[test_mask].cpu().numpy(),
            )
            clean_f1 = f1_score(
                data_d.y[test_mask].cpu().numpy(),
                preds[test_mask].cpu().numpy(),
                average="macro",
            )

        print(f"  Clean: Acc={clean_acc:.4f}, F1={clean_f1:.4f}, Val F1={val_f1:.4f}")

        # Feature attack at eps=0.5
        attack05 = FeatureAttack(model, perturbation_bound=0.5, n_steps=30, lr=0.01)
        perturbed05, _ = attack05.attack(data, bot_mask, DEVICE)
        asr05, _ = compute_attack_success_rate(model, data, test_bot_mask, perturbed05, DEVICE)

        # Feature attack at eps=5.0
        attack5 = FeatureAttack(model, perturbation_bound=5.0, n_steps=30, lr=0.01)
        perturbed5, _ = attack5.attack(data, bot_mask, DEVICE)
        asr5, _ = compute_attack_success_rate(model, data, test_bot_mask, perturbed5, DEVICE)

        print(f"  ASR (eps=0.5)={asr05:.4f}, ASR (eps=5.0)={asr5:.4f}")

        results[name] = {
            "clean_acc": clean_acc,
            "clean_f1": clean_f1,
            "asr_eps0.5": asr05,
            "asr_eps5.0": asr5,
        }

        # TRADES defense on first model only (GCN) to confirm defense failure
        if name == "GCN":
            print(f"\n  TRADES (beta=6.0):")
            trades_model = build_model("GCN", emb_dim, edge_type)
            trades_model.apply(init_weights)
            trades = TRADESAdversarialTrainer(
                trades_model, perturbation_bound=5.0, n_steps=10, lr=0.01, beta=6.0
            )
            trained, history = trades.train(
                data_d, train_mask, val_mask, DEVICE,
                epochs=50, lr=1e-3, weight_decay=5e-4,
            )
            trained.eval()

            with torch.no_grad():
                out = trained(data_d.x, data_d.edge_index)
                preds = out.argmax(dim=1)
                trades_acc = accuracy_score(
                    data_d.y[test_mask].cpu().numpy(),
                    preds[test_mask].cpu().numpy(),
                )
                trades_f1 = f1_score(
                    data_d.y[test_mask].cpu().numpy(),
                    preds[test_mask].cpu().numpy(),
                    average="macro",
                )

            attack5_trades = FeatureAttack(trained, perturbation_bound=5.0, n_steps=50, lr=0.01)
            perturbed_trades, _ = attack5_trades.attack(data, bot_mask, DEVICE)
            asr_trades, _ = compute_attack_success_rate(
                trained, data, test_bot_mask, perturbed_trades, DEVICE
            )

            print(f"    Clean: Acc={trades_acc:.4f}, F1={trades_f1:.4f}")
            print(f"    ASR (eps=5.0)={asr_trades:.4f}")

            results["GCN_TRADES"] = {
                "clean_acc": trades_acc,
                "clean_f1": trades_f1,
                "asr_eps5.0": asr_trades,
            }

    # Print summary
    print(f"\n{'='*60}")
    print("TWIBOT-20 SUMMARY")
    print(f"{'='*60}")
    print(f"\n| Model       | Clean Acc | Clean F1 | ASR (eps=0.5) | ASR (eps=5.0) |")
    print(f"|-------------|-----------|----------|---------------|---------------|")
    for name in MODEL_NAMES:
        r = results[name]
        print(f"| {name:11s} | {r['clean_acc']:.4f}    | {r['clean_f1']:.4f}  | {r['asr_eps0.5']:.4f}       | {r['asr_eps5.0']:.4f}       |")
    if "GCN_TRADES" in results:
        r = results["GCN_TRADES"]
        print(f"| {'GCN+TRADES':11s} | {r['clean_acc']:.4f}    | {r['clean_f1']:.4f}  | ---           | {r['asr_eps5.0']:.4f}       |")


if __name__ == "__main__":
    run_twibot20_eval()
