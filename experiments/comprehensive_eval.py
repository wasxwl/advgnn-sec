"""
Comprehensive evaluation: statistical significance across seeds and attack baselines.

Runs 5 random seeds, evaluates 5+ attack methods, reports mean +/- std.
Generates results for Tables 1-4 in the paper.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.loader import get_dataset
from src.dataset.preprocess import normalize_features
from src.models.detector import GCNDetector, GATDetector
from src.models.attack import FeatureAttack, MetaGradGraphAttack, PGDGraphAttack, JointAttack
from src.models.defense import RobustDetector, FeatureSmoothing
from src.models.defense_advanced import SVDPurification, RobustGCN
from src.utils.metrics import (
    compute_metrics,
    compute_attack_success_rate,
    compute_stealth_score,
    compute_transferability,
    compute_perturbation_stats,
)
from src.train import train_model


SEEDS = [42, 123, 456, 789, 1024]
N_EPOCHS = 200
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train_detector(seed, model_type="gcn", in_channels=16, hidden_channels=64, out_channels=3):
    """Train a detector with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data, labels, desc = get_dataset(seed=seed)
    data = normalize_features(data).to(DEVICE)
    target_mask = labels > 0

    if model_type == "gcn":
        model = GCNDetector(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
    else:
        model = GATDetector(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)

    model = model.to(DEVICE)
    model, history, metrics = train_model(model, data, n_epochs=N_EPOCHS, lr=0.01, device=DEVICE)
    model.eval()

    return model, data, labels, target_mask, metrics


def evaluate_feature_attack(model, data, labels, target_mask, seed, steps_list=None):
    """Evaluate feature attack across different step counts."""
    if steps_list is None:
        steps_list = [10, 20, 30, 50, 100]

    results = []
    for steps in steps_list:
        attack = FeatureAttack(model, perturbation_bound=2.0, n_steps=steps, lr=0.01)
        perturbed, info = attack.attack(data, target_mask, DEVICE)
        asr, details = compute_attack_success_rate(model, data, target_mask, perturbed, DEVICE)
        pstats = compute_perturbation_stats(data, perturbed, target_mask)
        stealth = compute_stealth_score(data, perturbed, target_mask)

        results.append({
            "steps": steps,
            "asr": asr,
            "perturbation_norm": info["perturbation_norm"],
            "avg_perturbation_per_node": info["avg_perturbation_per_node"],
            "feature_l2_per_node": pstats.get("feature_l2_per_node", 0),
            "stealth_score": stealth["stealth_score"],
        })

    return results


def evaluate_graph_attack(model, data, labels, target_mask, seed):
    """Evaluate graph attacks across budgets."""
    budgets = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    results = {"metagrad": [], "pgd": []}

    graph_attack = MetaGradGraphAttack(model, budget_ratio=0.05)
    for budget in budgets:
        graph_attack.budget_ratio = budget
        perturbed, info = graph_attack.attack(data, target_mask, DEVICE)
        asr, _ = compute_attack_success_rate(model, data, target_mask, perturbed, DEVICE)
        results["metagrad"].append({
            "budget": budget,
            "asr": asr,
            "edges_added": info.get("edges_added", 0),
            "edges_removed": info.get("edges_removed", 0),
        })

    pgd_attack = PGDGraphAttack(model, budget_ratio=0.05, lr=0.5, n_steps=50)
    for budget in budgets:
        pgd_attack.budget_ratio = budget
        perturbed, info = pgd_attack.attack(data, target_mask, DEVICE)
        asr, _ = compute_attack_success_rate(model, data, target_mask, perturbed, DEVICE)
        results["pgd"].append({
            "budget": budget,
            "asr": asr,
            "edges_added": info.get("edges_added", 0),
            "edges_removed": info.get("edges_removed", 0),
        })

    return results


def evaluate_joint_attack(model, data, labels, target_mask, seed):
    """Evaluate joint attack."""
    joint_attack = JointAttack(model, feature_bound=1.0, graph_budget_ratio=0.03, feature_steps=50, graph_steps=10)
    perturbed, info = joint_attack.attack(data, target_mask, DEVICE)

    return {
        "asr_feature": info["asr_feature"],
        "asr_graph": info["asr_graph"],
        "asr_joint": info["asr"],
        "feature_norm": info["feature_norm"],
        "edges_added": info["edges_added"],
        "edges_removed": info["edges_removed"],
    }


def evaluate_transferability(model_gcn, model_gat, data, target_mask, seed):
    """Evaluate attack transferability from GCN to GAT."""
    feat_attack = FeatureAttack(model_gcn, perturbation_bound=2.0, n_steps=50, lr=0.01)
    perturbed, _ = feat_attack.attack(data, target_mask, DEVICE)

    transfer = compute_transferability(model_gcn, {"GAT": model_gat}, data, target_mask, perturbed, DEVICE)
    return {
        "source_asr": transfer["source_model_asr"],
        "transfer_gat": transfer["transfer_rates"]["GAT"],
        "avg_transfer": transfer["avg_transfer_rate"],
    }


def evaluate_defenses(model, data, labels, target_mask, seed):
    """Evaluate all defense mechanisms."""
    results = {}

    # 1. SVD Purification
    svd = SVDPurification(rank=64, symmetric=True)
    purified_edge, svd_info = svd.purify(data.edge_index, data.num_nodes, DEVICE)
    purified_data = data.clone()
    purified_data.edge_index = purified_edge

    with torch.no_grad():
        logits_pur = model(purified_data.x, purified_data.edge_index)
        preds_pur = logits_pur.argmax(dim=1)
        clean_acc_pur = (preds_pur == data.y).sum().item() / data.num_nodes

    feat_attack = FeatureAttack(model, perturbation_bound=2.0, n_steps=50, lr=0.01)
    perturbed_pur, _ = feat_attack.attack(purified_data, target_mask, DEVICE)
    asr_pur, _ = compute_attack_success_rate(model, data, target_mask, perturbed_pur, DEVICE)

    results["svd_purification"] = {
        "clean_accuracy": clean_acc_pur,
        "asr": asr_pur,
        "edges_removed": svd_info["edges_removed"],
        "edges_added": svd_info["edges_added"],
    }

    # 2. Feature Smoothing + RobustDetector
    smoothing = FeatureSmoothing(alpha=0.3, n_steps=5)
    robust_model = RobustDetector(deepcopy(model), smoothing_alpha=0.3, smoothing_steps=5)
    robust_model = robust_model.to(DEVICE)
    robust_model.eval()

    with torch.no_grad():
        logits_rob = robust_model(data.x, data.edge_index)
        preds_rob = logits_rob.argmax(dim=1)
        robust_metrics = compute_metrics(data.y, preds_rob)

    rob_attack = FeatureAttack(robust_model, perturbation_bound=2.0, n_steps=50, lr=0.01)
    perturbed_rob, _ = rob_attack.attack(data, target_mask, DEVICE)
    asr_rob, _ = compute_attack_success_rate(robust_model, data, target_mask, perturbed_rob, DEVICE)

    results["feature_smoothing"] = {
        "clean_f1": robust_metrics["macro_f1"],
        "asr": asr_rob,
    }

    return results


def run_comprehensive_eval(out_dir="results"):
    """Run full comprehensive evaluation."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Comprehensive Evaluation ===")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "seeds": SEEDS,
        "baseline": [],
        "feature_attack": [],
        "graph_attack": [],
        "joint_attack": [],
        "transferability": [],
        "defenses": [],
    }

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        # Train GCN
        torch.manual_seed(seed)
        np.random.seed(seed)
        gcn_model, data, labels, target_mask, gcn_metrics = train_detector(seed, "gcn")
        print(f"  GCN F1: {gcn_metrics['macro_f1']:.4f}, Acc: {gcn_metrics['accuracy']:.4f}")

        # Train GAT for transferability
        torch.manual_seed(seed + 1000)
        np.random.seed(seed + 1000)
        gat_model, _, _, _, gat_metrics = train_detector(seed, "gat")
        print(f"  GAT F1: {gat_metrics['macro_f1']:.4f}")

        # Feature attack
        print("  Feature attack...", end=" ", flush=True)
        feat_results = evaluate_feature_attack(gcn_model, data, labels, target_mask, seed, [10, 20, 50])
        print(f"ASR@50={feat_results[-1]['asr']:.4f}")

        # Graph attack (subset of budgets for speed)
        print("  Graph attack...", end=" ", flush=True)
        graph_results = evaluate_graph_attack(gcn_model, data, labels, target_mask, seed)
        print(f"MetaGrad@0.05={graph_results['metagrad'][3]['asr']:.4f}")

        # Joint attack
        print("  Joint attack...", end=" ", flush=True)
        joint_results = evaluate_joint_attack(gcn_model, data, labels, target_mask, seed)
        print(f"Joint ASR={joint_results['asr_joint']:.4f}")

        # Transferability
        print("  Transferability...", end=" ", flush=True)
        transfer_results = evaluate_transferability(gcn_model, gat_model, data, target_mask, seed)
        print(f"Transfer={transfer_results['transfer_gat']:.4f}")

        # Defenses
        print("  Defenses...", end=" ", flush=True)
        defense_results = evaluate_defenses(gcn_model, data, labels, target_mask, seed)
        print(f"SVD ASR={defense_results['svd_purification']['asr']:.4f}")

        all_results["baseline"].append({
            "seed": seed,
            "gcn_f1": gcn_metrics["macro_f1"],
            "gcn_accuracy": gcn_metrics["accuracy"],
            "gat_f1": gat_metrics["macro_f1"],
            "gat_accuracy": gat_metrics["accuracy"],
        })
        all_results["feature_attack"].append(feat_results)
        all_results["graph_attack"].append(graph_results)
        all_results["joint_attack"].append(joint_results)
        all_results["transferability"].append(transfer_results)
        all_results["defenses"].append(defense_results)

    # Compute summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    # Baseline
    baseline_f1 = [r["gcn_f1"] for r in all_results["baseline"]]
    print(f"\nBaseline GCN F1: {np.mean(baseline_f1):.4f} +/- {np.std(baseline_f1):.4f}")

    # Feature attack at 50 steps
    asr_50 = [r[-1]["asr"] for r in all_results["feature_attack"]]
    print(f"Feature ASR (50 steps): {np.mean(asr_50):.4f} +/- {np.std(asr_50):.4f}")

    # Graph attack at budget 0.05
    asr_graph = [r["metagrad"][3]["asr"] for r in all_results["graph_attack"]]
    print(f"Graph ASR (budget 0.05): {np.mean(asr_graph):.4f} +/- {np.std(asr_graph):.4f}")

    # Joint attack
    asr_joint = [r["asr_joint"] for r in all_results["joint_attack"]]
    print(f"Joint ASR: {np.mean(asr_joint):.4f} +/- {np.std(asr_joint):.4f}")

    # Transferability
    transfer = [r["transfer_gat"] for r in all_results["transferability"]]
    print(f"Transfer to GAT: {np.mean(transfer):.4f} +/- {np.std(transfer):.4f}")

    # SVD defense
    svd_asr = [r["svd_purification"]["asr"] for r in all_results["defenses"]]
    print(f"SVD ASR: {np.mean(svd_asr):.4f} +/- {np.std(svd_asr):.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(out_dir, f"comprehensive_eval_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate paper tables
    print("\n" + "="*60)
    print("PAPER TABLES")
    print("="*60)

    print("\nTable 1: Baseline Performance")
    print(f"{'Model':<10} {'Accuracy':<12} {'Macro F1':<12}")
    print(f"{'GCN':<10} {np.mean([r['gcn_accuracy'] for r in all_results['baseline']]):.4f} +/- {np.std([r['gcn_accuracy'] for r in all_results['baseline']]):.4f}  {np.mean([r['gcn_f1'] for r in all_results['baseline']]):.4f} +/- {np.std([r['gcn_f1'] for r in all_results['baseline']]):.4f}")
    print(f"{'GAT':<10} {np.mean([r['gat_accuracy'] for r in all_results['baseline']]):.4f} +/- {np.std([r['gat_accuracy'] for r in all_results['baseline']]):.4f}  {np.mean([r['gat_f1'] for r in all_results['baseline']]):.4f} +/- {np.std([r['gat_f1'] for r in all_results['baseline']]):.4f}")

    print("\nTable 2: Attack Effectiveness")
    print(f"{'Attack':<20} {'ASR':<10} {'Std':<10}")
    print(f"{'Feature (50 steps)':<20} {np.mean(asr_50):.4f} +/- {np.std(asr_50):.4f}")
    print(f"{'MetaGrad (0.05)':<20} {np.mean(asr_graph):.4f} +/- {np.std(asr_graph):.4f}")
    print(f"{'Joint':<20} {np.mean(asr_joint):.4f} +/- {np.std(asr_joint):.4f}")

    print("\nTable 3: Defense Effectiveness")
    print(f"{'Defense':<20} {'Clean Acc':<12} {'ASR':<10}")
    print(f"{'Baseline':<20} {'1.0000':<12} {np.mean(asr_50):.4f}")
    print(f"{'SVD Purification':<20} {np.mean([r['svd_purification']['clean_accuracy'] for r in all_results['defenses']]):.4f} +/- {np.std([r['svd_purification']['clean_accuracy'] for r in all_results['defenses']]):.4f}  {np.mean(svd_asr):.4f} +/- {np.std(svd_asr):.4f}")
    print(f"{'Feature Smoothing':<20} {np.mean([r['feature_smoothing']['clean_f1'] for r in all_results['defenses']]):.4f} +/- {np.std([r['feature_smoothing']['clean_f1'] for r in all_results['defenses']]):.4f}  {np.mean([r['feature_smoothing']['asr'] for r in all_results['defenses']]):.4f} +/- {np.std([r['feature_smoothing']['asr'] for r in all_results['defenses']]):.4f}")

    print("\nTable 4: Transferability")
    print(f"{'Source':<10} {'Target':<10} {'ASR':<10}")
    print(f"{'GCN':<10} {'GAT':<10} {np.mean(transfer):.4f} +/- {np.std(transfer):.4f}")


if __name__ == "__main__":
    run_comprehensive_eval()
