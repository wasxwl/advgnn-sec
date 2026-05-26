"""
Adaptive evaluation: full arms race simulation.

Simulates an adaptive attacker who knows about defenses and adapts:
1. Train clean detector
2. Attack it -> get ASR
3. Add defense -> train defended detector
4. Attack defended detector -> get adapted ASR
5. Repeat with stronger attacks

This models a realistic attacker-defender dynamic rather than
static attack/defense evaluation.
"""

import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.loader import get_dataset
from src.dataset.preprocess import normalize_features
from src.models.detector import GCNDetector
from src.models.attack import FeatureAttack, JointAttack
from src.models.defense import AdversarialTraining
from src.models.defense_advanced import SVDPurification, EnsembleDetector
from src.utils.metrics import compute_attack_success_rate, compute_metrics
from src.train import train_model


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def run_adaptive_eval(n_rounds=5, seed=42, out_dir="results"):
    """Run adaptive arms race simulation.

    Args:
        n_rounds: number of attack-defense rounds
        seed: random seed
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    print(f"=== Adaptive Arms Race Evaluation ===")
    print(f"Device: {DEVICE}, Seed: {seed}, Rounds: {n_rounds}")

    # Load data
    data, labels, desc = get_dataset(seed=seed)
    data = normalize_features(data).to(DEVICE)
    target_mask = labels > 0

    history = []

    # Round 0: Clean baseline
    print(f"\nRound 0: Training clean detector...")
    model = GCNDetector(in_channels=16, hidden_channels=64, out_channels=3)
    model = model.to(DEVICE)
    model, _, clean_metrics = train_model(model, data, n_epochs=200, lr=0.01, device=DEVICE)
    model.eval()

    print(f"  Clean F1: {clean_metrics['macro_f1']:.4f}")
    history.append({
        "round": 0,
        "stage": "baseline",
        "model_f1": clean_metrics["macro_f1"],
        "asr_feature": 0.0,
        "asr_joint": 0.0,
    })

    # Round 1: Attack clean model
    print(f"\nRound 1: Attacking clean model...")
    feat_attack = FeatureAttack(model, perturbation_bound=2.0, n_steps=50, lr=0.01)
    perturbed, feat_info = feat_attack.attack(data, target_mask, DEVICE)
    asr_feature, _ = compute_attack_success_rate(model, data, target_mask, perturbed, DEVICE)
    print(f"  Feature ASR: {asr_feature:.4f}")

    history.append({
        "round": 1,
        "stage": "attack_clean",
        "asr_feature": asr_feature,
    })

    # Round 2: Adversarial training defense
    print(f"\nRound 2: Training with adversarial training...")
    adv_model = GCNDetector(in_channels=16, hidden_channels=64, out_channels=3)
    adv_model = adv_model.to(DEVICE)

    class_counts = torch.bincount(data.y)
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(adv_model.parameters(), lr=0.01, weight_decay=5e-4)

    adv_trainer = AdversarialTraining(adv_model, attack_budget=0.03, feature_bound=1.0, attack_freq=3, alpha=0.5)

    best_val_f1 = 0.0
    best_state = None
    for epoch in range(200):
        adv_model.train()
        # Simplified training: use standard training with periodic attack
        optimizer.zero_grad()
        logits = adv_model(data.x, data.edge_index)

        # Create train mask
        train_ratio = 0.6
        n_train = int(train_ratio * data.num_nodes)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=DEVICE)
        idx = torch.randperm(data.num_nodes, device=DEVICE)[:n_train]
        train_mask[idx] = True

        clean_loss = criterion(logits[train_mask], data.y[train_mask])

        # Add adversarial loss every 5 epochs
        if epoch % 5 == 0:
            adv_target = (data.y > 0) & train_mask
            if adv_target.sum() > 0:
                attack_copy = FeatureAttack(adv_model, perturbation_bound=1.0, n_steps=10, lr=0.01)
                adv_data, _ = attack_copy.attack(data, adv_target, DEVICE)
                adv_logits = adv_model(adv_data.x, adv_data.edge_index)
                adv_loss = criterion(adv_logits[train_mask], data.y[train_mask])
                total_loss = clean_loss + 0.5 * adv_loss
            else:
                total_loss = clean_loss
        else:
            total_loss = clean_loss

        total_loss.backward()
        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            adv_model.eval()
            with torch.no_grad():
                val_logits = adv_model(data.x, data.edge_index)
                val_preds = val_logits.argmax(dim=1)
                val_metrics = compute_metrics(data.y, val_preds)
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.clone() for k, v in adv_model.state_dict().items()}

    if best_state:
        adv_model.load_state_dict(best_state)
    adv_model.eval()

    with torch.no_grad():
        adv_logits = adv_model(data.x, data.edge_index)
        adv_preds = adv_logits.argmax(dim=1)
        adv_metrics = compute_metrics(data.y, adv_preds)
    print(f"  Defended F1: {adv_metrics['macro_f1']:.4f}")

    history.append({
        "round": 2,
        "stage": "adversarial_training",
        "model_f1": adv_metrics["macro_f1"],
    })

    # Round 3: Attack defended model
    print(f"\nRound 3: Attacking defended model...")
    adv_feat_attack = FeatureAttack(adv_model, perturbation_bound=2.0, n_steps=50, lr=0.01)
    adv_perturbed, adv_feat_info = adv_feat_attack.attack(data, target_mask, DEVICE)
    adv_asr, _ = compute_attack_success_rate(adv_model, data, target_mask, adv_perturbed, DEVICE)
    print(f"  Defended Feature ASR: {adv_asr:.4f}")

    history.append({
        "round": 3,
        "stage": "attack_defended",
        "asr_feature": adv_asr,
    })

    # Round 4: SVD Purification
    print(f"\nRound 4: SVD Purification defense...")
    svd = SVDPurification(rank=64, symmetric=True)
    purified_edge, svd_info = svd.purify(data.edge_index, data.num_nodes, DEVICE)

    svd_model = deepcopy(model)
    svd_model.eval()

    with torch.no_grad():
        svd_logits = svd_model(data.x, purified_edge)
        svd_preds = svd_logits.argmax(dim=1)
        svd_metrics = compute_metrics(data.y, svd_preds)
    print(f"  SVD-defended F1: {svd_metrics['macro_f1']:.4f}")

    history.append({
        "round": 4,
        "stage": "svd_purification",
        "model_f1": svd_metrics["macro_f1"],
    })

    # Round 5: Attack SVD-defended model
    print(f"\nRound 5: Attacking SVD-defended model...")
    svd_data = data.clone()
    svd_data.edge_index = purified_edge
    svd_attack = FeatureAttack(svd_model, perturbation_bound=2.0, n_steps=50, lr=0.01)
    svd_perturbed, _ = svd_attack.attack(svd_data, target_mask, DEVICE)
    svd_asr, _ = compute_attack_success_rate(svd_model, data, target_mask, svd_perturbed, DEVICE)
    print(f"  SVD-defended Feature ASR: {svd_asr:.4f}")

    history.append({
        "round": 5,
        "stage": "attack_svd_defended",
        "asr_feature": svd_asr,
    })

    # Summary
    print(f"\n{'='*60}")
    print("ARMS RACE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Stage':<30} {'F1':<10} {'ASR':<10}")
    print(f"{'-'*50}")
    print(f"{'Clean baseline':<30} {clean_metrics['macro_f1']:.4f}     -")
    print(f"{'Feature attack (clean)':<30} -     {asr_feature:.4f}")
    print(f"{'Adversarial training':<30} {adv_metrics['macro_f1']:.4f}     -")
    print(f"{'Feature attack (adv trained)':<30} -     {adv_asr:.4f}")
    print(f"{'SVD purification':<30} {svd_metrics['macro_f1']:.4f}     -")
    print(f"{'Feature attack (SVD defended)':<30} -     {svd_asr:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "arms_race": history,
        "summary": {
            "clean_f1": clean_metrics["macro_f1"],
            "asr_clean": asr_feature,
            "adv_training_f1": adv_metrics["macro_f1"],
            "asr_adv_trained": adv_asr,
            "svd_f1": svd_metrics["macro_f1"],
            "asr_svd_defended": svd_asr,
        },
    }

    results_path = os.path.join(out_dir, f"adaptive_eval_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_adaptive_eval()
