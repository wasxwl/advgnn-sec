"""
Visualization utilities for experiment results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(
    train_losses, val_losses, val_accs,
    save_path: str = "results/training_curve.png",
):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_accs, label="Val Accuracy", marker="o", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attack_results(
    budgets, asr_values, defense_asr=None,
    save_path: str = "results/attack_curve.png",
):
    """Plot attack success rate vs perturbation budget."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(budgets, asr_values, "r-o", label="Attack (No Defense)", linewidth=2)

    if defense_asr is not None:
        ax.plot(budgets, defense_asr, "g-s", label="Attack (With Defense)", linewidth=2)

    ax.set_xlabel("Perturbation Budget (edge ratio)", fontsize=12)
    ax.set_ylabel("Attack Success Rate", fontsize=12)
    ax.set_title("Adversarial Attack Effectiveness", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm, class_names,
    save_path: str = "results/confusion_matrix.png",
):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_bar(
    categories, values_before, values_after, labels,
    save_path: str = "results/comparison.png",
    metric_name: str = "Metric",
):
    """Plot before/after comparison bar chart."""
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, values_before, width, label=labels[0])
    rects2 = ax.bar(x + width / 2, values_after, width, label=labels[1])

    ax.set_ylabel(metric_name)
    ax.set_title("Before vs After Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
