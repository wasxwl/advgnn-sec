"""
Training loop for GNN-based anomaly detector.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset.preprocess import normalize_features, create_train_val_test_split
from src.utils.metrics import compute_metrics


def train_model(
    model,
    data,
    n_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    early_stopping_patience: int = 20,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train the anomaly detector.

    Args:
        model: GNN model to train
        data: PyG Data object with x, edge_index, y
        n_epochs: maximum number of training epochs
        lr: learning rate
        weight_decay: L2 regularization
        early_stopping_patience: stop if val loss doesn't improve
        device: compute device
        verbose: print progress

    Returns:
        best_model: model with best validation performance
        history: dict with training metrics per epoch
    """
    model = model.to(device)
    data = data.to(device)
    data = normalize_features(data)

    # Train/val/test split
    train_mask, val_mask, test_mask = create_train_val_test_split(data)
    train_mask, val_mask, test_mask = (
        train_mask.to(device),
        val_mask.to(device),
        test_mask.to(device),
    )

    # Use cross-entropy loss with class weights for imbalanced data
    class_counts = torch.bincount(data.y)
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "test_acc": [],
        "test_f1": [],
    }

    epoch_range = tqdm(range(n_epochs), desc="Training", disable=not verbose)

    for epoch in epoch_range:
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_loss = criterion(val_logits[val_mask], data.y[val_mask]).item()
            val_preds = val_logits.argmax(dim=1)
            val_metrics = compute_metrics(data.y[val_mask], val_preds[val_mask])

        # Test
        with torch.no_grad():
            test_logits = model(data.x, data.edge_index)
            test_preds = test_logits.argmax(dim=1)
            test_metrics = compute_metrics(data.y[test_mask], test_preds[test_mask])

        # Track
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["macro_f1"])
        history["test_acc"].append(test_metrics["accuracy"])
        history["test_f1"].append(test_metrics["macro_f1"])

        # Early stopping based on validation F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            epoch_range.set_postfix({
                "train_loss": f"{loss.item():.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_f1": f"{val_metrics['macro_f1']:.4f}",
                "test_f1": f"{test_metrics['macro_f1']:.4f}",
            })

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x, data.edge_index)
        final_preds = final_logits.argmax(dim=1)
        final_metrics = compute_metrics(data.y[test_mask], final_preds[test_mask])

    if verbose:
        print(f"\nFinal Test Results:")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {final_metrics['macro_f1']:.4f}")
        print(f"  Binary F1: {final_metrics['binary_f1']:.4f}")

    return model, history, final_metrics
