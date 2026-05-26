"""
Advanced defense mechanisms for GNN-based anomaly detectors.

Defense strategies (SOTA):
1. SVDPurification: Low-rank spectral purification of adjacency matrix
2. RobustGCNAggregation: Median and trimmed-mean message aggregation
3. SelfSupervisedDefense: Contrastive pre-training (GraphCL-style)
4. EnsembleDefense: Multi-model ensemble for robustness
5. FeatureRandomizationDefense: Randomized smoothing for feature-space robustness (novel)
6. RobustFeatureTrainer: Noise-injected training for inherent feature robustness (novel)

References:
- Entezari et al., "All you need is low-rank", WSDM 2020
- Wang et al., "Robust Graph Convolutional Networks", AAAI 2020
- Suresh et al., "Graph Contrastive Learning with Adaptive Augmentation", ICML 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from copy import deepcopy


# ============================================================
# SVDPurification: Low-rank graph purification
# ============================================================

class SVDPurification:
    """Purify graph structure via low-rank SVD approximation.

    Removes adversarial edge perturbations by projecting the
    adjacency matrix onto its top-k singular vectors. This
    exploits the observation that adversarial edges tend to
    disrupt the low-rank structure of the original graph.

    Args:
        rank: number of singular values to keep
        symmetric: whether to enforce symmetry in reconstruction
    """

    def __init__(self, rank: int = 64, symmetric: bool = True):
        self.rank = rank
        self.symmetric = symmetric

    def purify(self, edge_index, num_nodes, device="cpu"):
        """Apply SVD purification to adjacency matrix.

        Args:
            edge_index: edge index tensor [2, n_edges]
            num_nodes: number of nodes
            device: compute device

        Returns:
            purified_edge_index: edges of purified graph
            info: dict with purification statistics
        """
        edge_index = edge_index.to(device)

        # Convert to dense adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

        if self.symmetric:
            adj = (adj + adj.T) / 2.0

        # Truncated SVD on CPU (more stable for large matrices)
        adj_cpu = adj.cpu().float()
        U, S, V = torch.linalg.svd(adj_cpu, full_matrices=False)

        # Keep top-k singular values
        U_k = U[:, :self.rank]
        S_k = S[:self.rank]
        V_k = V[:self.rank, :]

        # Reconstruct low-rank approximation
        adj_low_rank = U_k @ torch.diag(S_k) @ V_k

        if self.symmetric:
            adj_low_rank = (adj_low_rank + adj_low_rank.T) / 2.0

        n_orig = edge_index.shape[1] // 2  # undirected

        # Threshold to get binary edges
        # Use a percentile-based threshold to preserve edge density
        positive_vals = adj_low_rank[adj_low_rank > 0]
        top_k = min(n_orig, len(positive_vals))  # preserve same number of edges
        if top_k > 0 and len(positive_vals) > 0:
            threshold = torch.topk(positive_vals, min(top_k, len(positive_vals))).values[-1].item()
        else:
            threshold = 0.0

        adj_binary = (adj_low_rank > threshold).float()

        # Remove self-loops
        adj_binary.fill_diagonal_(0)

        # Convert back to edge index
        purified_edge_index, _ = dense_to_sparse(adj_binary)
        purified_edge_index = purified_edge_index.to(device)

        n_purified = purified_edge_index.shape[1] // 2

        return purified_edge_index, {
            "n_edges_original": n_orig,
            "n_edges_purified": n_purified,
            "edges_removed": max(0, n_orig - n_purified),
            "edges_added": max(0, n_purified - n_orig),
            "rank": self.rank,
            "top_singular_values": S_k[:5].tolist(),
        }


# ============================================================
# RobustGCNAggregation: Median/trimmed-mean aggregation
# ============================================================

class RobustGCNLayer(nn.Module):
    """Robust GCN layer with median aggregation.

    Replaces the standard sum aggregation with median
    aggregation, which is resistant to outlier neighbors
    introduced by adversarial attacks.

    Reference: Wang et al., "Robust Graph Convolutional Networks", AAAI 2020
    """

    def __init__(self, in_channels, out_channels, aggregation="median"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation

        self.weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        """Forward pass with robust aggregation.

        Args:
            x: node features [n_nodes, in_channels]
            edge_index: edge index [2, n_edges]

        Returns:
            output: transformed features [n_nodes, out_channels]
        """
        n = x.size(0)
        row, col = edge_index

        # Symmetric normalization
        deg = torch.bincount(row, minlength=n).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Linear transform
        x_transformed = x @ self.weight

        if self.aggregation == "mean":
            # Standard GCN aggregation
            msg = x_transformed[col] * norm.unsqueeze(1)
            out = torch.zeros_like(x_transformed[:, :self.out_channels])
            out.index_add_(0, row, msg)

        elif self.aggregation == "median":
            # Median aggregation: for each node, take median of neighbor features
            out = self._median_aggregate(x_transformed, row, col, n)

        elif self.aggregation == "trimmed_mean":
            # Trimmed mean: remove top and bottom 25% of neighbor values
            out = self._trimmed_mean_aggregate(x_transformed, row, col, n, trim_ratio=0.25)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        out = out + self.bias
        return out

    def _median_aggregate(self, x, row, col, n_nodes):
        """Aggregate using median of normalized neighbor features."""
        out = torch.zeros(n_nodes, x.size(1), device=x.device)
        for node in range(n_nodes):
            neighbor_mask = row == node
            if neighbor_mask.sum() == 0:
                out[node] = x[node]  # self if no neighbors
            else:
                neighbors = x[col[neighbor_mask]]
                out[node] = neighbors.median(dim=0).values
        return out

    def _trimmed_mean_aggregate(self, x, row, col, n_nodes, trim_ratio=0.25):
        """Aggregate using trimmed mean of neighbor features."""
        out = torch.zeros(n_nodes, x.size(1), device=x.device)
        for node in range(n_nodes):
            neighbor_mask = row == node
            n_neighbors = neighbor_mask.sum()
            if n_neighbors == 0:
                out[node] = x[node]
            else:
                neighbors = x[col[neighbor_mask]]
                n_trim = max(1, int(n_neighbors * trim_ratio))
                if n_neighbors <= 2 * n_trim:
                    out[node] = neighbors.mean(dim=0)
                else:
                    sorted_neighbors, _ = torch.sort(neighbors, dim=0)
                    trimmed = sorted_neighbors[n_trim:n_neighbors - n_trim]
                    out[node] = trimmed.mean(dim=0)
        return out


class RobustGCN(nn.Module):
    """Robust GCN detector with median aggregation.

    Uses robust aggregation layers instead of standard GCN
    to resist adversarial edge perturbations.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, aggregation="median"):
        super().__init__()
        self.conv1 = RobustGCNLayer(in_channels, hidden_channels, aggregation=aggregation)
        self.conv2 = RobustGCNLayer(hidden_channels, hidden_channels, aggregation=aggregation)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.dropout(h)
        return self.classifier(h)

    def get_embedding(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h


# ============================================================
# SelfSupervisedDefense: Contrastive pre-training
# ============================================================

class GraphCLPretrainer:
    """Contrastive pre-training for graph representations (GraphCL-style).

    Pre-trains the GNN encoder using contrastive learning on
    augmented views of the graph, then fine-tunes for anomaly
    detection. The contrastive objective learns representations
    that are invariant to small perturbations, providing
    inherent adversarial robustness.

    Augmentations:
    1. Edge dropping: randomly remove edges
    2. Feature masking: randomly mask feature dimensions

    Reference: You et al., "Graph Contrastive Learning with Adaptive Augmentation", ICML 2021
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.2,
        edge_drop_rate: float = 0.2,
        feature_mask_rate: float = 0.1,
    ):
        self.encoder = encoder
        self.temperature = temperature
        self.edge_drop_rate = edge_drop_rate
        self.feature_mask_rate = feature_mask_rate

        # Projection head for contrastive learning
        self._build_projection_heads(projection_dim)

    def _build_projection_heads(self, proj_dim):
        """Build projection heads for contrastive learning."""
        # Get embedding dimension from encoder
        if hasattr(self.encoder, 'conv2'):
            embed_dim = self.encoder.conv2.out_channels if hasattr(self.encoder.conv2, 'out_channels') else 64
        else:
            embed_dim = 64

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def _augment_edge(self, edge_index, num_nodes):
        """Create augmented view by dropping edges."""
        n_edges = edge_index.shape[1]
        keep_mask = torch.rand(n_edges, device=edge_index.device) > self.edge_drop_rate
        return edge_index[:, keep_mask]

    def _augment_feature(self, x):
        """Create augmented view by masking features."""
        x_aug = x.clone()
        mask = torch.rand_like(x) < self.feature_mask_rate
        x_aug[mask] = 0.0
        return x_aug

    def _get_embeddings(self, x, edge_index):
        """Get embeddings from encoder."""
        if hasattr(self.encoder, 'get_embedding'):
            return self.encoder.get_embedding(x, edge_index)
        else:
            # Fallback: run forward without classifier
            h = F.relu(self.encoder.conv1(x, edge_index))
            h = F.dropout(h, p=0.5, training=self.encoder.training)
            return self.encoder.conv2(h, edge_index)

    def contrastive_loss(self, z1, z2):
        """Compute NT-Xent contrastive loss.

        Args:
            z1: embeddings from first augmented view
            z2: embeddings from second augmented view

        Returns:
            loss: contrastive loss value
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Similarity matrix
        sim_matrix = z1 @ z2.T / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(z1.size(0), device=z1.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def pretrain(self, data, n_epochs=100, lr=0.001, device="cpu"):
        """Pre-train encoder with contrastive learning.

        Args:
            data: PyG Data object
            n_epochs: number of pre-training epochs
            lr: learning rate
            device: compute device

        Returns:
            history: list of loss values per epoch
        """
        self.encoder = self.encoder.to(device)
        self.projector = self.projector.to(device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=lr,
        )

        history = []
        for epoch in range(n_epochs):
            self.encoder.train()
            optimizer.zero_grad()

            # Create two augmented views
            edge1 = self._augment_edge(data.edge_index, data.num_nodes)
            edge2 = self._augment_edge(data.edge_index, data.num_nodes)
            x1 = self._augment_feature(data.x)
            x2 = self._augment_feature(data.x)

            # Get embeddings
            z1 = self._get_embeddings(x1, edge1)
            z2 = self._get_embeddings(x2, edge2)

            # Project
            z1_proj = self.projector(z1)
            z2_proj = self.projector(z2)

            # Contrastive loss
            loss = self.contrastive_loss(z1_proj, z2_proj)
            loss.backward()
            optimizer.step()

            history.append(loss.item())

            if (epoch + 1) % 20 == 0:
                print(f"  Contrastive pretrain epoch {epoch+1}/{n_epochs}, loss={loss.item():.4f}")

        return history

    def get_pretrained_encoder(self):
        """Return the pre-trained encoder."""
        return self.encoder


# ============================================================
# EnsembleDefense: Multi-model ensemble
# ============================================================

class EnsembleDetector(nn.Module):
    """Ensemble of GNN detectors for improved robustness.

    Uses multiple independently trained detectors and
    combines their predictions through averaging.
    Adversarial examples crafted for one model may not
    transfer to others, providing inherent robustness.
    """

    def __init__(self, models: list, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0 / len(models)] * len(models)

    def forward(self, x, edge_index):
        """Forward pass: average predictions from all models."""
        logits_list = []
        for model in self.models:
            # Support both inference (no_grad) and attack (with grad)
            if torch.is_grad_enabled():
                logits = model(x, edge_index)
            else:
                model.eval()
                with torch.no_grad():
                    logits = model(x, edge_index)
            logits_list.append(logits)

        # Weighted average of logits
        avg_logits = sum(w * l for w, l in zip(self.weights, logits_list))
        return avg_logits

    def individual_predictions(self, x, edge_index):
        """Get individual model predictions for analysis."""
        preds_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                preds_list.append(logits.argmax(dim=1))
        return preds_list


# ============================================================
# FeatureRandomizationDefense: Randomized smoothing for feature robustness
# ============================================================

class FeatureRandomizationDefense:
    """Randomized feature smoothing defense against feature-space attacks.

    Inspired by randomized smoothing certificates (Cohen et al., ICML 2019),
    this defense adds calibrated Gaussian noise to input features at inference
    time and averages predictions over multiple noise samples. The key insight:
    adversarial perturbations are precise and localized; noise blurs the
    perturbation signal, forcing the attacker to use much larger (and more
    detectable) perturbations to succeed.

    Two modes:
    1. Inference-time: add noise + multi-sample voting (no retraining needed)
    2. Training-time: noise injection during training for inherent robustness

    Args:
        noise_std: standard deviation of Gaussian noise added to features
        n_samples: number of noise samples to average over at inference
        smooth_logits: if True, average logits before argmax; if False, vote on predictions
    """

    def __init__(self, noise_std: float = 0.1, n_samples: int = 10, smooth_logits: bool = True):
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.smooth_logits = smooth_logits

    def defend_predict(self, model, x, edge_index):
        """Make prediction using randomized smoothing.

        Args:
            model: GNN model
            x: node features [n_nodes, n_features]
            edge_index: graph structure

        Returns:
            averaged_logits: smoothed model output
            predictions: class predictions
        """
        model.eval()
        device = x.device

        if self.n_samples == 1:
            with torch.no_grad():
                return model(x, edge_index), model(x, edge_index).argmax(dim=1)

        logits_sum = torch.zeros(x.shape[0], model(x, edge_index).shape[1], device=device)

        with torch.no_grad():
            for _ in range(self.n_samples):
                noise = torch.randn_like(x) * self.noise_std
                x_noisy = x + noise
                logits = model(x_noisy, edge_index)
                logits_sum += logits

        avg_logits = logits_sum / self.n_samples
        return avg_logits, avg_logits.argmax(dim=1)

    def defend_predict_batch(self, model, data, batch_size=1000):
        """Memory-efficient version for large graphs."""
        model.eval()
        device = data.x.device
        n_nodes = data.x.shape[0]

        logits_sum = torch.zeros(n_nodes, 2, device=device)

        with torch.no_grad():
            for i in range(0, self.n_samples, batch_size):
                n_batch = min(batch_size, self.n_samples - i)
                noise = torch.randn(n_batch, *data.x.shape, device=device) * self.noise_std
                x_noisy = data.x.unsqueeze(0) + noise
                # Process each sample
                for j in range(n_batch):
                    logits = model(x_noisy[j], data.edge_index)
                    logits_sum += logits

        avg_logits = logits_sum / self.n_samples
        return avg_logits, avg_logits.argmax(dim=1)


class RobustFeatureTrainer:
    """Training wrapper that injects feature noise for inherent robustness.

    Unlike adversarial training (which uses targeted adversarial examples),
    this approach adds random Gaussian noise to features during training.
    This is computationally cheaper and provides robustness against a
    broader class of perturbations, at the cost of slightly lower clean accuracy.

    The noise_std parameter should be calibrated to the expected attack strength:
    too low → no robustness; too high → clean accuracy collapses.

    Args:
        noise_std: standard deviation of noise injected during training
        noise_schedule: 'constant' or 'anneal' (decrease noise over epochs)
    """

    def __init__(self, noise_std: float = 0.1, noise_schedule: str = "constant"):
        self.noise_std = noise_std
        self.noise_schedule = noise_schedule

    def train_robust(self, model, data, train_mask, val_mask, epochs=200,
                     lr=1e-3, weight_decay=5e-4, device="cpu"):
        """Train model with feature noise injection.

        Args:
            model: GNN model
            data: PyG Data object
            train_mask: training node mask
            val_mask: validation node mask
            epochs: training epochs
            lr: learning rate
            weight_decay: weight decay for optimizer
            device: compute device

        Returns:
            model: trained model
            history: dict with clean_f1, robust_f1 per epoch
        """
        from sklearn.metrics import f1_score, accuracy_score

        model = model.to(device)
        model.apply(self._init_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        data_d = data.to(device)
        train_mask_d = train_mask.to(device)
        val_mask_d = val_mask.to(device)

        best_val_acc = 0.0
        best_state = None
        history = {"clean_acc": [], "clean_f1": [], "robust_acc": [], "robust_f1": []}

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # Current noise level (anneal if schedule is set)
            if self.noise_schedule == "anneal":
                current_noise = self.noise_std * (1.0 - epoch / epochs)
            else:
                current_noise = self.noise_std

            # Add noise to features during training
            noise = torch.randn_like(data_d.x) * current_noise
            x_noisy = data_d.x + noise
            output = model(x_noisy, data_d.edge_index)
            loss = criterion(output[train_mask_d], data_d.y[train_mask_d])
            loss.backward()
            optimizer.step()

            # Evaluate every 20 epochs
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    # Clean evaluation
                    clean_out = model(data_d.x, data_d.edge_index)
                    clean_preds = clean_out.argmax(dim=1)
                    val_clean_acc = accuracy_score(
                        data_d.y[val_mask_d].cpu().numpy(),
                        clean_preds[val_mask_d].cpu().numpy()
                    )
                    val_clean_f1 = f1_score(
                        data_d.y[val_mask_d].cpu().numpy(),
                        clean_preds[val_mask_d].cpu().numpy(),
                        average="macro"
                    )

                    # Robust evaluation (with noise)
                    noise_eval = torch.randn_like(data_d.x) * self.noise_std
                    robust_out = model(data_d.x + noise_eval, data_d.edge_index)
                    robust_preds = robust_out.argmax(dim=1)
                    val_robust_acc = accuracy_score(
                        data_d.y[val_mask_d].cpu().numpy(),
                        robust_preds[val_mask_d].cpu().numpy()
                    )
                    val_robust_f1 = f1_score(
                        data_d.y[val_mask_d].cpu().numpy(),
                        robust_preds[val_mask_d].cpu().numpy(),
                        average="macro"
                    )

                history["clean_acc"].append(val_clean_acc)
                history["clean_f1"].append(val_clean_f1)
                history["robust_acc"].append(val_robust_acc)
                history["robust_f1"].append(val_robust_f1)

                if val_clean_acc > best_val_acc:
                    best_val_acc = val_clean_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        return model, history

    def _init_weights(self, m):
        """Initialize weights using kaiming uniform."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class TRADESAdversarialTrainer:
    """TRADES-style adversarial training for GNNs.

    Unlike naive adversarial training that adds adversarial loss with a fixed weight,
    TRADES (Theoretically principled Robustness via Adversarial training for
    Deep learning with Smoothness) uses KL divergence between clean and adversarial
    predictions to enforce a smooth decision boundary:

        L = L_ce(f(x), y) + (1/lambda) * KL(f(x) || f(x_adv))

    The KL term encourages the model to produce similar predictions for clean
    and adversarial inputs, creating a locally flat decision boundary around
    each training point. This is fundamentally different from naive AT, which
    simply tries to classify adversarial examples correctly — often leading
    to memorization rather than robust feature learning.

    Args:
        model: GNN model to train
        perturbation_bound: L2 budget for PGD attack during training
        n_steps: PGD optimization steps for generating adversarial examples
        lr: learning rate for PGD attack
        beta: TRADES beta parameter (1/lambda), controls robustness-accuracy tradeoff
    """

    def __init__(self, model, perturbation_bound=5.0, n_steps=50, lr=0.01, beta=6.0):
        self.model = model
        self.perturbation_bound = perturbation_bound
        self.n_steps = n_steps
        self.lr = lr
        self.beta = beta

    def _generate_adversarial(self, data, target_mask, device):
        """Generate adversarial examples using PGD (feature attack only)."""
        model = self.model.to(device)
        data = data.to(device)
        model.eval()

        target_nodes = target_mask.nonzero(as_tuple=True)[0]
        if len(target_nodes) == 0:
            return data

        x_original = data.x.clone().detach()
        delta = torch.zeros_like(data.x)
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for step in range(self.n_steps):
            optimizer.zero_grad()
            x_adv = x_original + delta
            logits = model(x_adv, data.edge_index)
            # Target: class 0 (normal) for all target nodes
            target_labels = torch.zeros(len(target_nodes), dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(
                logits[target_nodes], target_labels
            )
            loss.backward()
            delta.grad.zero_()

            # Project delta for target nodes only
            for node_idx in target_nodes:
                node_delta = delta[node_idx]
                node_norm = node_delta.norm(2)
                if node_norm > self.perturbation_bound:
                    delta.data[node_idx] = node_delta * (self.perturbation_bound / node_norm)

        # Create perturbed data
        from torch_geometric.data import Data
        x_adv = x_original + delta.detach()
        adv_data = Data(x=x_adv, edge_index=data.edge_index, y=data.y)
        return adv_data

    def train(self, data, train_mask, val_mask, device, epochs=200, lr=1e-3, weight_decay=5e-4):
        """Train model with TRADES loss.

        Args:
            data: PyG Data object
            train_mask: training mask
            val_mask: validation mask
            device: compute device
            epochs: number of training epochs
            lr: learning rate for model optimization
            weight_decay: weight decay for AdamW

        Returns:
            model: trained model
            history: training history dict
        """
        model = self.model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_acc = 0.0
        best_state = None
        history = {"clean_loss": [], "adv_loss": [], "kl_loss": [], "val_acc": []}

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # Clean forward pass
            logits_clean = model(data.x, data.edge_index)
            clean_loss = torch.nn.functional.cross_entropy(
                logits_clean[train_mask], data.y[train_mask]
            )

            # Generate adversarial examples
            adv_data = self._generate_adversarial(data, train_mask, device)
            logits_adv = model(adv_data.x, adv_data.edge_index)

            # TRADES loss: clean CE + beta * KL(clean || adv)
            # KL divergence encourages model to output similar predictions
            # for clean and adversarial inputs
            clean_probs = torch.nn.functional.softmax(logits_clean[train_mask], dim=1)
            adv_log_probs = torch.nn.functional.log_softmax(logits_adv[train_mask], dim=1)
            kl_loss = torch.nn.functional.kl_div(
                adv_log_probs, clean_probs, reduction="batchmean"
            )

            total_loss = clean_loss + self.beta * kl_loss
            total_loss.backward()
            optimizer.step()

            history["clean_loss"].append(clean_loss.item())
            history["kl_loss"].append(kl_loss.item())

            # Validation every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(data.x, data.edge_index)
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds[val_mask] == data.y[val_mask]).float().mean().item()
                history["val_acc"].append(val_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        return model, history
