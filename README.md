# AdvGNN-Sec: Adversarial Robustness of GNN-based Anomaly Detection in Social Networks

Research project on adversarial attacks and defenses for GNN-based anomaly detection in online social networks.

## Abstract

Graph Neural Networks (GNNs) have become a widely used method for anomaly detection in online social networks, yet their adversarial robustness remains poorly understood. This project systematically studies attack and defense methods for GNN-based anomaly detectors. We find that GNN detectors primarily rely on node features rather than graph structure, making them highly vulnerable to feature space perturbation attacks.

## Key Findings

1. **Feature attacks are devastating**: 100% attack success rate (ASR) across all GNN architectures (GCN, GAT, GraphSAGE, RGCN) with minimal perturbation
2. **Graph attacks are ineffective**: 0% ASR even with substantial edge modifications (13,000+ edges)
3. **Feature dominance**: A simple MLP achieves 88.7% accuracy on MGTAB vs 90.6% for best GNN (RGCN)
4. **Defenses insufficient**: All evaluated defenses (adversarial training, SVD purification, randomized smoothing) fail against adaptive white-box attackers
5. **Black-box transfer**: Adversarial examples transfer across models at 53%-98% success rate
6. **Economic threat**: Each evaded bot account costs only $0.20, with ROI up to 2400%

## Project Structure

```
advgnn-sec/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml              # Default experiment settings
│   └── attack.yaml               # Attack hyperparameters
├── data/
│   └── README.md                 # Instructions for downloading datasets
├── experiments/
│   ├── baseline.py               # Train baseline detectors
│   ├── attack_eval.py            # Evaluate attacks (synthetic data)
│   ├── defense_eval.py           # Evaluate defenses (synthetic data)
│   ├── mgtab_eval.py             # Evaluate attacks on MGTAB dataset
│   ├── mgtab_trades.py           # TRADES adversarial training on MGTAB
│   ├── mgtab_rgcn.py             # RGCN evaluation on MGTAB
│   ├── mgtab_defense_5seeds.py   # Defense evaluation across 5 seeds
│   ├── mgtab_blackbox.py         # Black-box transfer evaluation
│   ├── mgtab_comprehensive.py    # Comprehensive MGTAB evaluation
│   ├── mgtab_relation_aware.py   # Relation-aware evaluation
│   ├── mgtab_robust_training.py  # Robust training experiments
│   ├── twibot20_eval.py          # TwiBot-20 cross-dataset evaluation
│   ├── adaptive_eval.py          # Adaptive arms-race simulation
│   ├── ablation.py               # Ablation studies
│   └── comprehensive_eval.py     # Comprehensive evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_detector.ipynb
│   ├── 03_adversarial_attack.ipynb
│   ├── 04_adversarial_defense.ipynb
│   └── 05_full_pipeline.ipynb
├── paper/
│   └── *.pdf, *.png              # Paper figures
├── scripts/
│   ├── run_all.sh                # Full pipeline
│   ├── run_attack.sh             # Attack evaluation
│   ├── run_baseline.sh           # Baseline training
│   └── run_defense.sh            # Defense evaluation
└── src/
    ├── __init__.py
    ├── train.py                  # Training utilities
    ├── dataset/
    │   ├── loader.py             # Synthetic data generation
    │   ├── mgtab_loader.py       # MGTAB dataset loader
    │   ├── twibot_loader.py      # TwiBot dataset loader
    │   ├── twibot20_loader.py    # TwiBot-20 dataset loader
    │   └── preprocess.py         # Feature normalization
    ├── models/
    │   ├── detector.py           # GCN/GAT/GraphSAGE detectors
    │   ├── mgtab_models.py       # Rel-GCN/Rel-GAT for MGTAB
    │   ├── attack.py             # PGD feature/graph attacks
    │   ├── temporal_attack.py    # Temporal attack across snapshots
    │   ├── defense.py            # Adversarial training, SVD purification
    │   └── defense_advanced.py   # Advanced defense methods
    └── utils/
        ├── metrics.py            # ASR, F1, accuracy metrics
        ├── visualization.py      # Plotting utilities
        └── economic_analysis.py  # Cost-benefit analysis
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline detector
python experiments/baseline.py --device cuda:0

# Evaluate adversarial attacks
python experiments/attack_eval.py --device cuda:0

# Evaluate defenses
python experiments/defense_eval.py --device cuda:0

# Evaluate on MGTAB dataset (requires downloading data first)
python experiments/mgtab_eval.py --device cuda:0

# Run full pipeline
bash scripts/run_all.sh
```

## Datasets

This project evaluates on three datasets:

| Dataset | Nodes | Edges | Features | Description |
|---------|-------|-------|----------|-------------|
| Synthetic | 1,050 | 31,300 | 16 | Generated social network |
| MGTAB | 10,199 | 850,054 | 788 | Multi-relational Twitter benchmark |
| TwiBot-20 | 229,573 | 33.5M | Multi-modal | Comprehensive Twitter bot benchmark |

**Data Preparation**: See `data/README.md` for instructions on downloading and preprocessing the MGTAB and TwiBot-20 datasets. The synthetic dataset is generated automatically.

## Experiments

### Attack Evaluation

| Attack Type | Synthetic ASR | MGTAB ASR | TwiBot-20 ASR |
|-------------|---------------|-----------|---------------|
| Feature (eps=0.5) | 47.4% | 45.4%-78.5% | ~50% |
| Feature (eps=1.0) | 86.0% | 85.6%-95.2% | ~90% |
| Feature (eps=5.0) | 100% | 100% | 99%-100% |
| Graph (rho=0.05) | 0% | N/A | N/A |
| Temporal (snap 1) | 98.4% | N/A | N/A |

### Defense Evaluation

| Defense | Clean F1 | ASR (eps=5.0) |
|---------|----------|---------------|
| None (baseline) | 94.6% | 100% |
| Adversarial Training | 34.4% | 100% |
| SVD Purification | 76.9% | 78.3% |
| Feature Randomization | 74.4% | 100% |
| Robust Feature Trainer | 82.3% | 100% |

## Citation

If you use this code in your research, please cite:

```
@article{advgnnsec2026,
  title={AdvGNN-Sec: Adversarial Robustness of Graph Neural Network-based Anomaly Detection in Social Networks},
  author={Lai, Tian-Yi},
  journal={Chinese Journal of Computers},
  year={2026}
}
```

## License

This project is for educational and research purposes.
