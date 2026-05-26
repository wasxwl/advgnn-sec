# Datasets

## MGTAB (Real-World Twitter Bot Detection)
- **Source**: https://github.com/GraphDetec/MGTAB
- **License**: CC BY-NC-ND 4.0
- **Download**: Google Drive (see repo README)
- **Contents**:
  - `features.pt`: 10,199 x 788 node feature matrix
  - `edge_index.pt`: (2, 1,700,108) edge connectivity
  - `edge_type.pt`: 7 edge types (followers, friends, mention, reply, quoted, URL, hashtag)
  - `edge_weight.pt`: edge weights
  - `labels_bot.pt`: bot labels (0=human, 1=bot) — 2,748 bots (26.94%)
  - `labels_stance.pt`: stance labels (0=neutral, 1=against, 2=support)
- **Usage**: Place all .pt files in `data/mgtab/`
- **Loader**: `from src.dataset.mgtab_loader import MGTABDataset`

## TwiBot-20 (Twitter Bot Detection Benchmark)
- **Source**: https://github.com/safe-graph/TwiBot-20
- **License**: CC BY-NC-ND 4.0 (see original repo for details)
- **Contents**:
  - 229,573 users, 33,488,192 tweets, 8,723,736 user property items
  - 455,958 follow relationships
  - 13-dimensional node features (profile attributes)
- **Download**: Follow instructions at the official TwiBot-20 repository
- **Usage**: Place processed `.pt` files in `data/twibot20/`
- **Loader**: Use `TwiBot22StyleDataset` in `src/dataset/loader.py` (adapted for TwiBot-20 schema)

## Synthetic Data
Generated on-the-fly by `src/dataset/loader.py`:
- `SyntheticSocialNetwork`: 1,050 nodes, 16-dim features
- `TwiBot22StyleDataset`: 10,000 nodes, 13-dim Twitter profile features

## Custom Edge List Format
- `edges.txt` — two columns: `src dst` (0-indexed), one edge per line
- `features.npy` — numpy array of shape `(n_nodes, feature_dim)`
- `labels.txt` — one label per line (0=normal, 1=anomaly, 2=attacker)
