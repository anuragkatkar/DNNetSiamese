# CLAUDE.md — DNNet Project Guide

This file helps AI coding assistants understand the project structure, conventions, and common workflows.

## Project Overview

Re-implementation of **"Dog Nose-Print Identification Using Deep Neural Networks"** (Bae et al., IEEE Access 2021).

A Siamese neural network that identifies individual dogs from nose-print photographs.
The model combines ResNet-152 feature extraction, a Dual Attention Network (DAN), ArcFace classification, and contrastive loss.

---

## Repository Layout

```
dnnet/
├── CLAUDE.md               ← this file
├── train.py                ← training loop + cross-validation runner
├── analyse.py              ← post-training analysis: metrics, plots, TSV export
├── inference.py            ← checkpoint evaluation + single-query identification
├── configs/
│   └── config.py           ← all hyperparameters — edit this before training
├── data/
│   └── dataset.py          ← dataset scanning, Siamese pair sampling, k-fold splits
├── models/
│   ├── feature_extraction.py  ← ResNet-152 backbone + extra conv blocks
│   ├── attention.py           ← channel + spatial attention modules (DAN)
│   ├── dnnet.py               ← full Siamese model: forward(), get_embedding()
│   └── losses.py              ← ContrastiveLoss, ArcFaceLoss, TotalLoss (eq. 9)
├── utils/
│   ├── evaluation.py       ← extract_embeddings, rank-K, VR@FAR, ROC, confusion
│   ├── schedulers.py       ← fixed + linear-decay LR schedule
│   ├── checkpoint.py       ← save_checkpoint / load_checkpoint
│   └── visualization.py   ← t-SNE, ROC, confusion matrix plot helpers
├── checkpoints/            ← saved model weights (gitignored)
├── logs/                   ← training logs (gitignored)
└── wandb/                  ← W&B run artefacts (gitignored)
```

---

## Dataset Format

```
dataset/
├── dog_001/
│   ├── nose_01.jpg
│   └── nose_02.jpg
├── dog_002/
│   └── ...
```

Each subdirectory name is a unique dog identity. Each dog needs ≥ 2 images.
Set `DATA_ROOT` in `configs/config.py` to the dataset root.

---

## Key Config Options (`configs/config.py`)

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `"./preprocess"` | Path to dataset root |
| `CHECKPOINT_DIR` | `"./checkpoints"` | Where checkpoints are saved |
| `NUM_EPOCHS` | `200` | Total training epochs |
| `PATIENCE` | `0` | Early stop after N non-improving validations (`0` = disabled) |
| `BATCH_SIZE` | `16` | Training batch size |
| `USE_WANDB` | `True` | Toggle Weights & Biases logging |
| `EMBEDDING_DIM` | `128` | Embedding vector dimension |
| `LR_CONTRASTIVE` | `1e-4` | Adam LR for feature/attention branch |
| `LR_ARCFACE` | `1e-4` | SGD LR for ArcFace head |
| `LR_WARMUP_EPOCHS` | `100` | Epochs before LR starts linear decay |

---

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train (single fold)
```bash
python train.py --fold 0
```

### Train (all 5 folds, cross-validation)
```bash
python train.py --cv
```

### Resume training
```bash
python train.py --fold 0 --resume checkpoints/fold_0/epoch_0099.pth
```

### Full post-training analysis (metrics + plots + TSV)
```bash
python analyse.py eval \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --data_root  ./dataset \
    --fold       0 \
    --output_dir ./analysis
```

### Embed a folder of images
```bash
python analyse.py embed \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --image_dir  ./my_nose_images \
    --output_dir ./analysis
```

### Evaluate a checkpoint
```bash
python inference.py eval \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --data_root  ./dataset \
    --fold       0
```

### Identify a query image
```bash
python inference.py identify \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --gallery    ./dataset \
    --query      ./query_nose.jpg \
    --top_k      5
```

---

## TensorBoard Embedding Projector

Both `analyse.py eval` and `analyse.py embed` save two TSV files alongside `embeddings.npz`:

| File | Contents |
|---|---|
| `embeddings_projector.tsv` | N × D matrix, tab-separated, no header |
| `metadata_projector.tsv` | Header + one row per embedding: `label`, `filename` |

**Load in TensorBoard:**
```bash
tensorboard --logdir ./analysis
```
Then open the Projector tab and point it to the TSV files.

**Or upload directly** at https://projector.tensorflow.org — load `embeddings_projector.tsv` as vectors and `metadata_projector.tsv` as metadata.

---

## Model Architecture

```
Input (B, 3, 256, 256)
    │
    ▼  FeatureExtractionModule
    ResNet-152 (up to layer4, no GAP/FC)    → (B, 2048, 8, 8)
    ExtraConvBlock(2048 → 512)              → (B,  512, 8, 8)
    ExtraConvBlock( 512 → 256)              → (B,  256, 8, 8)
    │
    ▼  AttentionModule (DAN)
    ChannelAttentionModule                  → (B, 256, 8, 8)
    SpatialAttentionModule                  → (B, 256, 8, 8)
    Concatenate [channel, spatial, input]   → (B, 768, 8, 8)
    GlobalAveragePooling                    → (B, 768)
    FC(768 → EMBEDDING_DIM) + L2-normalise  → (B, EMBEDDING_DIM)
    │
    ▼  Siamese head
    anchor branch ──┐
                    ├──► ContrastiveLoss  (eq. 7)
    pair   branch ──┘
    both  → ArcFaceLinear → ArcFaceLoss (eq. 8)
    TotalLoss = L_con + ½(L_arc_anchor + L_arc_pair)  (eq. 9)
```

---

## Code Conventions

- All hyperparameters live in `configs/config.py`. Import as `from configs import config`.
- Model weights are always loaded via `utils/checkpoint.py` (`load_checkpoint`).
- Embeddings are L2-normalised inside `model.get_embedding()`.
- Evaluation uses leave-one-out gallery (query excluded from its own retrieval).
- Validation runs every 5 epochs during training to save time.
- Early stopping tracks Rank-1 improvement counted in validation events, not raw epochs.
- W&B logging is guarded by `config.USE_WANDB` and wrapped in try/except so training continues if W&B is unavailable.
