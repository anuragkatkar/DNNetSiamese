# DNNet — Dog Nose-Print Identification

Implementation of **"Dog Nose-Print Identification Using Deep Neural Networks"**
(Bae, Pak & Lee, IEEE Access 2021, DOI: 10.1109/ACCESS.2021.3068517).

---

## Project Structure

```
dnnet/
├── configs/
│   └── config.py           # All hyperparameters (edit this first)
├── data/
│   └── dataset.py          # Dataset, Siamese pair sampling, k-fold splits
├── models/
│   ├── feature_extraction.py  # ResNet-152 backbone + extra conv blocks
│   ├── attention.py           # Dual Attention Network (channel + spatial)
│   ├── dnnet.py               # Full Siamese DNNet model
│   └── losses.py              # ContrastiveLoss, ArcFaceLoss, TotalLoss
├── utils/
│   ├── evaluation.py       # Rank-K, VR@FAR, ROC, t-SNE, confusion matrix
│   ├── schedulers.py       # Fixed + linear-decay LR schedule
│   ├── checkpoint.py       # Save / load checkpoints
│   └── visualization.py   # t-SNE, ROC, confusion matrix plots
├── train.py                # Training loop + cross-validation runner
├── inference.py            # Evaluation & single-query identification
└── requirements.txt
```

---

## Dataset Format

```
dataset/
├── dog_001/
│   ├── img_01.jpg
│   ├── img_02.jpg
│   └── ...            (≥ 2 images per dog required)
├── dog_002/
│   └── ...
└── ...
```

Each subdirectory name is treated as a unique dog identity.
Every image in a subdirectory belongs to that identity.

---

## Setup

```bash
pip install -r requirements.txt
```

Configure paths and hyperparameters in `configs/config.py`:

```python
DATA_ROOT      = "./dataset"      # path to your dataset
CHECKPOINT_DIR = "./checkpoints"
USE_WANDB      = True             # set False to disable W&B
PATIENCE       = 0                # early stop after N non-improving validations (0 disables)
```

Log in to Weights & Biases (if enabled):

```bash
wandb login
```

---

## Training

### Single fold (quick start)
```bash
python train.py --fold 0
```

### Full 5-fold cross-validation (paper evaluation)
```bash
python train.py --cv
```

### Resume training
```bash
python train.py --fold 0 --resume checkpoints/fold_0/epoch_0099.pth
```

---

## Evaluation

### Evaluate a checkpoint on a validation fold
```bash
python inference.py eval \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --data_root  ./dataset \
    --fold       0
```

### Identify a query image against a gallery
```bash
python inference.py identify \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --gallery    ./dataset \
    --query      ./query_nose.jpg \
    --top_k      5
```

---

## Analysis

### Analyse a checkpoint on data
```bash
python analyse.py embed \
    --checkpoint checkpoints/fold_0/best_model.pth \
    --image_dir ./dir \
    --output_dir ./analysis
```

---

## Architecture Summary

```
Input (B, 3, 256, 256)
    │
    ▼  Feature Extraction Module
    ResNet-152 (conv1 → layer4, no GAP/FC)   → (B, 2048, 8, 8)
    ExtraConvBlock(2048 → 512)               → (B,  512, 8, 8)
    ExtraConvBlock( 512 → 256)               → (B,  256, 8, 8)
    │
    ▼  Attention Module (DAN)
    ChannelAttentionModule                   → (B, 256, 8, 8)
    SpatialAttentionModule                   → (B, 256, 8, 8)
    Concatenate [channel, spatial, original] → (B, 768, 8, 8)
    GlobalAveragePooling                     → (B, 768)
    FC(768 → 1024) + L2-normalise            → (B, 1024)
    │
    ▼  Siamese Network
    anchor branch ──┐
                    ├──► ContrastiveLoss  (eq. 7)
    pair   branch ──┘
    both branches → ArcFaceLinear → ArcFaceLoss (eq. 8)
    Total Loss = L_con + ½(L_arc_anchor + L_arc_pair)  (eq. 9)
```

---

## Optimisers (paper §IV-B)

| Component        | Optimiser | LR     | Params              |
|-----------------|-----------|--------|---------------------|
| Feature + Attn  | Adam      | 1e-4   | β1=0.5, β2=0.999    |
| ArcFace head    | SGD       | 1e-4   | momentum=0.9, wd=5e-4 |

**LR schedule**: constant for epochs 0–99, linear decay to 0 for epochs 100–199.

---

## Expected Results (paper Table 4, ResNet-152 + all modules)

| Metric            | Paper    |
|-------------------|----------|
| Rank-1 Accuracy   | 98.972%  |
| VR @ FAR=0.1%     | 72.2%    |
| VR @ FAR=0.01%    | 63.5%    |
