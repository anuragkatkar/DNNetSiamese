"""
Configuration file for DNNet training.
All hyperparameters are taken directly from the paper:
  "Dog Nose-Print Identification Using Deep Neural Networks"
  Bae et al., IEEE Access 2021
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_TRAIN       = "./split/train"          # root dir: each subdir = one dog ID
DATA_VAL         = "./split/val"          # root dir: each subdir = one dog ID
CHECKPOINT_DIR   = "./checkpoints"
LOG_DIR          = "./logs"

# ── Dataset ───────────────────────────────────────────────────────────────────
# IMAGE_SIZE      = 256                  # paper: resize to 256×256
IMAGE_SIZE      = 512                  # paper: resize to 256×256
MIN_PIXELS      = 640                  # paper: only images with >640px included
SEED            = 42                   # fixed seed for reproducible splits
VAL_FOLD        = 0                    # which of the 5 folds to use as val
NUM_FOLDS       = 5                    # paper: 5-fold cross-validation
NUM_WORKERS     = 2

# ── Model ─────────────────────────────────────────────────────────────────────
# BACKBONE        = "resnet152"          # paper backbone
BACKBONE        = "resnet18"          # paper backbone
EMBEDDING_DIM   = 128                  # paper: 1024-dimensional embedding vector
EXTRA_CHANNELS  = [512, 256]           # two extra conv blocks after ResNet
FREEZE_BACKBONE_EPOCHS = 0            # set 0 to disable (paper behaviour)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64                   # paper
NUM_EPOCHS      = 150                  # paper
PATIENCE        = 50                   # early stop after N non-improving validations (0 disables)
LR_CONTRASTIVE  = 5e-5                 # paper: initial LR 0.0001
USE_AMP = False

# Adam for contrastive/feature branch (paper: β1=0.5, β2=0.999)
ADAM_BETA1      = 0.5
ADAM_BETA2      = 0.999

# SGD for ArcFace branch (paper: momentum=0.9, weight_decay=0.0005)
SGD_MOMENTUM    = 0.9
SGD_WEIGHT_DECAY = 5e-4
LR_ARCFACE      = 1e-4

# LR schedule: fixed for first 100 epochs, linear decay to 0 for next 100
LR_WARMUP_EPOCHS = 5

# ── Loss Functions ────────────────────────────────────────────────────────────
USE_CONTRASTIVE = False
USE_COSINE      = True
USE_ARCFACE     = True

# ── Loss hyperparameters ───────────────────────────────────────────────────────
CONTRASTIVE_MARGIN = 1.0               # paper: m=2 for contrastive loss
COSINE_MARGIN = 1.5
ARCFACE_SCALE      = 30.0             # paper: s=30
ARCFACE_MARGIN     = 0.5              # paper: m=0.5

# ── Logging ───────────────────────────────────────────────────────────────────
PROJECT_NAME    = "DNNet-Dog-NosePrint"
USE_WANDB       = True                 # set False to disable W&B logging

# ── Evaluation ────────────────────────────────────────────────────────────────
RANK_K          = [1, 5]               # Rank-1 and Rank-5 accuracy
FAR_THRESHOLDS  = [0.001, 0.0001]      # VR@FAR=0.1% and VR@FAR=0.01%

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
