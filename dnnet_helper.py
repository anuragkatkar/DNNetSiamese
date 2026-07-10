"""
dnnet_helper.py
===============
Drop this file into your dnnet/ folder and import it in a Jupyter notebook.

Quick start
-----------
from dnnet_helper import DNNetHelper

dn = DNNetHelper("checkpoints/fold_0/best_model.pth")

# Single image
emb = dn.embed_image("path/to/nose.jpg")          # (128,)

# Multiple images as a list of paths
embs = dn.embed_images(["img1.jpg", "img2.jpg"])   # (2, 128)

# Whole folder
embs, paths = dn.embed_folder("./my_nose_photos")  # (N, 128), [paths]

# Similarity between two images
score = dn.similarity("img1.jpg", "img2.jpg")       # float, 0-1

# Find closest match in a folder
best = dn.find_closest("query.jpg", "./gallery")
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Make sure the dnnet package root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from configs import config
from models.dnnet import build_model


# ── Image transform (same as validation) ──────────────────────────────────────

def _get_transform(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Helper class ──────────────────────────────────────────────────────────────

class DNNetHelper:
    """
    Thin wrapper around a trained DNNet checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to best_model.pth (or any .pth saved by train.py).
    device : str, optional
        'cuda', 'cpu', or 'auto' (default – picks GPU if available).
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = checkpoint_path
        self.transform = _get_transform()
        self.model = self._load(checkpoint_path)
        print(f"Model ready on {self.device}  |  embedding dim = {self.embedding_dim}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self, path: str) -> torch.nn.Module:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model_state"]

        # Infer dims from saved weights so config mismatches never matter
        embedding_dim = state["dnnet.attention.fc.weight"].shape[0]
        num_classes   = state["arcface_head.weight"].shape[0]
        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes

        # Temporarily patch config so build_model uses the right dim
        _orig = config.EMBEDDING_DIM
        config.EMBEDDING_DIM = embedding_dim
        model = build_model(num_classes=num_classes, cfg=config).to(self.device)
        config.EMBEDDING_DIM = _orig

        model.load_state_dict(state)
        model.eval()
        return model

    def _load_image(self, path: str) -> torch.Tensor:
        """Load one image → (1, 3, H, W) tensor on device."""
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _embed_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Run a batch tensor through the model → numpy array."""
        emb = self.model.get_embedding(tensor)   # (B, D)
        return emb.cpu().numpy()

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Embed a single image.

        Returns
        -------
        np.ndarray of shape (embedding_dim,)
        """
        tensor = self._load_image(image_path)
        return self._embed_tensor(tensor).squeeze(0)   # (D,)

    @torch.no_grad()
    def embed_images(self, image_paths: list, batch_size: int = 16) -> np.ndarray:
        """
        Embed a list of image paths in batches.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim)
        """
        all_embs = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors = torch.cat([self._load_image(p) for p in batch_paths], dim=0)
            all_embs.append(self._embed_tensor(tensors))
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def embed_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Embed a PIL Image directly (useful when you already have the image in memory).

        Returns
        -------
        np.ndarray of shape (embedding_dim,)
        """
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return self._embed_tensor(tensor).squeeze(0)

    @torch.no_grad()
    def embed_folder(self, folder_path: str, batch_size: int = 16):
        """
        Embed every image inside a folder (non-recursive).

        Returns
        -------
        embeddings : np.ndarray of shape (N, embedding_dim)
        paths      : list of str, matching order of embeddings
        """
        folder = Path(folder_path)
        paths  = sorted([
            str(f) for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        if not paths:
            raise FileNotFoundError(f"No images found in {folder_path}")

        print(f"Embedding {len(paths)} images from {folder_path} …")
        embs = self.embed_images(paths, batch_size=batch_size)
        return embs, paths

    def similarity(self, path1: str, path2: str) -> float:
        """
        Cosine similarity between two nose-print images.
        Returns a float in [-1, 1]; closer to 1 = more similar.
        """
        e1 = self.embed_image(path1)
        e2 = self.embed_image(path2)
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))

    def distance(self, path1: str, path2: str) -> float:
        """Euclidean distance between two embeddings (lower = more similar)."""
        e1 = self.embed_image(path1)
        e2 = self.embed_image(path2)
        return float(np.linalg.norm(e1 - e2))

    def find_closest(self, query_path: str, gallery_path: str, top_k: int = 5):
        """
        Find the top-k closest images in a gallery folder to a query image.

        Returns
        -------
        list of dicts: [{'path': str, 'distance': float, 'similarity': float}, ...]
        """
        query_emb          = self.embed_image(query_path)
        gallery_embs, paths = self.embed_folder(gallery_path)

        distances    = np.linalg.norm(gallery_embs - query_emb, axis=1)
        similarities = gallery_embs @ query_emb  # already L2-normalised

        top_k_idx = np.argsort(distances)[:top_k]
        return [
            {
                "rank":       i + 1,
                "path":       paths[idx],
                "distance":   float(distances[idx]),
                "similarity": float(similarities[idx]),
            }
            for i, idx in enumerate(top_k_idx)
        ]
