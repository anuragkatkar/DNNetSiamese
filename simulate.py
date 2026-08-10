from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import shutil
import torch
from configs import config
from data.dataset import get_val_transforms
from PIL import Image
from pathlib import Path

if config.MODEL == 'alpha':
    from models.alpha import build_model
elif config.MODEL == 'beta':
    from models.beta import build_model
else:
    raise ImportError

# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    embedding_dim    = 128
    num_classes_ckpt = state["arcface_head.weight"].shape[0]

    _orig = config.EMBEDDING_DIM
    config.EMBEDDING_DIM = embedding_dim
    model = build_model(num_classes=num_classes_ckpt, cfg=config).to(device)
    config.EMBEDDING_DIM = _orig

    model.load_state_dict(state)
    model.eval()
    
    return model

# ── Save Embedding Projector TSV ──────────────────────────────────────────────────

def save_tsv_projector(
    embeddings:   np.ndarray,
    labels:       np.ndarray,
    paths:        list,
    out_dir:      str,
):
    """
    Save embeddings and metadata in the two-file TSV format expected by
    TensorBoard Embedding Projector (https://projector.tensorflow.org):

      embeddings_projector.tsv  – (N × D) values, tab-separated, no header
      metadata_projector.tsv   – header row + one row per point

    Load in TensorBoard:
      tensorboard --logdir <out_dir>
    Or upload both files at https://projector.tensorflow.org.
    """
    os.makedirs(out_dir, exist_ok=True)
    emb_path  = os.path.join(out_dir, "embeddings_projector.tsv")
    meta_path = os.path.join(out_dir, "metadata_projector.tsv")

    # Embedding vectors — no header, values tab-separated, no quoting
    np.savetxt(emb_path, embeddings, delimiter="\t", fmt="%.6f")

    # Metadata — first row is column headers (triggers multi-column mode in
    # TensorBoard Projector), then one data row per embedding.
    # Written directly (no csv.writer) to guarantee no quoting is added.
    lines = ["label\tfilename"]
    for lbl, path in zip(labels, paths):
        lines.append(f"{lbl}\t{path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ── Simulation ───────────────────────────────────────────────────────────

@torch.no_grad()
def _simulate(threshold, source_dir, save_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 1 - threshold
    os.makedirs(save_dir, exist_ok=True)
    model = load_model(args.checkpoint, device)
    tfm   = get_val_transforms()
    image_names = os.listdir(source_dir)

    all_embeddings, all_paths = [], []
    for img in image_names:
        path = os.path.join(source_dir, img)
        try:
            img    = Image.open(path).convert("RGB")
            tensor = tfm(img).unsqueeze(0).to(device)
            emb    = model.get_embedding(tensor)
            all_embeddings.append(emb.squeeze(0).cpu().numpy())

            all_paths.append(path)
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    embeddings = np.stack(all_embeddings, axis=0)    
    sim_matrix = cosine_similarity(np.array(embeddings))

    id_ = 10000
    folder_name = f"DOG_{id_}"
    folder_names = []

    for i in range(sim_matrix.shape[0])[0:]:
        sim_matrix[i][i] = -10

    for i in range(sim_matrix.shape[0])[0:]:

        max_ = sim_matrix[i][0:i+1].max()
        argmax_ = sim_matrix[i][0:i+1].argmax()

        if max_ == -10:
            print('First Image')
            folder_path = os.path.join(save_dir, folder_name)
            folder_names.append(folder_name)
            # print(folder_names)
            os.makedirs(folder_path, exist_ok=True)
            shutil.copy(os.path.join(source_dir, image_names[i]), os.path.join(folder_path, folder_name + '__' + image_names[i]))
                
        
        elif max_ >= threshold:
            print('Match')
            folder_name = folder_names[argmax_]
            folder_names.append(folder_name)
            # print(folder_names, argmax_)
            folder_path = os.path.join(save_dir, folder_name)
            shutil.copy(os.path.join(source_dir, image_names[i]), os.path.join(folder_path, folder_name + '__' + image_names[i]))

        else:
            print('No Match')
            id_ = 10000
            id_ = id_ + i
            folder_name = f"DOG_{id_}"
            folder_names.append(folder_name)

            folder_path = os.path.join(save_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            shutil.copy(os.path.join(source_dir, image_names[i]), os.path.join(folder_path, folder_name + '__' + image_names[i]))
        

@torch.no_grad()
def simulate(threshold, source_dir, save_dir):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    model = load_model(args.checkpoint, device)
    tfm   = get_val_transforms()
    image_names = os.listdir(source_dir)

    id_ = 10000
    folder_name = f"DOG_{id_}"
    folder_names = []

    embeddings = []
    all_embedding = []
    all_labels = []
    paths = []
    for i, img_ in enumerate(image_names):
        path = os.path.join(source_dir, img_)
        try:
            img    = Image.open(path).convert("RGB")
            tensor = tfm(img).unsqueeze(0).to(device)
            emb    = model.get_embedding(tensor)
            emb    = emb.squeeze(0).cpu().numpy()

            if len(embeddings) == 0:
                embeddings = list(embeddings)
                embeddings.append(emb)
                all_embedding.append(emb)
                all_labels.append(folder_name)

                print('First Image')
                folder_names.append(folder_name)
                folder_path = os.path.join(save_dir, folder_name)
                
                os.makedirs(folder_path, exist_ok=True)
                shutil.copy(path, os.path.join(folder_path, folder_name + 'A' + '__' + img_))
                continue

            embeddings = np.array(embeddings)
            similarity = embeddings @ emb.T
            distance = 1 - similarity
            min_ = distance.min()
            argmin_ = distance.argmin()

            if min_ > threshold:
                embeddings = list(embeddings)
                embeddings.append(emb)
                all_embedding.append(emb)

                id_ = 10000
                id_ = id_ + i
                folder_name = f"DOG_{id_}"
                folder_names.append(folder_name)
                all_labels.append(folder_name)

                folder_path = os.path.join(save_dir, folder_name)

                os.makedirs(folder_path, exist_ok=True)
                shutil.copy(path, os.path.join(folder_path, folder_name + 'A' + '__' + img_))
            else:
                print('Match')
                folder_name = folder_names[argmin_]
                folder_path = os.path.join(save_dir, folder_name)

                all_embedding.append(emb)
                all_labels.append(folder_name)

                os.makedirs(folder_path, exist_ok=True)
                shutil.copy(path, os.path.join(folder_path, folder_name + '__' + img_))

            embeddings = np.array(embeddings)
            print(similarity, embeddings.shape, emb.shape)

            paths.append(path)
        except Exception as e:
                print(f"  Skipping {path}: {e}")    
    save_tsv_projector(np.array(all_embedding), np.array(all_labels), image_names, save_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flatten_dir", required=True)
    parser.add_argument("--output_dir", default="./simulation")
    parser.add_argument("--threshold", type=float, required=True, default=0.4)

    args = parser.parse_args()
    simulate(args.threshold, args.flatten_dir, args.output_dir)

