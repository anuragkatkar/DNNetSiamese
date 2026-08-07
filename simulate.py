from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import shutil
import torch
from configs import config
from data.dataset import get_val_transforms
from PIL import Image

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


# Simulation
@torch.no_grad()
def simulate(threshold, source_dir, save_dir):

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

