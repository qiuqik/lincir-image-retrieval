import os
import sys
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

# Add repo root to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from models import build_text_encoder
from utils import extract_image_features, device


def build_index_from_split(split_json: str, images_root: str, out_dir: str, clip_model_name: str = 'large', cache_dir: str = './hf_models', batch_size: int = 32, num_workers: int = 4):
    args = argparse.Namespace(clip_model_name=clip_model_name, cache_dir=cache_dir, mixed_precision='fp16')
    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)
    image_encoder = image_encoder.float().to(device)

    # build a dataset-like wrapper reading images from split
    class EUFCCDataset:
        def __init__(self, split_path: Path, images_root: Path, preprocess):
            import json
            with open(split_path, 'r', encoding='utf-8') as f:
                self.split = json.load(f)
            self.ids = list(self.split.keys())
            self.images_root = Path(images_root)
            self.preprocess = preprocess
            self.split_name = 'eufcc'

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            from PIL import Image, ImageFile
            # allow loading truncated images (tolerate some corrupted files)
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            name = self.ids[idx]
            # resolve path from split mapping
            rel = self.split[name]
            p = Path(rel)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent / rel
            if not p.exists():
                # fallback to images_root/name.jpg
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    cand = Path(images_root) / f"{name}{ext}"
                    if cand.exists():
                        p = cand
                        break
            try:
                image = Image.open(p)
                image = image.convert('RGB')
            except Exception:
                # on any error (truncated/corrupt), substitute a black placeholder
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            pixel_values = self.preprocess(image, return_tensors='pt')['pixel_values'][0]
            return {'image': pixel_values, 'image_name': name}

    dataset = EUFCCDataset(Path(split_json), Path(images_root), clip_preprocess)

    # extract features
    features, names = extract_image_features(dataset, image_encoder, batch_size=batch_size, num_workers=num_workers)

    # normalize
    features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy().astype('float32')

    # try to build FAISS index
    try:
        import faiss
    except Exception:
        print('faiss not installed. Save features and names to disk. To enable fast search install faiss-cpu: pip install faiss-cpu')
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        np.save(out_dir_path / 'features.npy', features)
        with open(out_dir_path / 'names.pkl', 'wb') as f:
            pickle.dump(names, f)
        print(f"Saved features and names to {out_dir_path}. You can build a FAISS index after installing faiss.")
        return

    d = features.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(features)
    index.add(features)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    # name the index file eufcc.index
    faiss.write_index(index, str(out_dir_path / 'eufcc.index'))
    with open(out_dir_path / 'names.pkl', 'wb') as f:
        pickle.dump(names, f)

    print(f"FAISS index and names saved to {out_dir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-json', default='./test/split.eufcc.json', help='Path to split.eufcc.json')
    parser.add_argument('--images-root', default='./data/test_ood/EUROPEANA/images', help='Images root folder')
    parser.add_argument('--out-dir', default='../experiment/eufcc_index', help='Output directory to save index and names')
    parser.add_argument('--clip-model-name', default='large')
    parser.add_argument('--cache-dir', default='./hf_models')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    build_index_from_split(args.split_json, args.images_root, args.out_dir, args.clip_model_name, args.cache_dir, args.batch_size, args.num_workers)
