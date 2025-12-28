import os
import sys
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（上层目录）
root_dir = os.path.dirname(current_dir)
# 将根目录添加到 Python 搜索路径
sys.path.append(root_dir)

from models import build_text_encoder
from utils import extract_image_features, device


class Flickr8kDataset:
    def __init__(self, dataset_path: Path, preprocess):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'Images'
        self.names = sorted([p.stem for p in self.images_dir.glob('*.jpg')])
        self.preprocess = preprocess
        self.split = 'flickr8k'

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        from PIL import Image
        name = self.names[idx]
        path = self.images_dir / f"{name}.jpg"
        image = Image.open(path).convert('RGB')
        pixel_values = self.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return {'image': pixel_values, 'image_name': name}


def build_index(dataset_path: str, out_dir: str, clip_model_name: str = 'large', cache_dir: str = './hf_models', batch_size: int = 32, num_workers: int = 4):
    args = argparse.Namespace(clip_model_name=clip_model_name, cache_dir=cache_dir, mixed_precision='fp16')
    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)
    image_encoder = image_encoder.float().to(device)

    dataset = Flickr8kDataset(Path(dataset_path), clip_preprocess)

    # extract features
    features, names = extract_image_features(dataset, image_encoder, batch_size=batch_size, num_workers=num_workers)

    # normalize
    features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy().astype('float32')

    # try to build FAISS index
    try:
        import faiss
    except Exception as e:
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
    faiss.write_index(index, str(out_dir_path / 'flickr8k.index'))
    with open(out_dir_path / 'names.pkl', 'wb') as f:
        pickle.dump(names, f)

    print(f"FAISS index and names saved to {out_dir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True, help='Path to Flickr8k dataset root (contains Images/)')
    parser.add_argument('--out-dir', required=True, help='Output directory to save index and names')
    parser.add_argument('--clip-model-name', default='large')
    parser.add_argument('--cache-dir', default='./hf_models')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    build_index(args.dataset_path, args.out_dir, args.clip_model_name, args.cache_dir, args.batch_size, args.num_workers)
