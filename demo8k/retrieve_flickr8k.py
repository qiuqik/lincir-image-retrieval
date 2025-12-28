import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（上层目录）
root_dir = os.path.dirname(current_dir)
# 将根目录添加到 Python 搜索路径
sys.path.append(root_dir)

from models import build_text_encoder, Phi
from utils import device
from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF


def load_index(index_dir: str):
    try:
        import faiss
    except Exception:
        raise RuntimeError('faiss is required to load index. Install faiss-cpu')
    index_path = Path(index_dir) / 'flickr8k.index'
    idx = faiss.read_index(str(index_path))
    with open(Path(index_dir) / 'names.pkl', 'rb') as f:
        names = pickle.load(f)
    return idx, names


class SimplePreprocess:
    """Adapter to call HF processor signature like preprocess(image, return_tensors='pt')"""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image: Image.Image, return_tensors='pt'):
        return self.processor(images=image, return_tensors=return_tensors)

import requests

def translate_to_english(text):
    """使用 DeepSeek API 将中文翻译为英文"""
    # 检查是否包含中文字符，如果没有则直接返回，节省 API 调用
    if not any('\u4e00' <= char <= '\u9fff' for char in text):
        return text

    api_key = "sk-xx"
    api_url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a professional translator. Translate the following user input to concise English image descriptions. Return only the translated text."},
            {"role": "user", "content": text}
        ],
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        translated_text = response.json()['choices'][0]['message']['content'].strip()
        print(f"[Translate] Original: {text} -> Translated: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"[Translate] Error: {e}")
        return text # 翻译失败则返回原文本，防止程序崩溃

def retrieve_by_image(image_path: str, index, names: List[str], image_encoder, topk: int = 10):
    image = Image.open(image_path).convert('RGB')
    pixel_values = image_encoder._processor(images=image, return_tensors='pt')['pixel_values'][0] if hasattr(image_encoder, '_processor') else None
    # prefer using processor passed externally; fallback requires user to pass preprocessed image
    if pixel_values is None:
        raise RuntimeError('Please pass an image_encoder built with HF processor; use build_text_encoder and pass its preprocess via SimplePreprocess wrapper')

    image_tensor = pixel_values.to(device).unsqueeze(0).to(image_encoder.dtype)
    with torch.no_grad():
        feat = image_encoder(pixel_values=image_tensor).image_embeds
    feat = feat.cpu().numpy().astype('float32')
    import faiss
    faiss.normalize_L2(feat)
    D, I = index.search(feat, topk)
    results = [(names[int(i)], float(d)) for i, d in zip(I[0], D[0])]
    return results


def retrieve_relative(reference_image_path: str, rel_caption: str, index, names: List[str], image_encoder, text_encoder, phi: Phi, tokenizer, topk: int = 10, l2_normalize: bool = True):
    # compute pseudo token for reference image
    from utils import extract_pseudo_tokens_with_phi
    # create a tiny dataset for single image encoding
    from torchvision.transforms.functional import to_pil_image
    image = Image.open(reference_image_path).convert('RGB')
    proc = getattr(image_encoder, '_processor', None)
    if proc is None:
        raise RuntimeError('image_encoder must have a processor; construct via build_text_encoder and pass preprocess as SimplePreprocess')
    pixel_values = proc(images=image, return_tensors='pt')['pixel_values'][0]
    image_tensor = pixel_values.unsqueeze(0).to(device).to(image_encoder.dtype)
    with torch.no_grad():
        image_features = image_encoder(pixel_values=image_tensor).image_embeds
        if l2_normalize:
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
        pseudo = phi(image_features)

    # prepare caption with placeholder
    input_caption = f"a photo of $ that {rel_caption}"
    tokenized = tokenizer(text=[input_caption], return_tensors='pt', padding='max_length', truncation=True)
    input_ids = tokenized['input_ids'].to(text_encoder.device)

    pseudo_tokens = pseudo.squeeze(0).unsqueeze(0).to(text_encoder.device).type(text_encoder.dtype)
    text_feat = encode_with_pseudo_tokens_HF(text_encoder, input_ids, pseudo_tokens)
    text_feat = text_feat.detach().cpu().numpy().astype('float32')
    import faiss
    faiss.normalize_L2(text_feat)
    D, I = index.search(text_feat, topk)
    results = [(names[int(i)], float(d)) for i, d in zip(I[0], D[0])]
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--mode', choices=['image', 'relative'], default='image')
    parser.add_argument('--image-path', help='Query image path')
    parser.add_argument('--ref-image-path', help='Reference image for relative mode')
    parser.add_argument('--caption', help='Relative caption (for relative mode)')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--clip-model-name', default='large')
    parser.add_argument('--phi-checkpoint', default='./experiment/checkpoints/phi_best.pt')
    parser.add_argument('--cache-dir', default='./hf_models')
    args = parser.parse_args()

    idx, names = load_index(args.index_dir)

    # build models
    args_ns = argparse.Namespace(clip_model_name=args.clip_model_name, cache_dir=args.cache_dir, mixed_precision='fp16')
    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args_ns)
    image_encoder = image_encoder.float().to(device)
    text_encoder = text_encoder.float().to(device)

    # attach processor to image_encoder for helper use
    image_encoder._processor = clip_preprocess

    if args.mode == 'image':
        if not args.image_path:
            raise SystemExit('Provide --image-path for image mode')
        results = retrieve_by_image(args.image_path, idx, names, image_encoder, topk=args.topk)
        print('Results:', results)
    else:
        if not args.ref_image_path or not args.caption:
            raise SystemExit('Provide --ref-image-path and --caption for relative mode')
        phi = Phi(input_dim=text_encoder.config.projection_dim,
                  hidden_dim=text_encoder.config.projection_dim * 4,
                  output_dim=text_encoder.config.hidden_size, dropout=0.5).to(device)
        phi.load_state_dict(torch.load(args.phi_checkpoint, map_location=device)[phi.__class__.__name__])
        phi = phi.eval()
        results = retrieve_relative(args.ref_image_path, args.caption, idx, names, image_encoder, text_encoder, phi, tokenizer, topk=args.topk)
        print('Results:', results)
