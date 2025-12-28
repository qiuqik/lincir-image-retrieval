#!/usr/bin/env python3
"""Evaluate EUFCC_CIR using CLIP (large) and optional phi checkpoint.

Workflow:
- load `test/split.eufcc.json` and `test/cap.eufcc.json`
- encode images (per-members) with CLIP image encoder
- encode captions with CLIP text encoder
- for each caption, compute similarity to its `img_set.members` and compute rank of `reference`
- aggregate R@1/5/10 and MRR, save results to `test/results.json`.

Dependencies: torch, transformers, pillow, tqdm
Run example:
  python3 evaluate_eufcc.py --split test/split.eufcc.json --cap test/cap.eufcc.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import clip
from models import build_text_encoder, Phi
from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF
from utils import device as utils_device


def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def batch(iterable, n=32):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:i+n]


def main(args):
    root = Path(__file__).resolve().parent
    split_path = Path(args.split)
    cap_path = Path(args.cap)
    out_dir = root / 'test'
    out_dir.mkdir(exist_ok=True)

    split_map = load_json(split_path)
    cap_list = load_json(cap_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # choose evaluation path: standard CLIP HF or phi-based pseudo-token path
    if args.use_phi:
        # build HF CLIP components used by project (vision + text)
        args_ns = argparse.Namespace(clip_model_name=args.clip_model_name, cache_dir=args.cache_dir, mixed_precision='fp16')
        image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args_ns)
        image_encoder = image_encoder.float().to(device).eval()
        text_encoder = text_encoder.float().to(device).eval()
        image_encoder._processor = clip_preprocess

        # load phi
        if not args.phi_checkpoint:
            raise SystemExit('When --use_phi is set you must provide --phi_checkpoint')
        phi = Phi(input_dim=text_encoder.config.projection_dim,
                  hidden_dim=text_encoder.config.projection_dim * 4,
                  output_dim=text_encoder.config.hidden_size, dropout=0.5).to(device)
        phi.load_state_dict(torch.load(args.phi_checkpoint, map_location=device)[phi.__class__.__name__])
        phi = phi.eval()

        # all ids from split
        needed_ids = list(split_map.keys())
        id2path = {iid: Path(root) / split_map[iid] for iid in needed_ids}

        # encode all images to get image embeddings (index)
        print(f'Encoding {len(id2path)} images with HF CLIP vision encoder (full search)...')
        image_embeds = {}
        for batch_ids in batch(needed_ids, n=args.img_batch_size):
            images = []
            real_ids = []
            for iid in batch_ids:
                p = id2path[iid]
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    img = Image.new('RGB', (224, 224), color=(0, 0, 0))
                images.append(img)
                real_ids.append(iid)
            inputs = clip_preprocess(images=images, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(device)
            with torch.no_grad():
                feats = image_encoder(pixel_values=pixel_values).image_embeds
            feats = feats.cpu()
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            for iid, emb in zip(real_ids, feats):
                image_embeds[iid] = emb

        all_ids = list(image_embeds.keys())
        all_embs = torch.stack([image_embeds[i] for i in all_ids])

        # precompute pseudo tokens for unique references
        unique_refs = list({it.get('reference') for it in cap_list})
        ref2pseudo = {}
        print(f'Computing pseudo tokens for {len(unique_refs)} references...')
        for batch_refs in batch(unique_refs, n=args.img_batch_size):
            images = []
            real_refs = []
            for rid in batch_refs:
                p = id2path.get(rid, Path(root) / 'data' / 'test_ood' / 'EUROPEANA' / 'images' / f"{rid}.jpg")
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    img = Image.new('RGB', (224, 224), color=(0, 0, 0))
                images.append(img)
                real_refs.append(rid)
            inputs = clip_preprocess(images=images, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(device)
            with torch.no_grad():
                image_feats = image_encoder(pixel_values=pixel_values).image_embeds
            if args.l2_normalize:
                image_feats = torch.nn.functional.normalize(image_feats, dim=-1)
            with torch.no_grad():
                pseudo = phi(image_feats)
            for rid, pvec in zip(real_refs, pseudo.cpu()):
                ref2pseudo[rid] = pvec

        # prepare openai clip tokenizer for token ids used by encode_with_pseudo_tokens_HF
        # clip.tokenize returns LongTensor of token ids (with placeholder $ mapped accordingly)
    else:
        print('Loading HF CLIP model:', args.clip_model)
        model = CLIPModel.from_pretrained(args.clip_model).to(device)
        processor = CLIPProcessor.from_pretrained(args.clip_model)

        # collect all image ids from the split (full-dataset search)
        needed_ids = set(split_map.keys())

        # map id -> image path
        id2path = {}
        for iid in needed_ids:
            if iid in split_map:
                id2path[iid] = Path(root) / split_map[iid]
            else:
                # try images dir fallback
                candidate = root / 'data' / 'test_ood' / 'EUROPEANA' / 'images' / (f"{iid}.jpg")
                id2path[iid] = candidate

        # encode all images in the split
        print(f'Encoding {len(id2path)} images with CLIP (full search)...')
        image_embeds = {}
        ids = list(id2path.keys())
        for batch_ids in batch(ids, n=args.img_batch_size):
            images = []
            real_ids = []
            for iid in batch_ids:
                p = id2path[iid]
                try:
                    img = Image.open(p).convert('RGB')
                    images.append(img)
                    real_ids.append(iid)
                except Exception:
                    images.append(Image.new('RGB', (224, 224), color=(0, 0, 0)))
                    real_ids.append(iid)

            inputs = processor(images=images, return_tensors='pt').to(device)
            with torch.no_grad():
                img_feats = model.get_image_features(**inputs)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            for iid, emb in zip(real_ids, img_feats.cpu()):
                image_embeds[iid] = emb

        # create ordered list and tensor for fast scoring
        all_ids = list(image_embeds.keys())
        all_embs = torch.stack([image_embeds[i] for i in all_ids])  # N x D (cpu)

    # encode captions in batches and evaluate per-entry
    print('Encoding captions and evaluating...')
    ranks = []
    results_per = []
    for batch_items in batch(cap_list, n=args.txt_batch_size):
        texts = [it.get('caption', '') for it in batch_items]
        if args.use_phi:
            # when using phi, build text features by injecting pseudo tokens per reference
            tokenized_inputs = [f"a photo of $ that {it.get('caption','')}" for it in batch_items]
            token_ids = clip.tokenize(tokenized_inputs, context_length=77)
            for i, it in enumerate(batch_items):
                reference = it.get('reference')
                pairid = it.get('pairid')
                # ensure token ids and pseudo tokens are on the text encoder device
                caption_token_ids = token_ids[i].unsqueeze(0).to(text_encoder.device)
                pseudo = ref2pseudo.get(reference)
                if pseudo is not None:
                    pseudo = pseudo.to(text_encoder.device)
                if pseudo is None:
                    ranks.append(None)
                    results_per.append({'pairid': pairid, 'rank': None})
                    continue
                # encode with HF text encoder using pseudo tokens
                with torch.no_grad():
                    text_feat = encode_with_pseudo_tokens_HF(text_encoder, caption_token_ids, pseudo.unsqueeze(0))
                text_feat = text_feat.cpu()
                text_feat = text_feat / text_feat.norm(p=2)
                sims = (all_embs @ text_feat.squeeze(0)).numpy()
                order = sims.argsort()[::-1]
                try:
                    ref_idx = all_ids.index(reference)
                except ValueError:
                    ref_idx = -1
                if ref_idx == -1:
                    ranks.append(None)
                    results_per.append({'pairid': pairid, 'rank': None})
                    continue
                rank_pos = int((order == ref_idx).nonzero()[0][0])
                ranks.append(rank_pos)
                results_per.append({'pairid': pairid, 'rank': rank_pos})
        else:
            inputs = processor(text=texts, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                txt_feats = model.get_text_features(**inputs)
            txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

            for i, it in enumerate(batch_items):
                caption_emb = txt_feats[i].cpu()
                reference = it.get('reference')

                # compute similarity against all images
                sims = (all_embs @ caption_emb).numpy()
                order = sims.argsort()[::-1]

                # find index of reference in all_ids
                try:
                    ref_idx = all_ids.index(reference)
                except ValueError:
                    ref_idx = -1

                if ref_idx == -1:
                    ranks.append(None)
                    results_per.append({'pairid': it.get('pairid'), 'rank': None})
                    continue

                # rank position of reference in sorted order
                rank_pos = int((order == ref_idx).nonzero()[0][0])
                ranks.append(rank_pos)
                results_per.append({'pairid': it.get('pairid'), 'rank': rank_pos})

    # aggregate metrics (only entries with a rank)
    valid_ranks = [r for r in ranks if isinstance(r, int) and r >= 0]
    n = len(valid_ranks)
    metrics = {}
    if n == 0:
        metrics = {'n': 0}
    else:
        metrics['n'] = n
        metrics['R@1'] = sum(1 for r in valid_ranks if r < 1) / n
        metrics['R@5'] = sum(1 for r in valid_ranks if r < 5) / n
        metrics['R@10'] = sum(1 for r in valid_ranks if r < 10) / n
        metrics['R@50'] = sum(1 for r in valid_ranks if r < 50) / n
        metrics['MRR'] = sum(1.0 / (r + 1) for r in valid_ranks) / n

    out = {'metrics': metrics, 'per_pair': results_per}
    out_path = out_dir / 'results.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('Done. Results saved to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test/split.eufcc.json')
    parser.add_argument('--cap', default='test/cap.eufcc.json')
    parser.add_argument('--clip_model', default='openai/clip-vit-large-patch14')
    parser.add_argument('--use_phi', action='store_true', help='Use Phi to generate pseudo tokens and evaluate with pseudo-token pipeline')
    parser.add_argument('--clip_model_name', default='large', help='short name for build_text_encoder (large|huge|giga)')
    parser.add_argument('--cache_dir', default='./hf_models')
    parser.add_argument('--l2_normalize', action='store_true', help='L2 normalize image features before passing to Phi')
    parser.add_argument('--img_batch_size', type=int, default=32)
    parser.add_argument('--txt_batch_size', type=int, default=64)
    parser.add_argument('--phi_checkpoint', default=None, help='optional phi checkpoint path (not used by default)')
    args = parser.parse_args()
    main(args)
