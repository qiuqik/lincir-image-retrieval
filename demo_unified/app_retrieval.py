from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import argparse
import os
import glob
import sys
import json
import pickle
import faiss
import torch
import numpy as np
from PIL import Image
import requests

# 导入项目内部模块
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# 尝试导入核心检索模块
try:
    from models import build_text_encoder, Phi
    from utils import device
    from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF
except ImportError as e:
    print(f"警告：导入项目核心模块失败 - {e}，请确保项目路径正确")
    raise e

app = Flask(__name__)

# 数据集路径配置
DATASET_PATHS = {
  'flickr8k': {
    'index_dir': '/home/yqzheng/projects/lincir/experiment/flickr8k_index/flickr8k.index',
    'image_root': '/home/yqzheng/projects/lincir/datasets/Flickr8k/Images'
  },
  'flickr30k': {
    'index_dir': '/home/yqzheng/projects/lincir/experiment/flickr30k_index/flickr30k.index',
    'image_root': '/home/yqzheng/projects/lincir/datasets/Flickr30k/flickr30k-images'
  },
  'eufcc': {
    'index_dir': '/home/yqzheng/projects/lincir/experiment/eufcc_index/eufcc.index',
    'image_root': '/home/yqzheng/projects/lincir/EUFCC_CIR/data/test_ood/EUROPEANA/images',
  }
}

# 上传目录配置
UPLOADS_DIR = current_dir / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 全局缓存：避免重复加载索引和模型
MODEL_CACHE = {}
INDEX_CACHE = {}

# -------------------------- 前端界面 --------------------------
HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
  <title>Unified Image Retrieval System</title>
  <style>
    body { 
      background-color: #f8f9fa; 
      min-height: 100vh;
    }
    .container {
      max-width: 1200px;
    }
    .result-img { 
      width:100%; 
      height: 200px; 
      object-fit:cover; 
      transition: transform 0.3s ease;
    }
    .result-img:hover {
      transform: scale(1.02);
    }
    .query-img { 
      width:256px; 
      height:256px; 
      object-fit:cover; 
      border: 3px solid #0d6efd;
    }
    .card {
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border: none;
      transition: box-shadow 0.3s ease;
    }
    .card:hover {
      box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .form-container {
      background-color: #fff;
      padding: 2rem;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      margin-bottom: 2rem;
    }
    .info-card {
      background-color: #fff;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .loading-spinner {
      display: none;
      text-align: center;
      padding: 3rem 0;
    }
    .score-badge {
      background-color: #0d6efd;
      color: white;
      font-size: 0.85rem;
    }
    .dataset-badge {
      font-size: 0.9rem;
      margin-right: 0.5rem;
    }
  </style>
</head>
<body class="py-4">
<div class="container">
  <h1 class="mb-5 text-center">
    <i class="bi bi-search-heart me-2"></i>Unified Image Retrieval System
  </h1>

  <!-- 表单容器 -->
  <div class="form-container">
    <form id="retrieval-form" method="post" enctype="multipart/form-data" class="row g-3">
      <div class="col-md-2">
        <label class="form-label">Dataset</label>
        <select class="form-select" name="dataset" required>
          <option value="flickr8k" {% if dataset == 'flickr8k' %}selected{% endif %}>Flickr8k</option>
          <option value="flickr30k" {% if dataset == 'flickr30k' %}selected{% endif %}>Flickr30k</option>
          <option value="eufcc" {% if dataset == 'eufcc' %}selected{% endif %}>EUFCC_CIR</option>
        </select>
      </div>
      <div class="col-md-2">
        <label class="form-label">Query Image</label>
        <input class="form-control" type="file" name="query_image" accept="image/*" required>
      </div>
      <div class="col-md-2">
        <label class="form-label">Retrieval Mode</label>
        <select class="form-select" name="mode" required>
          <option value="image" {% if mode == 'image' %}selected{% endif %}>Image-only</option>
          <option value="relative" {% if mode == 'relative' %}selected{% endif %}>Image + Caption</option>
        </select>
      </div>
      <div class="col-md-3">
        <label class="form-label">Caption (for Relative Mode)</label>
        <input class="form-control" type="text" name="caption" placeholder="Enter image description (support Chinese)" value="{{ caption|default('') }}">
      </div>
      <div class="col-md-1">
        <label class="form-label">Top N</label>
        <select class="form-select" name="top_k" required>
          <option value="10" {% if top_k == 10 %}selected{% endif %}>10</option>
          <option value="20" {% if top_k == 20 %}selected{% endif %}>20</option>
          <option value="30" {% if top_k == 30 %}selected{% endif %}>30</option>
          <option value="50" {% if top_k == 50 %}selected{% endif %}>50</option>
        </select>
      </div>
      <div class="col-md-2 d-flex align-items-end">
        <button class="btn btn-primary w-100" type="submit">
          <i class="bi bi-search me-1"></i> Search
        </button>
      </div>
    </form>
  </div>

  <!-- 加载动画 -->
  <div class="loading-spinner" id="loading">
    <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
    <p class="mt-3 text-muted">Searching for similar images, please wait...</p>
  </div>

  <!-- 内容展示区 -->
  <div id="content">
  {% if query %}
    <div class="row g-4 mb-5">
      <!-- 查询图像和信息 -->
      <div class="col-md-3">
        <div class="info-card p-3">
          <h5 class="mb-3 text-center">Query Image</h5>
          <img src="{{ url_for('serve_images', filename=query) }}" class="query-img img-thumbnail mx-auto d-block">
          
          <!-- 检索信息展示 -->
          <div class="mt-4">
            <h6 class="text-secondary mb-2">Retrieval Info</h6>
            <p class="mb-1">
              <span class="badge bg-primary dataset-badge">{{ dataset.upper() }}</span>
              <span class="badge bg-secondary">{{ mode|capitalize }} Mode</span>
            </p>
            <p class="mb-1"><strong>Top N:</strong> {{ top_k }}</p>
            {% if mode == 'relative' %}
            <p class="mb-1"><strong>Input Caption:</strong> {{ caption or 'None' }}</p>
            <p class="mb-1 small text-success">
              <strong>Translated English:</strong> {{ translated_caption or 'None' }}
            </p>
            {% endif %}
          </div>
        </div>
      </div>

      <!-- 检索结果 -->
      <div class="col-md-9">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h5>Retrieval Results (Top {{ top_k }})</h5>
          <span class="badge bg-success">{{ results|length }} results found</span>
        </div>
        <div class="row row-cols-1 row-cols-md-3 row-cols-lg-4 g-3">
          {% for filename, score, stem in results %}
            <div class="col">
              <div class="card h-100">
                <img src="{{ url_for('serve_images', filename=filename) }}" class="card-img-top result-img">
                <div class="card-body d-flex flex-column justify-content-between">
                  <p class="card-text text-truncate">{{ stem }}</p>
                  <div>
                    <span class="badge score-badge">Score: {{ '%.3f'|format(score) }}</span>
                  </div>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>
      </div>
    </div>
  {% endif %}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
  // 表单提交时显示加载动画
  document.getElementById('retrieval-form').addEventListener('submit', function() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('content').style.display = 'none';
  });
</script>
</body>
</html>
'''

# -------------------------- 优化点2：工具函数封装（复用性更强） --------------------------
class SimplePreprocess:
    """Adapter for HF processor to unify the interface（彻底移除images关键字，解决参数报错）"""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image: Image.Image, return_tensors='pt'):
        # 完全不使用images关键字，仅适配单张图像的两种兼容调用方式
        try:
            # 优先：使用位置参数（兼容性最强，适配所有新旧版处理器）
            return self.processor(image, return_tensors=return_tensors)
        except TypeError as e1:
            try:
                # 备选：使用image关键字（新版HF处理器推荐方式）
                return self.processor(image=image, return_tensors=return_tensors)
            except TypeError as e2:
                # 仅抛出明确异常，不触发images参数调用
                raise RuntimeError(f"Processor call failed, neither position param nor 'image' keyword is supported: {e1}, {e2}")

def translate_to_english(text):
    """
    中文转英文（基于DeepSeek API）
    优化：增加空值判断，避免无效调用
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 检查是否包含中文字符
    if not any('\u4e00' <= char <= '\u9fff' for char in text):
        return text.strip()

    api_key = "sk-20e44ae65ccb4c128564a4ecf749b4cf"
    api_url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a professional translator. Translate the following user input to concise English image descriptions. Return only the translated text, no extra explanation."},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "temperature": 0.1
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        translated_text = response.json()['choices'][0]['message']['content'].strip()
        print(f"[Translate] Original: {text} -> Translated: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"[Translate Error] {e}")
        return text.strip()

def load_dataset_index(dataset_name):
    """
    加载数据集索引（带缓存，避免重复加载）
    优化：缓存机制提升重复请求效率
    """
    if dataset_name in INDEX_CACHE:
        return INDEX_CACHE[dataset_name]

    dataset_config = DATASET_PATHS.get(dataset_name)
    if not dataset_config:
        raise ValueError(f"Dataset {dataset_name} not found in configuration")

    index_path = dataset_config['index_dir']
    index_dir = os.path.dirname(index_path)
    names_path = os.path.join(index_dir, 'names.pkl')

    # 加载FAISS索引
    try:
        idx = faiss.read_index(index_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index for {dataset_name}: {e}")

    # 加载名称列表
    try:
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load names.pkl for {dataset_name}: {e}")

    # 存入缓存
    INDEX_CACHE[dataset_name] = (idx, names)
    return idx, names

def build_model_cache(clip_model_name='large', cache_dir='./hf_models'):
    """
    构建模型缓存（绑定修复后的处理器）
    """
    cache_key = f"{clip_model_name}_{cache_dir}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    args_ns = argparse.Namespace(
        clip_model_name=clip_model_name,
        cache_dir=cache_dir,
        mixed_precision='fp16'
    )

    try:
        image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args_ns)
        image_encoder = image_encoder.float().to(device)
        text_encoder = text_encoder.float().to(device)
        # 绑定重构后的SimplePreprocess，彻底杜绝images参数
        image_encoder._processor = SimplePreprocess(clip_preprocess)

        # 加载Phi模型（无需修改，此处无处理器调用）
        phi = Phi(
            input_dim=text_encoder.config.projection_dim,
            hidden_dim=text_encoder.config.projection_dim * 4,
            output_dim=text_encoder.config.hidden_size,
            dropout=0.5
        ).to(device)

        # 加载Phi权重
        phi_ckpt_path = os.path.join(root_dir, 'experiment', 'checkpoints', 'phi_best.pt')
        if os.path.exists(phi_ckpt_path):
            ckpt = torch.load(phi_ckpt_path, map_location=device)
            phi.load_state_dict(ckpt[phi.__class__.__name__])
        phi = phi.eval()

        # 存入缓存
        MODEL_CACHE[cache_key] = (image_encoder, text_encoder, phi, tokenizer)
        return image_encoder, text_encoder, phi, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to build model cache: {e}")

def find_image_file(stem, dataset_name):
    """
    查找图像文件（支持多后缀、多目录搜索）
    优化：灵活的文件查找逻辑，提升兼容性
    """
    # 搜索根目录：上传目录 + 数据集图像根目录
    search_roots = [UPLOADS_DIR]
    dataset_config = DATASET_PATHS.get(dataset_name)
    if dataset_config and 'image_root' in dataset_config:
        search_roots.append(Path(dataset_config['image_root']))

    # 常见图像后缀
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

    # 1. 按stem+后缀精确查找
    for root in search_roots:
        for ext in extensions:
            image_path = root / f"{stem}{ext}"
            if image_path.exists():
                return image_path.name

    # 2. 通配符模糊查找
    for root in search_roots:
        wildcard_path = str(root / f"{stem}.*")
        matches = glob.glob(wildcard_path)
        if matches and Path(matches[0]).exists():
            return Path(matches[0]).name

    return None

# -------------------------- 优化点3：接口优化（稳定性更强） --------------------------
@app.route('/images/<path:filename>')
def serve_images(filename):
    """
    提供图像访问接口（支持多目录搜索）
    优化：健壮的文件查找逻辑，避免404错误
    """
    # 先搜索上传目录
    upload_file = UPLOADS_DIR / filename
    if upload_file.exists():
        return send_from_directory(str(UPLOADS_DIR), filename)

    # 再搜索所有数据集图像目录
    for dataset_config in DATASET_PATHS.values():
        if 'image_root' not in dataset_config:
            continue
        image_root = Path(dataset_config['image_root'])
        # 精确查找
        dataset_file = image_root / filename
        if dataset_file.exists():
            return send_from_directory(str(image_root), filename)
        # 按stem查找
        stem = Path(filename).stem
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            candidate_file = image_root / f"{stem}{ext}"
            if candidate_file.exists():
                return send_from_directory(str(image_root), candidate_file.name)

    # 所有目录未找到
    return ("Image not found", 404)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    主检索接口
    优化：分步处理、异常捕获、状态保留，提升用户体验
    """
    # 初始化默认参数（用于表单状态保留）
    results = []
    query_filename = None
    dataset = request.form.get('dataset', 'flickr30k')
    mode = request.form.get('mode', 'image')
    caption = request.form.get('caption', '').strip()
    top_k = int(request.form.get('top_k', 20))
    translated_caption = None

    if request.method == 'POST':
        # 1. 获取并验证上传文件
        file = request.files.get('query_image')
        if not file or file.filename == '':
            return render_template_string(
                HTML,
                results=results,
                dataset=dataset,
                mode=mode,
                caption=caption,
                top_k=top_k,
                query=query_filename,
                translated_caption=translated_caption
            )

        # 2. 保存上传文件
        orig_filename = secure_filename(file.filename) or f"query_{int(os.time())}.jpg"
        save_path = UPLOADS_DIR / orig_filename
        file.save(save_path)
        query_filename = save_path.name

        try:
            # 3. 加载模型（带缓存）
            image_encoder, text_encoder, phi, tokenizer = build_model_cache()

            # 4. 加载数据集索引（带缓存）
            idx, names = load_dataset_index(dataset)

            # 5. 根据检索模式执行检索
            if mode == 'image':
                # 纯图像检索
                results_raw = retrieve_by_image(
                    str(save_path), idx, names, image_encoder, topk=top_k
                )
            else:
                # 图像+文本相对检索
                translated_caption = translate_to_english(caption)
                if not translated_caption:
                    translated_caption = "a general image"

                results_raw = retrieve_relative(
                    str(save_path), translated_caption, idx, names,
                    image_encoder, text_encoder, phi, tokenizer, topk=top_k
                )

            # 6. 处理检索结果，查找对应的图像文件
            for stem, score in results_raw:
                image_filename = find_image_file(stem, dataset)
                if image_filename:
                    results.append((image_filename, score, stem))

        except Exception as e:
            print(f"[Retrieval Error] {e}")
            # 异常时仍渲染页面，避免前端崩溃
            pass

    # 渲染模板（保留表单状态）
    return render_template_string(
        HTML,
        results=results,
        dataset=dataset,
        mode=mode,
        caption=caption,
        top_k=top_k,
        query=query_filename,
        translated_caption=translated_caption
    )

# -------------------------- 核心检索函数（与参考文件对齐） --------------------------
def retrieve_by_image(image_path: str, index, names: list, image_encoder, topk: int = 10):
    """纯图像检索函数（确保无images关键字调用）"""
    image = Image.open(image_path).convert('RGB')
    # 获取图像张量
    if not hasattr(image_encoder, '_processor'):
        raise RuntimeError("image_encoder must have _processor attribute")

    # 调用修复后的SimplePreprocess，无images参数
    try:
        inputs = image_encoder._processor(image, return_tensors='pt')  # 位置参数，最安全
    except:
        inputs = image_encoder._processor(image=image, return_tensors='pt')  # 备选image关键字
    pixel_values = inputs['pixel_values'].to(device).to(image_encoder.dtype)

    # 提取图像特征
    with torch.no_grad():
        feat = image_encoder(pixel_values=pixel_values).image_embeds
    feat = feat.cpu().numpy().astype('float32')

    # FAISS检索
    faiss.normalize_L2(feat)
    D, I = index.search(feat, topk)
    results = [(names[int(i)], float(d)) for i, d in zip(I[0], D[0])]
    return results

def retrieve_relative(reference_image_path: str, rel_caption: str, index, names: list,
                      image_encoder, text_encoder, phi: Phi, tokenizer, topk: int = 10,
                      l2_normalize: bool = True):
    """图像+文本相对检索函数（确保无images关键字调用）"""
    # 提取参考图像的伪token
    image = Image.open(reference_image_path).convert('RGB')
    proc = getattr(image_encoder, '_processor', None)
    if proc is None:
        raise RuntimeError("image_encoder must have _processor attribute")

    # 调用修复后的SimplePreprocess，无images参数
    try:
        inputs = proc(image, return_tensors='pt')  # 位置参数，最安全
    except:
        inputs = proc(image=image, return_tensors='pt')  # 备选image关键字
    pixel_values = inputs['pixel_values'].to(device).to(image_encoder.dtype)

    with torch.no_grad():
        image_features = image_encoder(pixel_values=pixel_values).image_embeds
        if l2_normalize:
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
        pseudo = phi(image_features)

    # 构建带占位符的文本
    input_caption = f"a photo of $ that {rel_caption}"
    tokenized = tokenizer(
        text=[input_caption],
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    input_ids = tokenized['input_ids'].to(text_encoder.device)

    # 文本特征编码
    pseudo_tokens = pseudo.squeeze(0).unsqueeze(0).to(text_encoder.device).type(text_encoder.dtype)
    with torch.no_grad():
        text_feat = encode_with_pseudo_tokens_HF(text_encoder, input_ids, pseudo_tokens)
    text_feat = text_feat.detach().cpu().numpy().astype('float32')

    # FAISS检索
    faiss.normalize_L2(text_feat)
    D, I = index.search(text_feat, topk)
    results = [(names[int(i)], float(d)) for i, d in zip(I[0], D[0])]
    return results

# -------------------------- 启动函数 --------------------------
def start_server(port=8000, host='0.0.0.0'):
    """启动Flask服务器"""
    print(f"Starting Unified Image Retrieval Server on http://{host}:{port}")
    print(f"Supported Datasets: {list(DATASET_PATHS.keys())}")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Multi-Dataset Image Retrieval Server")
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    start_server(port=args.port, host=args.host)