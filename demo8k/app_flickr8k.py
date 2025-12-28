from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import argparse
import os
import glob

from retrieve_flickr8k import load_index, retrieve_by_image, retrieve_relative, build_text_encoder, translate_to_english
from utils import device

app = Flask(__name__)
INDEX_DIR = None
IMAGE_ROOT = None


# 替换为绝对路径配置（用户提供的固定路径）
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
    'split_json': '/home/yqzheng/projects/lincir/EUFCC_CIR/test/split.eufcc.json'
  }
}

# 新增：HTML模板修改（保留表单状态、显示表单信息、添加图片数量选择）
HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <title>Unified Retrieval Demo</title>
  <style>
    .result-img { width:100%; height:180px; object-fit:cover; }
    .query-img { width:224px; height:224px; object-fit:cover; }
  </style>
</head>
<body class="p-4">
<div class="container">
  <h1 class="mb-3">Unified Retrieval Demo</h1>
  <form method="post" enctype="multipart/form-data" class="row g-3 mb-4">
    <div class="col-auto">
      <select class="form-select" name="dataset">
        <option value="flickr8k" {% if dataset == 'flickr8k' %}selected{% endif %}>Flickr8k</option>
        <option value="flickr30k" {% if dataset == 'flickr30k' %}selected{% endif %}>Flickr30k</option>
        <option value="eufcc" {% if dataset == 'eufcc' %}selected{% endif %}>EUFCC_CIR</option>
      </select>
    </div>
    <div class="col-auto">
      <input class="form-control" type="file" name="query_image" accept="image/*">
    </div>
    <div class="col-auto">
      <select class="form-select" name="mode">
        <option value="image" {% if mode == 'image' %}selected{% endif %}>Image-only</option>
        <option value="relative" {% if mode == 'relative' %}selected{% endif %}>Image + caption</option>
      </select>
    </div>
    <div class="col-auto">
      <input class="form-control" type="text" name="caption" placeholder="Caption (for relative mode)" value="{{ caption|default('') }}">
    </div>
    <!-- 新增：图片数量选择框 -->
    <div class="col-auto">
      <select class="form-select" name="top_k">
        <option value="10" {% if top_k == 10 %}selected{% endif %}>Top 10</option>
        <option value="20" {% if top_k == 20 %}selected{% endif %}>Top 20</option>
        <option value="30" {% if top_k == 30 %}selected{% endif %}>Top 30</option>
        <option value="50" {% if top_k == 50 %}selected{% endif %}>Top 50</option>
      </select>
    </div>
    <div class="col-auto">
      <button class="btn btn-primary" type="submit">Search</button>
    </div>
  </form>

  <div id="content">
  {% if query %}
    <div class="row">
      <div class="col-md-3">
        <h5>Query</h5>
        <img src="{{ url_for('images', filename=query) }}" class="query-img img-thumbnail">
        <!-- 新增：显示当前表单信息 -->
        <div class="mt-3 p-2 bg-light rounded">
          <p class="mb-1"><strong>Dataset:</strong> {{ dataset }}</p>
          <p class="mb-1"><strong>Mode:</strong> {{ mode }}</p>
          {% if mode == 'relative' %}
          <p class="mb-1"><strong>Input Caption:</strong> {{ caption }}</p>
          <p class="mb-1 small text-success"><strong>Translated English:</strong> {{ translated_caption }}</p>
          {% endif %}
          <p class="mb-1"><strong>Display Top N:</strong> {{ top_k }}</p>
        </div>
      </div>
      <div class="col-md-9">
        <h5>Results (Top {{ top_k }})</h5>
        <div class="row row-cols-1 row-cols-md-3 g-3">
          {% for filename, score, stem in results %}
            <div class="col">
              <div class="card">
                <img src="{{ url_for('images', filename=filename) }}" class="card-img-top result-img">
                <div class="card-body">
                  <p class="card-text">{{ stem }}<br><strong>score:</strong> {{ '%.3f'|format(score) }}</p>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>
      </div>
    </div>
  {% endif %}
  </div>
  <hr>
  <p class="text-muted">Select dataset, upload an image and choose mode.</p>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# 新增：中文转英文函数（用户提供的实现）
def translate_to_english(text):
    """使用 DeepSeek API 将中文翻译为英文"""
    # 检查是否包含中文字符，如果没有则直接返回，节省 API 调用
    if not any('\u4e00' <= char <= '\u9fff' for char in text):
        return text

    api_key = "xx"
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


@app.route('/images/<path:filename>')
def images(filename):
  # 优化：文件查找逻辑，避免提前返回404
  uploads_root = Path(current_dir) / 'uploads'
  search_roots = [uploads_root]
  # 遍历所有数据集的图像根目录
  for info in DATASET_PATHS.values():
    if 'image_root' in info:
      search_roots.append(Path(info['image_root']))

  stem = Path(filename).stem
  # 遍历所有搜索根目录
  for root in search_roots:
    # 1. 直接查找原文件名
    file_path = root / filename
    if file_path.exists():
      return send_from_directory(str(root), filename)
    # 2. 按stem+常见后缀查找
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
      cand = root / f"{stem}{ext}"
      if cand.exists():
        return send_from_directory(str(root), cand.name)
    # 3. 按stem通配符查找
    matches = glob.glob(str(root / f"{stem}.*"))
    if matches and Path(matches[0]).exists():
      return send_from_directory(str(root), Path(matches[0]).name)
  # 所有目录都未找到，返回404
  return ('Not Found', 404)


def build_eufcc_index(split_path, image_root, image_encoder, processor, batch_size=32):
    with open(split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)
    ids = list(split.keys())
    id2path = {iid: Path(image_root) / f"{iid}.jpg" for iid in ids}
    embeddings = []
    names = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        images = []
        for iid in batch_ids:
            p = id2path[iid]
            try:
                img = Image.open(p).convert('RGB')
            except Exception as e:
                # 异常时创建默认图像，避免批次中断
                img = Image.new('RGB', (224,224), (0,0,0))
            images.append(img)
        # 设备统一，避免张量设备不匹配
        inputs = processor(images=images, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(device)  # 直接移至目标设备
        with torch.no_grad():
            feats = image_encoder(pixel_values=pixel_values).image_embeds
        feats = feats.cpu().numpy()
        # 归一化，添加防除零保护
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / (norms + 1e-12)
        embeddings.append(feats)
        names.extend(batch_ids)
    # 先判断embeddings非空再堆叠
    if not embeddings:
        return np.array([]), names
    embeddings = np.vstack(embeddings)
    return embeddings, names


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query_filename = None
    # 初始化表单参数，用于保留状态
    dataset = request.form.get('dataset', 'flickr30k')  # 默认flickr30k
    mode = request.form.get('mode', 'image')
    caption = request.form.get('caption', '').strip()
    top_k = int(request.form.get('top_k', 30))  # 新增：默认显示30张
    translated_caption = None

    if request.method == 'POST':
        file = request.files.get('query_image')
        if file and file.filename != '':  # 判断文件是否有效
            orig_filename = secure_filename(file.filename) or 'query.jpg'
            # 确保uploads目录存在
            uploads_dir = Path(current_dir) / 'uploads'
            uploads_dir.mkdir(parents=True, exist_ok=True)
            save_path = uploads_dir / orig_filename
            file.save(save_path)
            query_filename = save_path.name

            # 按数据集分支处理
            if dataset in ('flickr8k', 'flickr30k'):
                # 动态导入对应模块，确保路径可访问
                try:
                    if dataset == 'flickr8k':
                      sys.path.append(os.path.join(root_dir, 'demo8k'))
                      from retrieve_flickr8k import load_index, retrieve_by_image, retrieve_relative
                    else:
                      sys.path.append(os.path.join(root_dir, 'demo30k'))
                      from retrieve_flickr30k import load_index, retrieve_by_image, retrieve_relative
                except ImportError as e:
                    print(f"导入模块失败: {e}")
                    return render_template_string(HTML, 
                                                  results=results, 
                                                  dataset=dataset,
                                                  mode=mode,
                                                  caption=caption,
                                                  top_k=top_k,
                                                  query=query_filename, 
                                                  translated_caption=translated_caption)

                # 使用DATASET_PATHS中的绝对索引路径
                idx_path = DATASET_PATHS.get(dataset, {}).get('index_dir')
                if not idx_path or not os.path.exists(idx_path):
                    print(f"索引文件不存在: {idx_path}")
                    return render_template_string(HTML, 
                                                  results=results, 
                                                  dataset=dataset,
                                                  mode=mode,
                                                  caption=caption,
                                                  top_k=top_k,
                                                  query=query_filename, 
                                                  translated_caption=translated_caption)
                
                # 加载索引（传入索引文件所在目录，而非文件路径）
                idx_dir = os.path.dirname(idx_path)
                INDEX, NAMES = load_index(idx_dir)

                # 构建文本编码器
                args_ns = argparse.Namespace(clip_model_name='large', cache_dir='./hf_models', mixed_precision='fp16')
                try:
                    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args_ns)
                    image_encoder = image_encoder.float().to(device)
                    text_encoder = text_encoder.float().to(device)
                    image_encoder._processor = clip_preprocess
                except Exception as e:
                    print(f"构建编码器失败: {e}")
                    return render_template_string(HTML, 
                                                  results=results, 
                                                  dataset=dataset,
                                                  mode=mode,
                                                  caption=caption,
                                                  top_k=top_k,
                                                  query=query_filename, 
                                                  translated_caption=translated_caption)

                # 检索逻辑
                try:
                    if mode == 'image':
                        results_raw = retrieve_by_image(str(save_path), INDEX, NAMES, image_encoder)
                    else:
                        # 新增：自动调用翻译函数（中文转英文）
                        translated_caption = translate_to_english(caption) if caption else ""
                        if not translated_caption:
                            translated_caption = "default caption"  # 避免空文本报错

                        # 加载Phi模型
                        phi = Phi(
                            input_dim=text_encoder.config.projection_dim,
                            hidden_dim=text_encoder.config.projection_dim*4,
                            output_dim=text_encoder.config.hidden_size,
                            dropout=0.5
                        ).to(device)

                        # Phi模型加载路径
                        phi_ckpt_path = os.path.join(root_dir, 'experiment', 'checkpoints', 'phi_best.pt')
                        if not os.path.exists(phi_ckpt_path):
                            print(f"Phi模型权重不存在: {phi_ckpt_path}")
                            return render_template_string(HTML, 
                                                          results=results, 
                                                          dataset=dataset,
                                                          mode=mode,
                                                          caption=caption,
                                                          top_k=top_k,
                                                          query=query_filename, 
                                                          translated_caption=translated_caption)

                        # 加载权重
                        ckpt = torch.load(phi_ckpt_path, map_location=device)
                        phi.load_state_dict(ckpt[phi.__class__.__name__])
                        phi = phi.eval()

                        # 相对检索
                        results_raw = retrieve_relative(
                            str(save_path), translated_caption, INDEX, NAMES,
                            image_encoder, text_encoder, phi, tokenizer
                        )
                except Exception as e:
                    print(f"检索失败: {e}")
                    return render_template_string(HTML, 
                                                  results=results, 
                                                  dataset=dataset,
                                                  mode=mode,
                                                  caption=caption,
                                                  top_k=top_k,
                                                  query=query_filename, 
                                                  translated_caption=translated_caption)

                # 处理检索结果，按top_k截取
                for stem, score in results_raw[:top_k]:  # 新增：只取前top_k个结果
                    found = None
                    # 搜索路径：uploads -> 数据集图像根目录
                    dataset_image_root = DATASET_PATHS.get(dataset, {}).get('image_root')
                    search_roots = [uploads_dir]
                    if dataset_image_root:
                        search_roots.append(Path(dataset_image_root))

                    # 查找图像文件
                    for root in search_roots:
                        # 尝试常见后缀
                        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                            cand = root / f"{stem}{ext}"
                            if cand.exists():
                                found = cand.name
                                break
                        if found:
                            break
                        # 通配符查找
                        gl = glob.glob(str(root / f"{stem}.*"))
                        if gl and Path(gl[0]).exists():
                            found = Path(gl[0]).name
                            break

                    if found:
                        results.append((found, score, stem))

            elif dataset == 'eufcc':
                # EUFCC数据集处理
                eufcc_config = DATASET_PATHS.get('eufcc', {})
                eufcc_index_path = eufcc_config.get('index_dir')
                eufcc_image_root = eufcc_config.get('image_root')
                eufcc_split_json = eufcc_config.get('split_json')

                # 检查必要文件
                if not eufcc_image_root or not os.path.exists(eufcc_image_root):
                    print(f"EUFCC图像根目录不存在: {eufcc_image_root}")
                    return render_template_string(HTML, 
                                                  results=results, 
                                                  dataset=dataset,
                                                  mode=mode,
                                                  caption=caption,
                                                  top_k=top_k,
                                                  query=query_filename, 
                                                  translated_caption=translated_caption)

                idx = None
                names = None
                # 优先加载预构建的FAISS索引
                if faiss is not None and eufcc_index_path and os.path.exists(eufcc_index_path):
                    try:
                        # 加载FAISS索引
                        idx = faiss.read_index(eufcc_index_path)
                        # 加载names文件（与索引同目录）
                        names_path = os.path.join(os.path.dirname(eufcc_index_path), 'names.pkl')
                        if os.path.exists(names_path):
                            with open(names_path, 'rb') as f:
                                names = pickle.load(f)
                        else:
                            print(f"names.pkl不存在: {names_path}")
                    except Exception as e:
                        print(f"加载EUFCC索引失败: {e}")

                # 构建编码器
                args_ns = argparse.Namespace(clip_model_name='large', cache_dir='../hf_models', mixed_precision='fp16')
                try:
                    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args_ns)
                    image_encoder = image_encoder.float().to(device)
                    image_encoder._processor = clip_preprocess
                except Exception as e:
                    print(f"构建EUFCC编码器失败: {e}")
                    return render_template_string(HTML, 
                                                  results=results, 
                                                  dataset=dataset,
                                                  mode=mode,
                                                  caption=caption,
                                                  top_k=top_k,
                                                  query=query_filename, 
                                                  translated_caption=translated_caption)

                # 编码查询图像
                try:
                    img = Image.open(save_path).convert('RGB')
                    inputs = clip_preprocess(images=img, return_tensors='pt')
                    pixel_values = inputs['pixel_values'].to(device)
                    with torch.no_grad():
                        q = image_encoder(pixel_values=pixel_values).image_embeds
                    q = q.cpu().numpy().astype('float32')
                    faiss.normalize_L2(q) if faiss is not None else None
                except Exception as e:
                    print(f"编码查询图像失败: {e}")
                    return render_template_string(HTML, 
                                                  results=results, 
                                                  dataset=dataset,
                                                  mode=mode,
                                                  caption=caption,
                                                  top_k=top_k,
                                                  query=query_filename, 
                                                  translated_caption=translated_caption)

                # 检索逻辑
                if idx is not None and names is not None and len(names) > 0:
                    # 使用FAISS索引检索，按top_k截取
                    D, I = idx.search(q, top_k)  # 新增：直接搜索前top_k个结果
                    results = [(names[int(i)]+'.jpg', float(D[0][k]), names[int(i)]) for k, i in enumerate(I[0])]
                else:
                    # 动态构建索引并检索
                    if not eufcc_split_json or not os.path.exists(eufcc_split_json):
                        print(f"EUFCC split json不存在: {eufcc_split_json}")
                        return render_template_string(HTML, 
                                                      results=results, 
                                                      dataset=dataset,
                                                      mode=mode,
                                                      caption=caption,
                                                      top_k=top_k,
                                                      query=query_filename, 
                                                      translated_caption=translated_caption)

                    # 缓存索引到app实例，避免重复构建
                    if not hasattr(app, 'eufcc_embeds') or not hasattr(app, 'eufcc_names'):
                        emb, names_list = build_eufcc_index(
                            eufcc_split_json, eufcc_image_root,
                            image_encoder, clip_preprocess, batch_size=32
                        )
                        app.eufcc_embeds = emb
                        app.eufcc_names = names_list

                    # 检查缓存的嵌入是否有效
                    if app.eufcc_embeds.size == 0 or len(app.eufcc_names) == 0:
                        print("EUFCC动态索引构建失败，无有效嵌入")
                        return render_template_string(HTML, 
                                                      results=results, 
                                                      dataset=dataset,
                                                      mode=mode,
                                                      caption=caption,
                                                      top_k=top_k,
                                                      query=query_filename, 
                                                      translated_caption=translated_caption)

                    # 计算相似度并排序，按top_k截取
                    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
                    sims = app.eufcc_embeds @ q_norm.T
                    sims = sims.squeeze()
                    if len(sims) == 0:
                        results = []
                    else:
                        idxs = sims.argsort()[::-1][:top_k]  # 新增：只取前top_k个索引
                        results = [(app.eufcc_names[i]+'.jpg', float(sims[i]), app.eufcc_names[i]) for i in idxs]

    # 渲染模板时传入所有表单参数，实现状态保留
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


def start(index_dir: str, image_root: str, phi_checkpoint: str, clip_model_name: str = 'large'):
    global INDEX_DIR, IMAGE_ROOT
    INDEX_DIR = index_dir
    IMAGE_ROOT = image_root
    # 关闭调试模式，允许外部访问
    app.run(host='0.0.0.0', port=8000, debug=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', default='../experiment/flickr30k_index')
    parser.add_argument('--image-root', default='../datasets/Flickr30k/flickr30k-images')
    parser.add_argument('--phi-checkpoint', default='../experiment/checkpoints/phi_best.pt')
    parser.add_argument('--clip-model-name', default='large')
    args = parser.parse_args()
    start(args.index_dir, args.image_root, args.phi_checkpoint, args.clip_model_name)