from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import argparse
import os
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
import sys
sys.path.append(root_dir)

from retrieve_flickr30k import load_index, retrieve_by_image, retrieve_relative, build_text_encoder, translate_to_english
from utils import device

app = Flask(__name__)
INDEX_DIR = None
IMAGE_ROOT = None

HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <title>Flickr30k Retrieval</title>
  <style>
    .result-img { width:100%; height:180px; object-fit:cover; }
    .query-img { width:224px; height:224px; object-fit:cover; }
  </style>
</head>
<body class="p-4">
<div class="container">
  <h1 class="mb-3">Flickr30k Retrieval</h1>
  <form method="post" enctype="multipart/form-data" class="row g-3 mb-4">
    <div class="col-auto">
      <input class="form-control" type="file" name="query_image" accept="image/*">
    </div>
    <div class="col-auto">
      <select class="form-select" name="mode">
        <option value="image">Image-only</option>
        <option value="relative">Image + caption</option>
      </select>
    </div>
    <div class="col-auto">
      <input class="form-control" type="text" name="caption" placeholder="Caption (for relative mode)">
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
        {% if translated_caption %}
          <p class="mt-2"><strong>Mode:</strong> {{ mode }}</p>
          <p class="mt-2"><strong>Caption:</strong> {{ caption }}</p>
          <p class="mt-2 small text-success"><strong>Translated:</strong> {{ translated_caption }}</p>
        {% endif %}
      </div>
      <div class="col-md-9">
        <h5>Results</h5>
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
  <p class="text-muted">Upload an image and choose mode.</p>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''


@app.route('/images/<path:filename>')
def images(filename):
  root = Path(IMAGE_ROOT)
  file_path = root / filename
  print(f"[app] images requested: {filename}, looking in: {root.resolve()}")
  if file_path.exists():
    return send_from_directory(str(root), filename)

  stem = Path(filename).stem
  for ext in ['.jpg', '.jpeg', '.png', '.webp']:
    cand = root / f"{stem}{ext}"
    if cand.exists():
      print(f"[app] images found by extension: {cand}")
      return send_from_directory(str(root), cand.name)

  matches = glob.glob(str(root / f"{stem}.*"))
  if matches:
    print(f"[app] images found by glob: {matches[0]}")
    return send_from_directory(str(root), Path(matches[0]).name)

  print(f"[app] images NOT found for: {filename}")
  return ('Not Found', 404)



@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query_name = None
    query_filename = None
    mode = 'image'
    caption = None
    translated_caption = None
    if request.method == 'POST':
        file = request.files.get('query_image')
        mode = request.form.get('mode', 'image')
        caption = request.form.get('caption', '')
        if file:
            orig_filename = secure_filename(file.filename)
            if orig_filename == '':
                orig_filename = 'query.jpg'
            save_path = Path(IMAGE_ROOT) / orig_filename
            file.save(save_path)
            query_filename = save_path.name
            query_name = query_filename
            if mode == 'image':
                results_raw = retrieve_by_image(str(save_path), INDEX, NAMES, IMAGE_ENCODER)
            else:
                translated_caption = translate_to_english(caption)
                results_raw = retrieve_relative(str(save_path), translated_caption, INDEX, NAMES, IMAGE_ENCODER, TEXT_ENCODER, PHI, TOKENIZER)

            results = []
            for stem, score in results_raw:
                found = None
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    cand = Path(IMAGE_ROOT) / f"{stem}{ext}"
                    if cand.exists():
                        found = cand.name
                        break
                if not found:
                    gl = glob.glob(str(Path(IMAGE_ROOT) / f"{stem}.*"))
                    if gl:
                        found = Path(gl[0]).name
                if not found:
                    continue
                results.append((found, score, stem))
    return render_template_string(
      HTML, 
      results=results, 
      mode=mode,
      caption=caption if mode != 'image' else None,
      query=(query_filename) if query_filename else None,
      translated_caption=translated_caption if mode != 'image' else None
      )


def start(index_dir: str, image_root: str, phi_checkpoint: str, clip_model_name: str = 'large', cache_dir: str = './hf_models'):
    global INDEX, NAMES, IMAGE_ENCODER, TEXT_ENCODER, TOKENIZER, PHI, INDEX_DIR, IMAGE_ROOT
    INDEX_DIR = index_dir
    IMAGE_ROOT = str(Path(image_root).resolve())
    INDEX, NAMES = load_index(index_dir)

    args_ns = argparse.Namespace(clip_model_name=clip_model_name, cache_dir=cache_dir, mixed_precision='fp16')
    image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args_ns)
    image_encoder = image_encoder.float().to(device)
    text_encoder = text_encoder.float().to(device)
    image_encoder._processor = clip_preprocess

    PHI = None
    from models import Phi
    PHI = Phi(input_dim=text_encoder.config.projection_dim, hidden_dim=text_encoder.config.projection_dim * 4, output_dim=text_encoder.config.hidden_size, dropout=0.5).to(device)
    PHI.load_state_dict(__import__('torch').load(phi_checkpoint, map_location=device)[PHI.__class__.__name__])
    PHI = PHI.eval()

    global IMAGE_ENCODER, TEXT_ENCODER, TOKENIZER
    IMAGE_ENCODER = image_encoder
    TEXT_ENCODER = text_encoder
    TOKENIZER = tokenizer

    app.run(host='0.0.0.0', port=8000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--image-root', default='./datasets/Flickr30k/flickr30k-images')
    parser.add_argument('--phi-checkpoint', default='./experiment/checkpoints/phi_best.pt')
    parser.add_argument('--clip-model-name', default='large')
    parser.add_argument('--cache-dir', default='./hf_models')
    args = parser.parse_args()
    start(args.index_dir, args.image_root, args.phi_checkpoint, args.clip_model_name, args.cache_dir)
