"""Original RGB preview preprocessing in the unchanged Pillow 8.2 environment.

This JPEG is solely for frozen landmark inference. MI still reads native paired
pixels. Do not replace this with cv2 resize or direct native RGB inference.
"""
from pathlib import Path
import hashlib
import json
import PIL
from PIL import Image
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_preview(rgb_path, expected_sha256, output_dir):
    if PIL.__version__ != '8.2.0':
        raise RuntimeError('Native RGB preview requires original Pillow 8.2.0')
    rgb_path = Path(rgb_path)
    if sha256(rgb_path) != expected_sha256:
        raise ValueError('Native RGB source hash mismatch')
    with Image.open(rgb_path) as image:
        raw = np.asarray(image.convert('RGB'))
    h, w = raw.shape[:2]
    pw = min(1600, w)
    size = (pw, max(1, round(h * pw / w)))
    small = Image.fromarray(raw).resize(size, Image.LANCZOS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'rgb_preview.jpg'
    with path.open('xb') as stream:
        small.save(stream, format='JPEG', quality=92, subsampling=0)
    summary = {'status': 'complete', 'input_sha256': {'rgb': expected_sha256},
               'artifact_sha256': {'rgb_preview.jpg': sha256(path)},
               'native_shape': [h, w], 'preview_size_wh': list(size),
               'pillow_version': PIL.__version__, 'source_sha256': sha256(__file__),
               'preprocess': 'PIL RGB -> array -> PIL LANCZOS -> JPEG quality92 subsampling0; no EXIF transpose'}
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_preview(args.rgb, args.expected_sha256, args.output)), flush=True)
