import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def new_directory(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path

def read_rgb(path):
    # Keep file-encoded coordinates, no implicit EXIF transpose.
    with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()

def read_mask(path):
    path=Path(path)
    if path.suffix.lower()=='.npy':m=np.load(path,allow_pickle=False)
    else:
        with Image.open(path) as im:m=np.asarray(im).copy()
    if m.ndim!=2:raise ValueError(f'Expected HxW label/mask: {path}')
    if not np.isfinite(m).all():raise ValueError('Non-finite mask')
    return m

def rgb_hash(rgb):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

def resolve_relative(base,path):
    p=Path(path);return p if p.is_absolute() else Path(base)/p
