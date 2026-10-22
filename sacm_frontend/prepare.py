"""Convert frozen local/remote curvi RGB+RLE export without reshuffling split."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from pycocotools import mask as coco
from sacm_frontend.data import read_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-split',type=int,default=0,help='0=all, otherwise deterministic software/pilot subset')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    groups=json.loads((a.source/'groups.json').read_text(encoding='utf-8'))
    inputs={r['file_name']:r for r in json.loads((a.source/'inference_inputs.json').read_text(encoding='utf-8'))['records']}
    labels={r['file_name']:r for r in json.loads(a.labels.read_text(encoding='utf-8'))['records']}
    (a.out/'rgb').mkdir();(a.out/'masks').mkdir();rows=[];counts={};hashes={}
    for g in groups['records']:
        split=g['split']
        if a.per_split and counts.get(split,0)>=a.per_split:continue
        # Source may contain stale Windows paths after upload: resolve by exported basename.
        inp=inputs[g['file_name']];src=a.source/'rgb'/Path(inp['image'].replace('\\','/')).name
        with Image.open(src) as im:
            if im.getexif().get(274,1)!=1:raise ValueError('EXIF requires explicit coordinate mapping')
            rgb=np.array(im.convert('RGB'))
        digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in hashes:raise ValueError('Duplicate selected RGB')
        hashes[digest]=g['file_name'];h,w=rgb.shape[:2]
        value=labels[g['file_name']]['masks_rle'];rles=json.loads(value) if isinstance(value,str) else value
        masks=[]
        for item in rles:
            item=dict(item)
            if isinstance(item['counts'],list):item=coco.frPyObjects(item,*item['size'])
            mask=coco.decode(item).astype(bool)
            if mask.shape!=(h,w):raise ValueError('RLE/native image coordinates differ')
            masks.append(mask)
        ident=f'{len(rows):06d}';image=f'rgb/{ident}.png';target=f'masks/{ident}.npz'
        # Exported PNG already contains the native pixels. Preserve its bytes instead
        # of spending CPU re-encoding thousands of unchanged full-resolution images.
        if src.suffix.lower()=='.png':shutil.copyfile(src,a.out/image)
        else:Image.fromarray(rgb).save(a.out/image)
        np.savez_compressed(a.out/target,masks=np.stack(masks) if masks else np.zeros((0,h,w),bool))
        rows.append(dict(id=ident,image=image,masks=target,split=split,group_id=g['group_id'],
            file_name=g['file_name'],rgb_sha256=digest,shape=[h,w]))
        counts[split]=counts.get(split,0)+1
    obj={k:groups[k] for k in ('grouping_kind','independent_scene_split','grouping_basis')}
    obj.update(schema=1,records=rows,counts=counts,source=str(a.source),subset_limit=a.per_split,
        label_contract='visible RLE instance masks; overlap retained; synthetic EXIF=1 only')
    manifest=a.out/'manifest.json';manifest.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    read_manifest(manifest,True)
    print(json.dumps({'prepared':counts,'manifest':str(manifest),'independent_scene_split':obj['independent_scene_split']}))
if __name__=='__main__':main()
