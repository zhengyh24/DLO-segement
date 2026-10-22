"""Export an existing WireSeg membership manifest from official local Parquet shards."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from pycocotools import mask as coco_mask

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    obj=json.loads(args.manifest.read_text(encoding='utf-8'))
    groups={}
    for row in obj['records']:groups.setdefault(row['source_shard'],[]).append(row)
    args.output.mkdir(parents=True)
    for shard,rows in groups.items():
        path=args.dataset_root/'data'/shard
        table=pq.read_table(path,columns=['image','file_name','masks_rle'])
        for row in rows:
            source=table.slice(row['source_row'],1).to_pylist()[0]
            assert source['file_name']==row['file_name']
            image=Image.open(io.BytesIO(source['image']['bytes'])).convert('RGB')
            rgb=np.asarray(image)
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['rgb_sha256']
            rles=json.loads(source['masks_rle']) if isinstance(source['masks_rle'],str) else source['masks_rle']
            masks=np.stack([coco_mask.decode(rle).astype(bool) for rle in rles]) if rles else np.zeros((0,*row['shape']),dtype=bool)
            assert masks.shape==(row['num_instances'],*row['shape'])
            if row.get('decoded_masks_sha256'):assert hashlib.sha256(str(tuple(masks.shape)).encode('ascii') + b'|bool|' + np.ascontiguousarray(masks).tobytes()).hexdigest()==row['decoded_masks_sha256']
            for key in ('image','masks'):(args.output/row[key]).parent.mkdir(exist_ok=True,parents=True)
            image.save(args.output/row['image'])
            np.savez_compressed(args.output/row['masks'],masks=masks)
            for key,hashkey in (('image','image_sha256'),('masks','masks_sha256')):
                row[hashkey]=hashlib.sha256((args.output/row[key]).read_bytes()).hexdigest()
                row[key+'_bytes']=(args.output/row[key]).stat().st_size
    obj['data_root']='.'
    obj['source']='https://huggingface.co/datasets/DeformX/WireSeg-36K'
    (args.output/'manifest.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','images':len(obj['records'])}))

if __name__=='__main__':main()
