"""Fixed-budget endpoint competition ablations. No GT identity enters features.

Historical 300-image development set is reused for PQ checkpoint selection.
Hard groups use training labels for sampling only, and geometric difficulty proxies.
"""
import argparse
from collections import defaultdict, Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
import torch
import torch.nn.functional as F
from curvi_identity.config import Config
from curvi_identity.io import read_json, read_rgb, rgb_hash
from curvi_identity.graph import Graph
from curvi_identity.model import PairNet
from curvi_identity.pipeline import score_graph
from curvi_cloud_data import CloudPairDataset
from run_gtunion_2000 import validate
from predict_gtunion_2000 import threshold_ids
from evaluate_gtunion_2000 import instance_metrics, aggregate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def rank_loss(logits, labels, endpoints, margin=.5):
    """All positive-negative comparisons sharing an endpoint; no unique-exit target.

    Unknown labels (<0) and groups missing either class contribute no ranking.
    BCE and rank use the same forward pass, preserving candidate-forward budgets.
    """
    losses = []
    for ep in {x for pair in endpoints for x in pair}:
        positive = [i for i, pair in enumerate(endpoints) if ep in pair and labels[i] == 1]
        negative = [i for i, pair in enumerate(endpoints) if ep in pair and labels[i] == 0]
        if positive and negative:
            losses.append(F.relu(margin - logits[positive, None] + logits[negative][None, :]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0


class EndpointSampler:
    def __init__(self, records, manifest, rng):
        self.rng = rng
        self.ends = []
        self.groups = defaultdict(list)
        self.tags = Counter()
        graphs = {}
        colour_proxies = {}
        labels = [r['label'] for r in records]
        for i, r in enumerate(records):
            path = str(Path(manifest).parent / r['graph_item'])
            if path not in graphs:
                graph_obj = read_json(path)
                graphs[path] = graph_obj['candidates']
                # Optional fixture graphs omit RGB; production manifests have it.
                rgb = read_rgb(graph_obj['image']) if 'image' in graph_obj else None
                similar=[]
                for candidate in graphs[path]:
                    colours=[]
                    if rgb is not None:
                        for side in ('a_tip','b_tip'):
                            x,y=np.rint(candidate[side]['xy']).astype(int)
                            patch=rgb[max(0,y-2):min(rgb.shape[0],y+3),max(0,x-2):min(rgb.shape[1],x+3)]
                            colours.append(np.median(patch.reshape(-1,3),axis=0)/255.)
                    similar.append(len(colours)==2 and np.linalg.norm(colours[0]-colours[1])<.15)
                colour_proxies[path]=similar
            c = graphs[path][r['candidate_index']]
            assert c['key'] == r['candidate']
            ends = tuple((r['file_name'], c[k], c[k + '_end']) for k in ('a', 'b'))
            self.ends.append(ends)
            for ep in ends:
                self.groups[ep].append(i)
        self.keys = list(self.groups)
        # Junction competition plus long gaps and parallel tips are proxies, not
        # verified physical occlusion or same-colour categories.
        self.hard = []
        for ep, indices in self.groups.items():
            mixed = {labels[i] for i in indices} == {0, 1}
            long_gap = parallel = same_colour_negative = False
            for i in indices:
                r = records[i]
                c = graphs[str(Path(manifest).parent / r['graph_item'])][r['candidate_index']]
                long_gap |= c['gap'] >= 24
                parallel |= abs(np.dot(c['a_tip']['direction'], c['b_tip']['direction'])) >= .85
                same_colour_negative |= labels[i]==0 and colour_proxies[str(Path(manifest).parent/r['graph_item'])][r['candidate_index']]
            if mixed or long_gap or same_colour_negative or (parallel and any(labels[i] == 0 for i in indices)):
                self.hard.append(ep)
                self.tags.update(junction_mixed=int(mixed), long_gap=int(long_gap), parallel_tip=int(parallel),same_colour_negative=int(same_colour_negative))
        if not self.keys:
            raise ValueError('No supervised training groups')

    def sample(self, size, hard):
        # Half the endpoint selections remain uniform in hard-sampling arms.
        pool = self.hard if hard and self.hard and self.rng.random() < .5 else self.keys
        ep = pool[int(self.rng.integers(len(pool)))]
        group = self.groups[ep]
        indices = self.rng.choice(group, min(size, len(group)), replace=False).tolist()
        if len(indices) < size:
            indices.extend(self.rng.integers(len(self.ends), size=size-len(indices)).tolist())
        return indices


def whole_image_eval(model, cfg, base, output, device):
    """Inference uses only original frozen predicted graphs and RGB; GT follows."""
    output.mkdir(parents=True, exist_ok=False)
    images = [r for r in read_json(base/'groups.json')['records'] if r['split']=='validation']
    inputs = {r['file_name']:r for r in read_json(base/'inference_inputs.json')['records']}
    predictions = []
    model.eval()
    for r in images:
        inp = inputs[r['file_name']]
        graph_dir = base/'prepared/graphs'/inp['name']
        obj = read_json(graph_dir/'graph.json')
        rgb = read_rgb(inp['image'])
        if rgb_hash(rgb) != inp['rgb_sha256']:
            raise ValueError('Development image hash changed')
        with np.load(graph_dir/'arrays.npz', allow_pickle=False) as z:
            partition, skeleton = z['partition'], z['skeleton'].astype(bool)
        graph = Graph(partition>0, skeleton, None, partition, {}, obj['candidates'], obj['audit'])
        scores = score_graph(rgb, graph, model, cfg, device)
        ids = threshold_ids(graph, scores, .8)
        if np.any((ids>0)!=(partition>0)):
            raise ValueError('Foreground was changed')
        np.savez_compressed(output/(inp['name']+'.npz'), ids=ids, scores=scores)
        selected = {c[k] for c,s in zip(graph.candidates,scores) if min(s)>=.8 for k in ('a','b')}
        predictions.append(dict(file_name=r['file_name'], name=inp['name'],
            unknown_pixels=int(((partition>0)&~np.isin(partition,list(selected))).sum()),
            foreground_pixels=int((partition>0).sum())))
    save(output/'prediction_complete.json', dict(images=len(predictions), gt_identity_used=False,
        threshold=.8, foreground='frozen_predicted_foreground', records=predictions))
    # Only after all predictions have been saved do we load supervision.
    from curvi_identity.data import decode_masks
    labels = {r['file_name']:r for r in read_json(base/'selected_labels.json')['records']}
    rows = []
    for r in predictions:
        with np.load(output/(r['name']+'.npz')) as z:
            ids=z['ids']
        gt=decode_masks(labels[r['file_name']]['masks_rle'], ids.shape)
        rows.append(dict(file_name=r['file_name'], **instance_metrics(ids,gt)))
    metrics=aggregate(rows)
    metrics['unknown_visible_fraction']=sum(r['unknown_pixels'] for r in predictions)/max(1,sum(r['foreground_pixels'] for r in predictions))
    metrics['unknown_definition']='Visible partitions without accepted link; not calibrated uncertainty'
    save(output/'metrics.json',dict(metrics=metrics,rows=rows,independent_test=False))
    return metrics


def main(args):
    manifest=args.manifest.resolve();out=args.output.resolve();obj=read_json(manifest)
    cfg=replace(Config(**obj['config']),seed=args.seed,batch_size=args.batch_size,threshold=.8)
    validate(obj,cfg)
    if not cfg.training_enabled: raise ValueError('Training disabled')
    torch.set_num_threads(4);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    ds=CloudPairDataset(manifest,'train',args.seed)
    rng=np.random.default_rng(args.seed)
    sampler=EndpointSampler(ds.records,manifest,rng)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    source_paths=[Path(__file__)]+list((ROOT/'curvi_identity').glob('*.py'))+[ROOT/'tools'/x for x in
        ('curvi_cloud_data.py','run_gtunion_2000.py','predict_gtunion_2000.py','evaluate_gtunion_2000.py')]
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in source_paths}
    # A manifest hash alone cannot detect changed graph arrays at the same paths.
    graph_files={manifest.parent/r['graph_item'] for r in obj['records']}
    dev_names={r['file_name'] for r in read_json(args.base/'groups.json')['records'] if r['split']=='validation'}
    for r in read_json(args.base/'inference_inputs.json')['records']:
        if r['file_name'] in dev_names:graph_files.add(args.base/'prepared/graphs'/r['name']/'graph.json')
    graph_hashes={str(p):[digest(p),digest(p.parent/'arrays.npz')] for p in sorted(graph_files)}
    provenance=dict(arguments=config,manifest_sha256=digest(manifest),sources=hashes,
        graph_files_sha256=hashlib.sha256(json.dumps(graph_hashes,sort_keys=True).encode()).hexdigest(),
        development_groups_sha256=digest(args.base/'groups.json'),
        development_inputs_sha256=digest(args.base/'inference_inputs.json'),
        development_labels_sha256=digest(args.base/'selected_labels.json'))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'provenance.json').exists():
        if read_json(out/'provenance.json')!=provenance:raise ValueError('Resume provenance mismatch')
        if not (out/'last.pt').exists():raise ValueError('Existing run lacks resumable checkpoint')
    else:
        if list(out.iterdir()):raise ValueError('Refuse populated output directory')
        save(out/'provenance.json',provenance)
        save(out/'graph_hashes.json',graph_hashes)
        for p in source_paths:
            dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    save(out/'sampling.json',dict(training_pairs=len(ds),label_counts=dict(Counter(str(r['label']) for r in ds.records)),
        supervised_groups=len(sampler.keys),hard_groups=len(sampler.hard),hard_proxy_counts=dict(sampler.tags),
        input_images=len([r for r in obj['images'] if r['split']=='train']),
        excluded_unknown_pairs='Not present in frozen supervised manifest; no fabricated negative labels',
        hard_sampling_fraction=.5,ordinary_endpoint_fraction=.5,
        limitations='RGB tip colour and geometric proxies do not certify same-colour crossing, parallel adhesion or actual occlusion. All modes group by endpoint.'))
    save(out/'training_supervision_audit.json',[dict(file_name=r['file_name'],supervision=r.get('supervision'),pairs=r['pairs'])
        for r in obj['images'] if r['split']=='train'])
    model=PairNet().to(args.device);optim=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    start=0;best=-1.;history=[]
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if ck['provenance']!=provenance:raise ValueError('Checkpoint provenance mismatch')
        model.load_state_dict(ck['model']);optim.load_state_dict(ck['optimizer'])
        start=ck['optimizer_steps'];best=ck['best_pq'];history=ck['history']
        rng.bit_generator.state=ck['sampling_rng'];ds.rng.bit_generator.state=ck['augment_rng']
        torch.set_rng_state(ck['torch_rng'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
    began=time.monotonic();loss_sum=rank_sum=0.;window=0
    for step in range(start+1,args.steps+1):
        model.train();idx=sampler.sample(args.batch_size,args.mode in ('hard','both'))
        batch=[ds[i] for i in idx];s,l,g,y=[torch.stack(x).to(args.device) for x in zip(*batch)]
        optim.zero_grad(set_to_none=True);logits=model(s,l,g)
        bce=F.binary_cross_entropy_with_logits(logits,y)
        rank=rank_loss(logits,[ds.records[i]['label'] for i in idx],[sampler.ends[i] for i in idx],args.margin)
        loss=bce+(args.rank_weight*rank if args.mode in ('competition','both') else 0.)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optim.step();loss_sum+=float(bce.detach());rank_sum+=float(rank.detach());window+=1
        if step%100==0:save(out/'progress.json',dict(step=step,total=args.steps,bce=loss_sum/window,ranking=rank_sum/window,seconds=time.monotonic()-began))
        if step%args.eval_interval==0 or step==args.steps:
            dest=out/f'eval_{step:07d}'
            # Resume interrupted evaluation without overwriting partial evidence.
            if dest.exists():dest=out/f'eval_{step:07d}_retry_{time.time_ns()}'
            metrics=whole_image_eval(model,cfg,args.base,dest,args.device)
            row=dict(step=step,train_bce=loss_sum/window,rank=rank_sum/window,metrics=metrics)
            history.append(row);improved=metrics['pq_style']>best;best=max(best,metrics['pq_style'])
            common=dict(schema=1,architecture='PairNet-v1',trained=True,model=model.state_dict(),config=cfg.to_dict(),
                epoch=step,epoch_semantics='Optimizer step, not a dataset pass',optimizer_steps=step,
                candidate_forward_budget=step*args.batch_size,selection='maximum actual-input development PQ; fixed threshold 0.8',
                validation=metrics,manifest_sha256=digest(manifest),provenance=provenance)
            if improved:checkpoint(out/'best.pt',common)
            checkpoint(out/'last.pt',dict(common,optimizer=optim.state_dict(),best_pq=best,history=history,
                sampling_rng=rng.bit_generator.state,augment_rng=ds.rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else []))
            save(out/'history.json',history);print(row,flush=True);loss_sum=rank_sum=0.;window=0
    save(out/'completion.json',dict(steps=args.steps,candidate_forward_budget=args.steps*args.batch_size,best_development_pq=best,
        converged=False,independent_test=False,scope='Fixed-budget 2000-image ablation on historical development; not full convergence'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,default=ROOT/'run');p.add_argument('--mode',choices=['bce','hard','competition','both'],required=True)
    p.add_argument('--seed',type=int,default=20261013);p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--eval-interval',type=int,default=5000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--margin',type=float,default=.5);p.add_argument('--rank-weight',type=float,default=.2);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if min(a.steps,a.eval_interval,a.batch_size)<=0:p.error('Budgets must be positive')
    # Lock file sits beside output so interrupted/duplicate launches cannot share it.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if os.name=='posix':
        import fcntl
        with a.output.with_name(a.output.name+'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main(a)
    else:
        main(a)
