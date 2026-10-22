import unittest
import tempfile
from pathlib import Path
import numpy as np
from .config import Config
from .graph import build_graph, decode

class GraphTests(unittest.TestCase):
    def scene(self):
        m=np.zeros((100,180),bool);m[45:52,15:70]=True;m[45:52,82:150]=True
        return m

    def test_occlusion_link_does_not_paint_gap(self):
        m=self.scene();g=build_graph(m);scores=np.full((len(g.candidates),2),.99)
        ids,a=decode(g,scores)
        self.assertTrue(np.array_equal(ids>0,m));self.assertFalse(ids[:,70:82].any())
        self.assertEqual(ids[48,30],ids[48,120]);self.assertTrue(a['selected'])

    def test_empty_foreground(self):
        g=build_graph(np.zeros((30,40),bool));ids,_=decode(g,[])
        self.assertEqual(ids.shape,(30,40));self.assertFalse(ids.any())

    def test_reject_preserves_fragments(self):
        g=build_graph(self.scene());ids,_=decode(g,np.zeros((len(g.candidates),2)))
        self.assertNotEqual(ids[48,30],ids[48,120])

    def test_unseeded_component_preserved(self):
        m=self.scene();m[3,3]=1;g=build_graph(m);ids,_=decode(g,np.zeros((len(g.candidates),2)))
        self.assertGreater(ids[3,3],0);self.assertTrue(np.array_equal(ids>0,m))

    def test_baseline_size_rejected(self):
        g=build_graph(self.scene())
        with self.assertRaises(ValueError):decode(g,np.zeros((len(g.candidates),2)),baseline=np.ones((2,2)))

    def constraint_graph(self):
        from .graph import Graph
        partition=np.asarray([[1,1,0,2,2,0,3,3]],np.uint16)
        def edge(a,b,ae,be):return {'a':a,'b':b,'a_end':ae,'b_end':be,'key':f'{a}:{ae}-{b}:{be}','gap':5.}
        candidates=[edge(1,2,0,0),edge(2,3,1,0),edge(1,3,1,1)]
        return Graph(partition>0,partition>0,partition,partition,{},candidates,{})

    def test_negative_pair_vetoes_transitive_merge(self):
        g=self.constraint_graph();ids,a=decode(g,[[.99,.99],[.95,.95],[.05,.05]])
        self.assertEqual(ids[0,0],ids[0,3]);self.assertNotEqual(ids[0,0],ids[0,6])
        self.assertEqual(a['transitive_vetoed'],['2:1-3:0'])

    def test_ambiguous_competing_endpoint_rejected(self):
        g=self.constraint_graph();g.candidates=g.candidates[:1]+[g.candidates[2]]
        g.candidates[1]['a_end']=0
        ids,a=decode(g,[[.99,.99],[.95,.95]])
        self.assertFalse(a['selected']);self.assertTrue(all(x=='ambiguous_alternative' for x in a['rejected'].values()))

class NetworkTests(unittest.TestCase):
    def test_native_crop_and_reverse(self):
        from .features import extract_pair,reverse_candidate,reverse_features,crop_native
        m=GraphTests().scene();g=build_graph(m);rgb=np.zeros((*m.shape,3),np.uint8);rgb[48,60]=[8,9,10]
        c=g.candidates[0];s,l,v=extract_pair(rgb,g,c)
        self.assertEqual(s.shape,(7,128,128));self.assertEqual(l.shape,(7,256,256))
        rev=extract_pair(rgb,g,reverse_candidate(c));cached=reverse_features(s,l,v)
        for a,b in zip(rev,cached):np.testing.assert_allclose(a,b,atol=1e-6)
        patch=crop_native(rgb,[0,0],32);np.testing.assert_array_equal(patch[16:,16:],rgb[:16,:16])

    def test_forward_backward_without_updates(self):
        import torch
        from .model import PairNet
        from .features import extract_pair,normalize_patch
        torch.set_num_threads(2);torch.manual_seed(0)
        m=GraphTests().scene();g=build_graph(m);s,l,v=extract_pair(np.zeros((*m.shape,3),np.uint8),g,g.candidates[0])
        model=PairNet();before={k:p.detach().clone() for k,p in model.named_parameters()}
        logit=model(torch.from_numpy(normalize_patch(s))[None],torch.from_numpy(normalize_patch(l))[None],torch.from_numpy(v)[None])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.ones_like(logit));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(torch.equal(p,before[k]) for k,p in model.named_parameters()))

class DataTests(unittest.TestCase):
    def test_condition_holdout_cannot_claim_scene_independence(self):
        from .data import group_map
        from .io import write_json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'groups.json'
            obj={'grouping_kind':'condition_holdout_pilot','independent_scene_split':False,
                 'grouping_basis':'Observed family/count conditions, not simulation IDs',
                 'records':[{'file_name':'data_center/4_cable/000001.png','group_id':'condition:data_center/4_cable','split':'train'}]}
            write_json(path,obj);self.assertEqual(len(group_map(path)[1]),1)
            obj['independent_scene_split']=True;write_json(path,obj)
            with self.assertRaisesRegex(ValueError,'disclaim'):group_map(path)
            obj['independent_scene_split']=False;obj['records'][0]['group_id']='condition:data_center/8_cable';write_json(path,obj)
            with self.assertRaisesRegex(ValueError,'parent path'):group_map(path)
            obj['records'][0]['group_id']='condition:data_center/4_cable';obj.pop('grouping_kind');write_json(path,obj)
            with self.assertRaisesRegex(ValueError,'explicit'):group_map(path)

    def fixture(self,base):
        import io,json,sys
        from PIL import Image
        from .data import parquet_module,ROOT
        from .io import write_json,rgb_hash
        pq=parquet_module();import pyarrow as pa
        sys.path.insert(0,str(ROOT/'third_party/padlos_deps'));from pycocotools import mask as M
        root=base/'dataset';(root/'data').mkdir(parents=True);cache=base/'cache';cache.mkdir()
        rows=[];groups=[];predictions=[]
        for i,split in enumerate(('train','validation')):
            mask=GraphTests().scene();mask[75:81,30:145]=1
            rgb=np.zeros((*mask.shape,3),np.uint8);rgb[:]=[10+i*20,15,30];rgb[mask]=[160,130,100]
            # Synthetic fixture foreground is intentionally GT-derived ONLY in tests.
            fg=cache/f'{i}.png';Image.fromarray(mask.astype(np.uint8)*255).save(fg)
            masks=[mask & (np.indices(mask.shape)[0]<60),mask & (np.indices(mask.shape)[0]>=60)]
            rles=[]
            for m in masks:
                rle=M.encode(np.asfortranarray(m.astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii');rles.append(rle)
            stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='PNG');name=f'fixture/{i}.png'
            rows.append({'file_name':name,'image':{'bytes':stream.getvalue(),'path':None},'masks_rle':json.dumps(rles),'scene_family':'fixture'})
            groups.append({'file_name':name,'group_id':f'fixture_scene_{i}','split':split})
            predictions.append({'file_name':name,'foreground':str(fg),'rgb_sha256':rgb_hash(rgb),
                                'producer':'SYNTHETIC_TEST_FIXTURE_ONLY','gt_used':False})
        pq.write_table(pa.Table.from_pylist(rows),root/'data'/'fixture.parquet')
        group_path=base/'groups.json';pred_path=base/'predictions.json'
        write_json(group_path,{'grouping_basis':'Two deterministic unit-test scenes, not research data','records':groups})
        write_json(pred_path,{'records':predictions})
        return root,group_path,pred_path

    def test_parquet_to_supervision_and_loader(self):
        from .data import prepare,PairDataset,export_images
        from .train import validate_manifest
        from .io import read_json
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root,groups,pred=self.fixture(base)
            summary=prepare(root,groups,pred,base/'prepared')
            manifest=base/'prepared'/'manifest.json';obj=read_json(manifest)
            validate_manifest(obj,Config());self.assertGreater(summary['pairs'],0)
            ds=PairDataset(manifest,'train');s,l,g,y=ds[0]
            self.assertEqual(s.shape,(7,128,128));self.assertEqual(g.shape,(16,));self.assertTrue(float(y) in (0,1))
            self.assertEqual(export_images(root,groups,base/'exported')['images'],2)
            with self.assertRaises(ValueError):PairDataset(manifest,'test')

    def test_group_and_content_leakage_rejected(self):
        from .data import group_map,prepare
        from .io import read_json,write_json
        from .train import validate_manifest
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root,groups,pred=self.fixture(base)
            prepare(root,groups,pred,base/'prepared');obj=read_json(base/'prepared'/'manifest.json')
            obj['images'][1]['rgb_sha256']=obj['images'][0]['rgb_sha256']
            with self.assertRaisesRegex(ValueError,'leakage'):validate_manifest(obj,Config())
            mapping=read_json(groups);mapping['records'][1]['group_id']=mapping['records'][0]['group_id'];write_json(groups,mapping)
            with self.assertRaisesRegex(ValueError,'crosses'):group_map(groups)

    def test_gt_overlap_ignored(self):
        from .data import arc_labels
        g=build_graph(GraphTests().scene());assigned,stats=arc_labels(g,[g.visible,g.visible],.9)
        self.assertEqual(assigned,{});self.assertGreater(stats['overlap_gt_pixels'],0)

    def test_gt_identity_changes_labels_not_network_inputs(self):
        import json
        from .data import prepare,parquet_module
        from .io import read_json
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root,groups,pred=self.fixture(base)
            prepare(root,groups,pred,base/'before');pq=parquet_module();import pyarrow as pa
            from pycocotools import mask as M
            path=root/'data'/'fixture.parquet';rows=pq.read_table(path).to_pylist()
            for row in rows:
                masks=M.decode(json.loads(row['masks_rle']));union=masks.any(axis=2).astype(np.uint8)
                rle=M.encode(np.asfortranarray(union));rle['counts']=rle['counts'].decode('ascii');row['masks_rle']=json.dumps([rle])
            pq.write_table(pa.Table.from_pylist(rows),path)
            prepare(root,groups,pred,base/'after')
            a=read_json(base/'before'/'manifest.json')['records'];b=read_json(base/'after'/'manifest.json')['records']
            self.assertEqual(len(a),len(b));self.assertTrue(any(x['label']!=y['label'] for x,y in zip(a,b)))
            for x,y in zip(a,b):
                self.assertEqual(x['candidate'],y['candidate'])
                with np.load(base/'before'/x['path']) as u,np.load(base/'after'/y['path']) as v:
                    for key in ('small','large','geometry'):np.testing.assert_array_equal(u[key],v[key])

    def test_training_disabled_before_reading_dataset(self):
        from .train import train
        with self.assertRaisesRegex(RuntimeError,'Training disabled'):
            train('DOES_NOT_EXIST','MUST_NOT_CREATE',Config())
        with self.assertRaisesRegex(ValueError,'JSON boolean'):Config(training_enabled='false')

class PipelineTests(unittest.TestCase):
    def test_random_model_smoke_preserves_native_union(self):
        import torch
        from .model import PairNet
        from .pipeline import score_graph,unknown_mask
        torch.manual_seed(0);m=GraphTests().scene();g=build_graph(m)
        rgb=np.zeros((*m.shape,3),np.uint8)
        scores=score_graph(rgb,g,PairNet())
        self.assertEqual(scores.shape,(len(g.candidates),2));self.assertTrue(np.isfinite(scores).all())
        ids,a=decode(g,scores);np.testing.assert_array_equal(ids>0,m)
        self.assertFalse(unknown_mask(g,a)[~m].any())

    def test_random_checkpoint_rejected(self):
        import torch
        from .pipeline import load_model
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'random.pt';torch.save({'architecture':'PairNet-v1','trained':False,'optimizer_steps':0},path)
            with self.assertRaisesRegex(ValueError,'trained'):load_model(path)

    def test_bidirectional_disagreement_rejected(self):
        m=GraphTests().scene();g=build_graph(m)
        scores=np.tile([.99,.1],(len(g.candidates),1));ids,a=decode(g,scores)
        self.assertFalse(a['selected']);self.assertNotEqual(ids[48,30],ids[48,120])

def run_tests(group='all',report=False):
    cases=[GraphTests]
    if group!='graph':cases.append(NetworkTests)
    if group in ('data','all'):cases.append(DataTests)
    if group=='all':cases.append(PipelineTests)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in cases)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not report:return result.wasSuccessful()
    return {'kind':'software_self_test_only','group':group,'passed':result.wasSuccessful(),
            'tests_run':result.testsRun,'failures':[str(t) for t,_ in result.failures],
            'errors':[str(t) for t,_ in result.errors], 'training_performed':False,'optimizer_steps':0,
            'accuracy_claim':None,'dataset':'temporary synthetic fixtures; downloading dataset never opened'}
