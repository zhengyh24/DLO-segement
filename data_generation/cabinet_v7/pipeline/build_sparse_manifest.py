import json,hashlib,time
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path, required=True)
root = parser.parse_args().run_root.resolve()
data = root/'data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];configs=[]
for name,count in [('easy1',1),('medium2',2),('easy3',3)]:
 scene=data/name;meta=json.loads((scene/'scene.json').read_text());complete=json.loads((scene/'COMPLETE.json').read_text());assert complete['complete'] and meta['bundle_count']==count
 cameras=[]
 for i in range(4):
  v=scene/f'view_{i:03d}';camera=json.loads((v/'camera.json').read_text());ids=np.load(v/'assembly.npy');visible=np.unique(ids[ids>0]).astype(int).tolist();assert visible==camera['visible_assembly_ids'];assert camera['role']=='close_local' and camera['no_digital_crop']
  files={p.stem:p for p in sorted(v.iterdir()) if p.is_file() and p.suffix in ['.json','.png','.npy']}
  paths={k:str(p.relative_to(data)) for k,p in files.items()};paths.update({n:str((scene/n).relative_to(data)) for n in ['scene.blend','scene.json','centerline_gt.npz','topology.json','part_map.json','object_meta.json','feature_anchors.json','cabinet_revision.json','support_routes.json'] if (scene/n).is_file()})
  hashes={k:sha(p) for k,p in files.items()}
  row={'image_id':f'panel_close_v7_{name}_{v.name}','scene_id':name,'group_id':'panel_close_v5_related_design_family_20261114','split':'quality_pilot_only','physical_assembly_count':count,'visible_assembly_ids':visible,'visible_assembly_pixels':{str(j):int((ids==j).sum()) for j in visible},'camera_distance_m':camera['actual_nearest_cable_surface_distance_m'],'paths':paths,'view_file_sha256':hashes};rows.append(row);cameras.append(row['camera_distance_m'])
 configs.append({'scene_id':name,'physical_assemblies':count,'views':4,'distance_m':cameras,'scene_sha256':sha(scene/'scene.json'),'mechanical_solution':meta.get('mechanical_solution_source'),'mechanical_solution_sha256':meta.get('mechanical_solution_sha256')})
(data/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
card={'name':'Control cabinet harness close inspection v7, 1-3 harnesses','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'generated_quality_pilot_qa_recorded_separately','scene_configurations':3,'physical_assemblies_total':6,'wire_members_total':24,'native_views':12,'shared_design_families':1,'configs':configs,'partition_policy':'All 1/2/3 configurations are nested wire subsets of one cabinet design and use the same camera poses. Keep all versions of these related designs and multi-views in one split; no train/val/test split or generalization claim in this pilot.','source_reference':'User image used only to specify framing and target appearance; it is not calibrated engineering geometry.','geometry':'Member centerlines inherited from the original three-assembly mechanical solve; passive cover envelopes derived from solved cross sections; clamps tied to declared constraints. No post-solve artistic spline or visual lifting of rods.','mechanical_model':'Quasi-static extensible isotropic Kirchhoff centerline rods with axial and bending energy, gravity, endpoint/clamp constraints, relative covered-member coupling, frictionless penalty contact. No twist/shear DOFs, fabric mechanics, complete DeformX dynamics, or calibrated material validation.','material_parameters_assumed':{'EA_N':120,'EI_N_m2':2e-5,'line_mass_kg_m':0.012,'wire_radius_m':0.00135},'camera':'Fixed 50 mm lens and 36 mm horizontal sensor, 1024 square native render; synthetic intrinsics, not measured AR camera. Native 3D local views, no digital crop.','depth':'Metric camera-axis Z and separate ray range ground truth. Input noise and invalidity are illustrative and not hardware-calibrated.','background_and_hardware':'Modeled cabinet with DIN rails, terminal blocks and wire duct. Round cushioned metal clamps are auxiliary objects with assembly ownership metadata.', 'annotation':'Assembly targets include cover, exposed wire and heatshrink. Visible single wires labeled only on PVC pixels. Junctions carry assembly ownership and are excluded from branch supervision. Clips/ties/connectors/labels/equipment/distractors are separate auxiliary classes.','e3_policy':'Clean depth and full/hidden centerlines are supervision/evaluation data, not deployable model inputs. Opaque-cover internal wire identities are not claimed observable.','training_performed':False,'model_accuracy_evaluated':False,'preserve_previous_versions':True,'manifest_sha256':sha(data/'manifest.jsonl')}
(data/'dataset_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n')
(root/'receipts/manifest_generation.json').write_text(json.dumps({'rows':len(rows),'configs':configs,'source_sha256':sha(Path(__file__)),'manifest_sha256':sha(data/'manifest.jsonl'),'status':'generated_awaiting_independent_final_qa'},indent=2))
print(json.dumps({'rows':len(rows),'manifest_sha256':sha(data/'manifest.jsonl'),'root':str(data)}))
