#!/usr/bin/env python3
"""Static extensible isotropic Kirchhoff centerline rods.
Bending + axial strain + gravity + frictionless capsule/plane penalty contact.
No twist/shear dynamics, no world-space guide penalty, no artistic post-smoothing.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import json, hashlib, time, argparse, itertools
import numpy as np
import torch
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial import cKDTree
from scipy.optimize import minimize
torch.set_num_threads(1)
DT=torch.float64

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,a):Path(p).write_text(json.dumps(a,indent=2)+"\n")
def T(x):return torch.as_tensor(x,dtype=DT)
class Problem:
 def __init__(self,contract,assembly_ids=None,gravity_scale=1.,rest_ratio=1.01):
  self.contract=contract
  self.wires=[w for w in contract["wires"] if assembly_ids is None or w["assembly_id"] in assembly_ids]
  self.ids=[w["wire_id"] for w in self.wires];self.offset={};self.slices={};base=[];fixed={};sega=[];segb=[];r=[];owners=[];local=[]
  for wi,w in enumerate(self.wires):
   pts=np.array(w["initial_positions_world_m"],dtype=float);off=len(base);self.offset[w["wire_id"]]=off
   self.slices[w["wire_id"]]=slice(off,off+len(pts));base.extend(pts)
   for k,v in w["fixed_node_positions_world_m"].items():fixed[off+int(k)]=v
   for j in range(len(pts)-1):sega.append(off+j);segb.append(off+j+1);r.append(w["radius_m"]);owners.append(wi);local.append(j)
  self.base=np.array(base);self.fixed=np.array(sorted(fixed),dtype=int);self.base[self.fixed]=np.array([fixed[x] for x in self.fixed])
  self.free=np.array([i for i in range(len(base)) if i not in fixed],dtype=int)
  self.sa=np.array(sega);self.sb=np.array(segb);self.radius=np.array(r);self.owner=np.array(owners);self.local=np.array(local)
  self.l0=np.linalg.norm(self.base[self.sb]-self.base[self.sa],axis=1)*rest_ratio
  both=np.isin(self.sa,self.fixed)&np.isin(self.sb,self.fixed);self.l0[both]/=rest_ratio
  self.rest_ratio=rest_ratio
  # Effective uncalibrated independent axial/bending stiffness and line mass.
  self.EA=np.full(len(self.sa),120.0);self.EI=np.full(len(self.sa),2.e-5);self.mu=np.full(len(self.sa),.012)
  self.mass=np.zeros(len(base));np.add.at(self.mass,self.sa,.5*self.mu*self.l0);np.add.at(self.mass,self.sb,.5*self.mu*self.l0)
  self.g=np.array(contract["gravity_world_m_s2"],float)*gravity_scale;self.gravity_scale=gravity_scale
  self.b0=np.where(self.owner[:-1]==self.owner[1:])[0];self.b1=self.b0+1
  self.lb=(self.l0[self.b0]+self.l0[self.b1])*.5
  cp=[];cd=[]
  for c in contract.get("cross_section_couplings",[]):
   if not all(m["wire_id"] in self.ids for m in c["members"]):continue
   for a,b in itertools.combinations(c["members"],2):
    i=self.offset[a["wire_id"]]+a["node_index"];j=self.offset[b["wire_id"]]+b["node_index"]
    cp.append((i,j));cd.append(np.linalg.norm(self.base[i]-self.base[j]))
  self.cp=np.array(cp,dtype=int).reshape(-1,2);self.cd=np.array(cd);self.kpack=2000.;self.kcontact=5000.;self.contact_margin=2.e-5
  self.normals=[];self.offsets=[]
  for o in contract.get("obstacles",[]):
   if o["type"]!="halfspace":raise ValueError("Unsupported obstacle type: "+o["type"])
   n=np.array(o["normal"],float);assert abs(np.linalg.norm(n)-1)<1e-6
   self.normals.append(n);self.offsets.append(o["surface_offset_m"])
  self.node_radius=np.zeros(len(base))
  for w in self.wires:self.node_radius[self.slices[w["wire_id"]]]=w["radius_m"]
  K=np.eye(len(base))*1e-4
  for i,j,k in zip(self.sa,self.sb,self.EA/self.l0):
   K[i,i]+=k;K[j,j]+=k;K[i,j]-=k;K[j,i]-=k
  for a,b,l,ei in zip(self.b0,self.b1,self.lb,self.EI[self.b0]):
   ids=[self.sa[a],self.sb[a],self.sb[b]];v=np.array([1.,-2.,1.]);K[np.ix_(ids,ids)]+=ei/l**3*np.outer(v,v)
  for i,j in self.cp:
   K[i,i]+=self.kpack;K[j,j]+=self.kpack;K[i,j]-=self.kpack;K[j,i]-=self.kpack
  self.L=cholesky(K[np.ix_(self.free,self.free)],lower=True)
  self.Escale=max(float(sum(self.mass)*9.81*.1),1e-4);self.sqrtE=np.sqrt(self.Escale)
  self.reference_free=self.base[self.free].copy();self.last={}
 def from_y(self,y):
  return self.reference_free+solve_triangular(self.L.T,y.reshape(-1,3)*self.sqrtE,lower=False,check_finite=False)
 def candidate_pairs(self,p):
  mid=(p[self.sa]+p[self.sb])*.5;length=np.linalg.norm(p[self.sb]-p[self.sa],axis=1)
  bound=2*max(self.radius)+max(length)+self.contact_margin+1e-7
  pairs=cKDTree(mid).query_pairs(bound,output_type="ndarray")
  if len(pairs)==0:return pairs.reshape(-1,2)
  i,j=pairs[:,0],pairs[:,1]
  keep=~((self.owner[i]==self.owner[j])&(np.abs(self.local[i]-self.local[j])<=3))
  return pairs[keep]
 def terms(self,p,pairs):
  d=p[self.sb]-p[self.sa];length=torch.linalg.vector_norm(d,dim=1);t=d/length[:,None]
  axial=(.5*T(self.EA/self.l0)*(length-T(self.l0))**2).sum()
  bending=(.5*T(self.EI[self.b0]/self.lb)*((t[self.b1]-t[self.b0])**2).sum(1)).sum()
  gravity=-(T(self.mass)[:,None]*p*T(self.g)[None,:]).sum()
  packing=(.5*self.kpack*(torch.linalg.vector_norm(p[self.cp[:,0]]-p[self.cp[:,1]],dim=1)-T(self.cd))**2).sum() if len(self.cp) else p.sum()*0
  board=p.sum()*0
  for normal,offset in zip(self.normals,self.offsets):
   pen=torch.relu(T(self.node_radius)+offset-(p*T(normal)).sum(1))
   board=board+.5*self.kcontact*(pen**2).sum()
  contact=p.sum()*0;clearance=torch.empty(0,dtype=DT)
  if len(pairs):
   i,j=pairs[:,0],pairs[:,1];a=p[self.sa[i]];b=p[self.sb[i]];c=p[self.sa[j]];e=p[self.sb[j]]
   u=b-a;v=e-c;w=a-c
   aa=(u*u).sum(1);bb=(u*v).sum(1);cc=(v*v).sum(1);dd=(u*w).sum(1);ee=(v*w).sum(1)
   def pd(pt,origin,direction,norm):
    tau=torch.clamp(((pt-origin)*direction).sum(1)/norm,0,1)
    return ((pt-origin-tau[:,None]*direction)**2).sum(1)
   values=[pd(a,c,v,cc),pd(b,c,v,cc),pd(c,a,u,aa),pd(e,a,u,aa)]
   denom=aa*cc-bb*bb;safe=torch.clamp(denom,min=1e-25)
   s=(bb*ee-cc*dd)/safe;tpar=(aa*ee-bb*dd)/safe
   interior=((w+s[:,None]*u-tpar[:,None]*v)**2).sum(1)
   valid=(denom>1e-22)&(s>=0)&(s<=1)&(tpar>=0)&(tpar<=1)
   values.append(torch.where(valid,interior,torch.full_like(interior,1.e6)))
   distance=torch.sqrt(torch.clamp(torch.stack(values).min(0).values,min=1e-24))
   clearance=distance-T(self.radius[i]+self.radius[j])
   contact=.5*self.kcontact*(torch.relu(self.contact_margin-clearance)**2).sum()
  return {"axial_J":axial,"bending_J":bending,"gravity_J":gravity,"packing_J":packing,"board_contact_J":board,"rod_contact_J":contact},clearance
 def energy_grad_physical(self,free,full_gradient=False):
  p_np=self.base.copy();p_np[self.free]=free;pair=self.candidate_pairs(p_np)
  if full_gradient:
   p=T(p_np).requires_grad_(True);terms,clear=self.terms(p,pair);energy=sum(terms.values());grad=torch.autograd.grad(energy,p)[0]
  else:
   f=T(free).requires_grad_(True);p=T(self.base).index_copy(0,torch.as_tensor(self.free),f)
   terms,clear=self.terms(p,pair);energy=sum(terms.values());grad=torch.autograd.grad(energy,f)[0]
  vals={k:float(v.detach()) for k,v in terms.items()}
  vals["total_J"]=float(energy.detach());vals["min_surface_clearance_m"]=float(clear.min().detach()) if len(clear) else None
  self.last=vals
  return float(energy.detach()),grad.detach().numpy(),p_np
 def objective(self,y):
  free=self.from_y(y);energy,grad,p=self.energy_grad_physical(free)
  gy=solve_triangular(self.L,grad,lower=True,check_finite=False)/self.sqrtE
  return energy/self.Escale,gy.ravel()
 def diagnostics(self,y):
  energy,g,p=self.energy_grad_physical(self.from_y(y),True);freeg=g[self.free]
  l=np.linalg.norm(p[self.sb]-p[self.sa],axis=1);strain=l/self.l0-1;t=(p[self.sb]-p[self.sa])/l[:,None]
  angle=np.arccos(np.clip((t[self.b0]*t[self.b1]).sum(1),-1,1));curv=2*np.sin(angle/2)/((l[self.b0]+l[self.b1])*.5)
  force_scale=max(float(self.mass.sum()*9.81),1e-6)
  rec=dict(self.last)
  rec.update({"max_free_force_N":float(np.linalg.norm(freeg,axis=1).max()),"rms_free_force_N":float(np.sqrt(np.mean(freeg**2))),"force_scale_N":force_scale,"normalized_max_free_force":float(np.linalg.norm(freeg,axis=1).max()/force_scale),"fixed_node_error_m":float(np.linalg.norm(p[self.fixed]-self.base[self.fixed],axis=1).max()),"max_abs_segment_strain":float(abs(strain).max()),"max_curvature_m_inv":float(curv.max()),"max_bending_strain_radius_times_curvature":float((curv*self.radius[self.b0]).max()),"board_max_penetration_m":max([float(np.maximum(self.node_radius+o-p@n,0).max()) for n,o in zip(self.normals,self.offsets)]+[0.]),"coupling_max_distance_error_m":float(abs(np.linalg.norm(p[self.cp[:,0]]-p[self.cp[:,1]],axis=1)-self.cd).max()) if len(self.cp) else 0.,"free_scalar_dofs":int(len(self.free)*3),"total_nodes":len(p),"contact_scope":"all capsule segment pairs dynamically broad-phased, including nonlocal same-rod segments; own segment neighbours within 3 segments excluded","constraint_reaction_N":{str(int(i)):g[i].tolist() for i in self.fixed},"wire_metrics":[]})
  for wi,w in enumerate(self.wires):
   mask=self.owner==wi;bi=self.owner[self.b0]==wi;sl=self.slices[w["wire_id"]]
   fi=np.array([i for i in range(sl.start,sl.stop) if i in set(self.free)],dtype=int);rodforce=np.linalg.norm(g[fi],axis=1)
   rodscale=max(float(self.mass[sl].sum()*9.81),float(self.EI[mask].mean()/self.l0[mask].sum()**2),1e-6)
   rec["wire_metrics"].append({"wire_id":w["wire_id"],"rest_length_m":float(self.l0[mask].sum()),"initial_length_m":float(np.linalg.norm(np.diff(self.base[sl],axis=0),axis=1).sum()),"solved_length_m":float(l[mask].sum()),"max_abs_segment_strain":float(abs(strain[mask]).max()),"max_curvature_m_inv":float(curv[bi].max()),"max_displacement_from_initial_m":float(np.linalg.norm(p[sl]-self.base[sl],axis=1).max()),"fixed_node_indices":w["fixed_node_indices"],"max_free_force_N":float(rodforce.max()),"rms_free_force_N":float(np.sqrt(np.mean(g[fi]**2))),"force_scale_N":rodscale,"normalized_max_free_force":float(rodforce.max()/rodscale),"normalization":"max(rod mass*reference 9.81 m/s2, EI/Lrod², 1e-6 N); independent of number of other rods","min_bending_radius_m":float(1/curv[bi].max()),"min_bending_radius_over_wire_diameter":float(1/(curv[bi].max()*2*w["radius_m"]))})
  rec["normalized_max_free_force"]=max(w["normalized_max_free_force"] for w in rec["wire_metrics"])
  rec["force_normalization"]="maximum per-rod normalized residual; scene-size independent"
  rec["constraint_reaction_convention"]="support reaction equals potential gradient on fixed nodes; unconstrained physical force is its negative"
  return rec,p
def solve(args):
 start=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 source_bytes=Path(__file__).read_bytes();source_sha=hashlib.sha256(source_bytes).hexdigest();(out/"solver_source.py").write_bytes(source_bytes)
 input_bytes=Path(args.input).read_bytes();input_sha=hashlib.sha256(input_bytes).hexdigest();contract=json.loads(input_bytes);model=Problem(contract,args.assemblies,args.gravity_scale,args.rest_ratio)
 (out/"input_contract_frozen.json").write_bytes(input_bytes)
 config={"model":"static extensible isotropic Kirchhoff centerline rod energy; no torsion/shear DOFs","source":str(Path(__file__).resolve()),"source_sha256":source_sha,"input_sha256":input_sha,"gravity_world_m_s2":model.g.tolist(),"EA_N":120.0,"EI_N_m2":2.e-5,"linear_mass_kg_m":.012,"wire_radii_m":[w["radius_m"] for w in model.wires],"material_assumptions":"effective independent EA and EI; not measured or calibrated; rest curvature zero","rest_length_treatment":"initial segment lengths times stated ratio, except segments with both nodes fixed use initial length","rest_ratio":args.rest_ratio,"contact_penalty_N_m":model.kcontact,"rod_contact_total_clearance_margin_m":model.contact_margin,"cross_section_distance_penalty_N_m":model.kpack,"packing_scope":"pair distances within only declared covered cross sections; rotational and translational world motion free","world_guide_penalty":False,"boundary_method":"fixed node variables eliminated exactly; fixed 3-node end ranges impose outlet tangent","optimizer":"scipy L-BFGS-B on stiffness-preconditioned free position DOFs","seed":42,"assembly_ids":args.assemblies,"max_iterations":args.maxiter,"raw_output_postprocessing":"none; all original material nodes exported directly","equation":"U=0.5 sum EA/l0*(length-l0)^2 +0.5 sum EI/lbar*|t_next-t_prev|^2 -sum mass*g dot x +0.5 k_c sum relu(contact_margin-clearance)^2 +0.5 k_p sum (member_distance-rest_distance)^2","contact_friction":False}
 dump(out/"solver_config.json",config)
 y=np.zeros(len(model.free)*3)
 if args.warmstart:
  z=np.load(args.warmstart);p=model.base.copy()
  for w in model.wires:p[model.slices[w["wire_id"]]]=z["wire_"+str(w["wire_id"])]
  y=(model.L.T@(p[model.free]-model.reference_free)/model.sqrtE).ravel()
 init,pinit=model.diagnostics(y);dump(out/"initial_diagnostics.json",init)
 history=out/"iteration_history.jsonl";iteration=0
 def callback(yi):
  nonlocal iteration
  iteration+=1
  if iteration%25==0 or iteration==1:
   d,p=model.diagnostics(yi);d={k:v for k,v in d.items() if k not in ["constraint_reaction_N","wire_metrics"]}
   d.update(iteration=iteration,elapsed_s=time.time()-start)
   with history.open("a") as f:f.write(json.dumps(d)+"\n")
   if iteration%100==0:print(json.dumps({k:d[k] for k in ["iteration","total_J","normalized_max_free_force","min_surface_clearance_m","elapsed_s"]}),flush=True)
 result=minimize(model.objective,y,jac=True,method="L-BFGS-B",callback=callback,options={"maxiter":args.maxiter,"ftol":1e-15,"gtol":1e-10,"maxcor":30,"maxls":40})
 final,p=model.diagnostics(result.x)
 arrays={};_,fullg,_=model.energy_grad_physical(model.from_y(result.x),True)
 for w in model.wires:
  wid=w["wire_id"];sl=model.slices[wid];arrays["wire_"+str(wid)]=p[sl];arrays["initial_wire_"+str(wid)]=model.base[sl];arrays["potential_gradient_wire_"+str(wid)]=fullg[sl];arrays["node_radius_wire_"+str(wid)]=model.node_radius[sl];arrays["rest_lengths_"+str(wid)]=model.l0[model.owner==model.ids.index(wid)]
 np.savez_compressed(out/"solved_rods.npz",**arrays)
 maxpen=max(0.,-(final["min_surface_clearance_m"] or 0.))
 finite=bool(np.isfinite(p).all());forceok=final["normalized_max_free_force"]<.005 and final["max_free_force_N"]<2e-5
 shapeok=final["max_abs_segment_strain"]<.03 and min(w["min_bending_radius_m"] for w in final["wire_metrics"])>=.020
 final["engineering_screen"]={"max_abs_axial_strain_limit":.03,"minimum_bending_radius_m":.020,"source":"author-selected screening assumptions, not manufacturer material certification","shape_gate_passed":bool(shapeok)}
 final.update(status="passed_static_equilibrium_and_shape_gate_with_scope" if finite and forceok and shapeok and maxpen<1e-7 and final["board_max_penetration_m"]<5e-5 and final["fixed_node_error_m"]<1e-10 else "not_accepted",optimizer_success=bool(result.success),optimizer_message=str(result.message),iterations=int(result.nit),function_evaluations=int(result.nfev),elapsed_s=time.time()-start,source_sha256=source_sha,contract_sha256=input_sha,output_sha256=sha(out/"solved_rods.npz"),initial_total_J=init["total_J"],output_node_count_unchanged=True,mechanical_model=config["model"],limits=["uncalibrated assumed effective parameters","no torsion/shear dynamics or viscoelasticity","finite-resolution centerline rods, frictionless penalty contact","cross-section packing is constrained only at authored covered material sections","fixed clamps and endpoints must be rendered as matching support hardware","not validated manufacture, electrical topology or real-world deformation"])
 dump(out/"result.json",final)
 print(json.dumps({"status":final["status"],"iterations":final["iterations"],"elapsed_s":final["elapsed_s"],"normalized_force":final["normalized_max_free_force"],"output":str(out)}),flush=True)
 return 0 if final["status"].startswith("passed") else 2
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--input",required=True);a.add_argument("--output",required=True);a.add_argument("--assemblies",type=int,nargs="+");a.add_argument("--gravity-scale",type=float,default=1.);a.add_argument("--rest-ratio",type=float,default=1.01);a.add_argument("--maxiter",type=int,default=3000);a.add_argument("--warmstart");raise SystemExit(solve(a.parse_args()))
