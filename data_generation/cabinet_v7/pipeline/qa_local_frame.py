"""Close-frame evidence from actual native masks, camera projection and per-object BVHs."""
import numpy as np
from mathutils import Vector
def project(p,K,Ti):
 q=np.c_[p,np.ones(len(p))]@Ti.T;uv=q[:,:3]@K.T
 return uv[:,:2]/uv[:,2,None],q[:,2]
def frame_evidence(arrays,camera,axes,topo,part_meta,meshes):
 asm=arrays['assembly'];sem=arrays['semantic'];h,w=asm.shape;K=np.array(camera['K']);T=np.array(camera['T_world_from_camera']);Ti=np.linalg.inv(T);widths=[];curves={};features=[];warnings=[]
 for t in topo:
  aid=t['assembly_id'];p=axes['assembly_%d_trunk'%aid];rad=np.broadcast_to(axes['assembly_%d_trunk_radius'%aid],(len(p),));uv,z=project(p,K,Ti);measure=[]
  for j in range(5,min(len(p)-4,50),2):
   u,v=uv[j];x,y=int(np.floor(u)),int(np.floor(v))
   if x<3 or x>=w-3 or y<3 or y>=h-3 or asm[y,x]!=aid or sem[y,x] not in [1,2,3]:continue
   tang=uv[j+1]-uv[j-1];tang/=max(np.linalg.norm(tang),1e-12);normal=np.array([-tang[1],tang[0]])
   approx=K[0,0]*2*rad[j]/z[j];extent=max(8.,2.0*approx);ds=np.arange(-extent,extent+.125,.25)
   xy=np.c_[u+normal[0]*ds,v+normal[1]*ds];xx=np.floor(xy[:,0]).astype(int);yy=np.floor(xy[:,1]).astype(int);ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
   m=np.zeros(len(ds),bool);m[ok]=(asm[yy[ok],xx[ok]]==aid)&np.isin(sem[yy[ok],xx[ok]],[1,2,3])
   mid=int(np.argmin(abs(ds)))
   if not m[mid]:continue
   lo=mid;hi=mid
   while lo>0 and m[lo-1]:lo-=1
   while hi+1<len(m) and m[hi+1]:hi+=1
   width=(hi-lo+1)*.25
   if width<.70*approx or width>1.40*approx or lo==0 or hi==len(m)-1:continue
   box=[max(0,int(u-approx*1.2)),max(0,int(v-approx*1.2)),min(w,int(u+approx*1.2)+1),min(h,int(v+approx*1.2)+1)]
   measure.append({'axis_node':j,'pixel_xy':[float(u),float(v)],'actual_contiguous_mask_width_px':float(width),'fraction_of_native_image_width':float(width/w),'nominal_projected_diameter_px':float(approx),'source_pixel_window_xyxy':box})
  area=int(np.count_nonzero(asm==aid));row={'assembly_id':aid,'visible_pixels':area,'visible_image_fraction':area/(w*h),'qualified_width_samples':len(measure),'width_samples':measure}
  if measure:row['median_actual_width_fraction']=float(np.median([x['fraction_of_native_image_width'] for x in measure]))
  widths.append(row)
  for suffix in ['trunk','left','right']:
   pp=axes['assembly_%d_%s'%(aid,suffix)];uu,zz=project(pp,K,Ti);curves[(aid,suffix)]=(uu,zz)
 # Find projected-axis crossings, then verify two actual target meshes hit the same pixel ray.
 byasm={}
 for pid,m in part_meta.items():
  if m.get('target'):byasm.setdefault(m['assembly_id'],[]).append(m['object_name'])
 crossings=[];keys=list(curves)
 for ia,ka in enumerate(keys):
  for kb in keys[ia+1:]:
   if ka[0]==kb[0]:continue
   ua,za=curves[ka];ub,zb=curves[kb];a=ua[:-1,None];b=ub[None,:-1];r=(ua[1:]-ua[:-1])[:,None];s=(ub[1:]-ub[:-1])[None,:]
   cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
   den=cross(r,s);delta=b-a;safe=np.where(abs(den)>1e-6,den,1);ta=cross(delta,s)/safe;tb=cross(delta,r)/safe
   ii,jj=np.where((abs(den)>1e-4)&(ta>=0)&(ta<=1)&(tb>=0)&(tb<=1))
   for i,j in zip(ii,jj):
    point=ua[i]+ta[i,j]*(ua[i+1]-ua[i]);x,y=np.floor(point).astype(int)
    if x<2 or x>=w-2 or y<2 or y>=h-2:continue
    ray=T[:3,:3]@np.linalg.inv(K)@np.array([x+.5,y+.5,1.]);ray/=np.linalg.norm(ray);hits=[]
    for aid in [ka[0],kb[0]]:
     rr=[]
     for name in byasm.get(aid,[]):
      q,n,face,d=meshes[name]['bvh'].ray_cast(Vector(T[:3,3]),Vector(ray),100.)
      if q is not None:rr.append(float(d))
     if rr:hits.append((min(rr),aid))
    if len(hits)!=2:continue
    hits.sort();front=hits[0][1];rear=hits[1][1]
    if asm[y,x]!=front or hits[1][0]-hits[0][0]<.0002:continue
    half=max(24,int(.09*w));box=[max(0,x-half),max(0,y-half),min(w,x+half+1),min(h,y+half+1)];patch=asm[box[1]:box[3],box[0]:box[2]];counts={str(aid):int(np.count_nonzero(patch==aid)) for aid in [front,rear]}
    if min(counts.values())<20:continue
    if any(np.linalg.norm(np.array(c['pixel_xy'])-np.array([x,y]))<10 for c in crossings):continue
    crossings.append({'pixel_xy':[int(x),int(y)],'source_pixel_window_xyxy':box,'front_assembly_id':front,'behind_assembly_id':rear,'front_target_mesh_range_m':hits[0][0],'behind_target_mesh_range_m':hits[1][0],'observed_front_pixel_assembly_id':int(asm[y,x]),'visible_pixels_in_window':counts})
    if len(crossings)>=12:break
   if len(crossings)>=12:break
  if len(crossings)>=12:break
 branch=[]
 for t in topo:
  aid=t['assembly_id'];yy,xx=np.where(arrays['junction']==aid)
  if len(xx)<5:continue
  pad=max(20,int(.065*w));box=[max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(w,int(xx.max())+pad+1),min(h,int(yy.max())+pad+1)]
  patch=arrays['branch'][box[1]:box[3],box[0]:box[2]];ids=[aid*10+k for k in [1,2,3]];counts={str(i):int(np.count_nonzero(patch==i)) for i in ids}
  if min(counts.values())>=10:branch.append({'assembly_id':aid,'source_pixel_window_xyxy':box,'observed_junction_pixels':len(xx),'observed_trunk_and_two_branch_pixels':counts,'meaning':'Visible bundle routing junction, not an electrical connection claim'})
 meaningful=[x['assembly_id'] for x in widths if x['visible_image_fraction']>=.002]
 medians=[x['median_actual_width_fraction'] for x in widths if 'median_actual_width_fraction' in x]
 reference_count=sum(.035<=x<=.09 for x in medians)
 if reference_count<min(3,len(topo)):warnings.append('Fewer than three measurable main bundles within expanded 3.5-9% framing reference; inspect widths and visual evidence')
 return {'native_image_wh':[w,h],'meaningful_visible_assembly_ids':meaningful,'meaningful_area_threshold_fraction':.002,'projected_scale_per_assembly':widths,'measured_main_bundles_near_4_8pct_reference':reference_count,'verified_crossing_windows':crossings,'verified_branch_windows':branch,'warnings':warnings,'scope':'Mask widths are sampled on visible covered trunk middles; 4-8% is visual framing reference, not a physical law. Crossing evidence uses two real target-mesh ray hits and visible neighboring pixels. Cropped objects are allowed.'}
