"""Independent view geometry checks; no renderer import. Runs inside Blender on CPU."""
import numpy as np
from mathutils import Vector

def stable_pixels(ids):
    m=np.ones(ids.shape,bool)
    m[[0,-1],:]=False;m[:,[0,-1]]=False
    m[1:,:]&=ids[1:,:]==ids[:-1,:];m[:-1,:]&=ids[:-1,:]==ids[1:,:]
    m[:,1:]&=ids[:,1:]==ids[:,:-1];m[:,:-1]&=ids[:,:-1]==ids[:,1:]
    return m

def stratified_samples(part,assembly,limit=900):
    stable=stable_pixels(part); pairs=[]; categories={}
    # Sampling by visible object IDs prevents large background panels hiding thin targets.
    for pid in np.unique(part):
        ys,xs=np.where(stable&(part==pid))
        if len(ys)==0: continue
        n=min(len(ys),max(4,limit//max(len(np.unique(part)),1)))
        ii=np.linspace(0,len(ys)-1,n).astype(int)
        pairs.extend(zip(ys[ii].tolist(),xs[ii].tolist()))
    # All four image boundaries, evenly sampled separately from stable interiors.
    border=np.zeros(part.shape,bool);border[[0,-1],:]=True;border[:,[0,-1]]=True
    ys,xs=np.where(border&(assembly>0))
    if len(ys):
        ii=np.linspace(0,len(ys)-1,min(400,len(ys))).astype(int)
        pairs.extend(zip(ys[ii].tolist(),xs[ii].tolist()))
    pairs=list(dict.fromkeys(pairs))
    return pairs,stable

def check_camera(c,h,w):
    K=np.array(c['K'],float);T=np.array(c['T_world_from_camera'],float);fail=[]
    if K.shape!=(3,3) or T.shape!=(4,4):return K,T,['invalid_K_T_shape']
    if not np.isfinite(K).all() or not np.isfinite(T).all():fail.append('nonfinite_camera')
    if K[0,0]<=0 or K[1,1]<=0 or np.max(np.abs(K[2]-[0,0,1]))>1e-7:fail.append('invalid_K')
    if np.max(np.abs(T[3]-[0,0,0,1]))>1e-7:fail.append('invalid_homogeneous_T')
    R=T[:3,:3]
    if np.max(np.abs(R.T@R-np.eye(3)))>1e-5 or abs(np.linalg.det(R)-1)>1e-5:fail.append('nonrigid_camera_T')
    # Image width and height are checked by adapter because source schema differs.
    return K,T,fail

def audit_view_geometry(scene,deps,meshes,target_names,camera,arrays):
    part=arrays['part_id'];assembly=arrays['assembly'];z=arrays['depth_z'];rng=arrays['depth_range']
    h,w=part.shape;K,T,fail=check_camera(camera,h,w)
    if fail:return {'status':'failed','failures':fail}
    origin=T[:3,3];R=T[:3,:3];Ti=np.linalg.inv(T)
    nearest=[]
    for name in target_names:
        got=meshes[name]['bvh'].find_nearest(Vector(origin))
        if got[0] is not None:nearest.append((float(got[3]),name,list(got[0])))
    if not nearest:return {'status':'failed','failures':['no_target_mesh_for_nearest_distance']}
    distance,name,point=min(nearest,key=lambda a:a[0])
    if not .199<=distance<=.351:fail.append({'distance_outside_20_35cm':distance})
    yy,xx=np.mgrid[:h,:w]
    factor=np.sqrt(1+((xx+.5-K[0,2])/K[0,0])**2+((yy+.5-K[1,2])/K[1,1])**2)
    finite=(np.isfinite(z)&np.isfinite(rng)&(z>0)&(rng>0))
    delta=np.abs(rng[finite]-z[finite]*factor[finite])
    range_p99=float(np.percentile(delta,99)) if delta.size else None
    if range_p99 is None or range_p99>1e-4:fail.append({'range_axial_z_inconsistency_p99_m':range_p99})
    pixels,stable=stratified_samples(part,assembly)
    stats={'stable':{'n':0,'matched':0,'depth_errors':[]},'border_or_boundary':{'n':0,'matched':0,'depth_errors':[]}}
    mismatches=[]
    for y,x in pixels:
        group='stable' if stable[y,x] else 'border_or_boundary';s=stats[group];s['n']+=1
        ray=R@np.array([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],1.])
        ray=ray/np.linalg.norm(ray)
        hit,p,n,i,obj,m=scene.ray_cast(deps,Vector(origin),Vector(ray),distance=100)
        actual=int(obj.pass_index) if hit else 0;expected=int(part[y,x])
        s['matched']+=int(actual==expected)
        if actual!=expected and len(mismatches)<30:mismatches.append({'y':int(y),'x':int(x),'array_part_id':expected,'ray_part_id':actual,'ray_object':obj.name if hit else None})
        if hit and finite[y,x]:
            hit_z=float((Ti@np.r_[p,1])[2]);s['depth_errors'].append(abs(hit_z-float(z[y,x])))
    summary={}
    for group,s in stats.items():
        errors=np.array(s['depth_errors']);agreement=s['matched']/max(s['n'],1)
        summary[group]={'n':s['n'],'id_agreement':agreement,'depth_median_error_m':float(np.median(errors)) if len(errors) else None,'depth_p95_error_m':float(np.percentile(errors,95)) if len(errors) else None}
    interior=summary['stable']
    if interior['n']<100:fail.append({'insufficient_stable_raycast_samples':interior['n']})
    if interior['id_agreement']<.99:fail.append({'stable_first_hit_ID_agreement_below_99pct':interior['id_agreement']})
    if interior['depth_p95_error_m'] is None or interior['depth_p95_error_m']>.002:fail.append({'stable_depth_raycast_error_above_2mm':interior['depth_p95_error_m']})
    inside=[]
    for n,m in meshes.items():
        if m['closed'] and np.all(origin>=m['min']) and np.all(origin<=m['max']):
            q,normal,face,di=m['bvh'].find_nearest(Vector(origin))
            if q is not None and float(di)>.0001:
                from qa_scene_contacts import inside_by_parity
                if inside_by_parity(m,origin)[0]:inside.append(n)
    if inside:fail.append({'camera_inside_closed_meshes':inside})
    edges=np.r_[assembly[0,:],assembly[-1,:],assembly[:,0],assembly[:,-1]]
    return {'status':'passed' if not fail else 'failed','nearest_target_mesh':name,'nearest_target_surface_point_m':point,'nearest_target_surface_distance_m':distance,'range_axial_z_p99_error_m':range_p99,'camera_inside_closed_meshes':inside,'raycast':summary,'raycast_mismatch_examples':mismatches,'visible_assembly_ids':sorted(int(i) for i in np.unique(assembly) if i),'border_touching_assembly_ids':sorted(int(i) for i in np.unique(edges) if i),'failures':fail}
