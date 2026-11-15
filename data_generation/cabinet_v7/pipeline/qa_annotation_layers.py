"""Independent numerical annotation checks. Schema adapters must explicitly name all layers."""
import numpy as np

def audit_layers(arrays,part_meta,expected_assemblies,primary=False):
    failures=[];required=('part_id','assembly','branch','wire','junction','depth_z','depth_range')
    missing=[k for k in required if k not in arrays]
    if missing:return {'status':'failed','failures':[{'missing_layers':missing}]}
    shape=arrays['part_id'].shape
    if len(shape)!=2:return {'status':'failed','failures':['part_id_not_2D']}
    for k in required:
        if arrays[k].shape!=shape:failures.append({'shape_mismatch':k,'actual':list(arrays[k].shape),'expected':list(shape)})
    if failures:return {'status':'failed','failures':failures}
    ids=arrays['part_id'];seen=set(int(i) for i in np.unique(ids))
    for k in ('part_id','assembly','branch','wire','junction'):
        a=arrays[k]
        if not np.issubdtype(a.dtype,np.integer):failures.append({'noninteger_ID_layer':k})
        if np.any(a<0):failures.append({'negative_ID_layer':k})
    missing_meta=sorted(seen-set(part_meta)-{0})
    if missing_meta:failures.append({'observed_parts_missing_metadata':missing_meta})
    count_by_part={str(pid):int(np.count_nonzero(ids==pid)) for pid in sorted(seen)}
    mapping_errors={}
    for layer in ('assembly','branch','wire','junction'):
        expected=np.zeros(shape,np.int64)
        for pid in seen:
            if pid==0:continue
            m=part_meta.get(pid,{})
            expected[ids==pid]=int(m.get(layer,0))
        err=int(np.count_nonzero(arrays[layer]!=expected))
        if err:mapping_errors[layer]=err
    if mapping_errors:failures.append({'part_to_hierarchy_mapping_mismatch_pixels':mapping_errors})
    actual_assemblies=sorted(int(i) for i in np.unique(arrays['assembly']) if i)
    unexpected=sorted(set(actual_assemblies)-set(expected_assemblies))
    if unexpected:failures.append({'undeclared_visible_assembly_ids':unexpected})
    if primary and set(actual_assemblies)!=set(expected_assemblies):failures.append({'primary_does_not_show_all_assemblies':actual_assemblies,'expected':list(expected_assemblies)})
    if not actual_assemblies:failures.append('all_background_target_view')
    if np.any((arrays['junction']>0)&(arrays['branch']>0)):failures.append('junction_has_fabricated_branch_assignment')
    if np.any((arrays['junction']>0)&(arrays['junction']!=arrays['assembly'])):failures.append('junction_does_not_resolve_to_assembly')
    if np.any((arrays['branch']>0)&(arrays['assembly']==0)):failures.append('branch_without_parent_assembly')
    if np.any((arrays['wire']>0)&(arrays['assembly']==0)):failures.append('wire_without_parent_assembly')
    for k in ('depth_z','depth_range'):
        a=arrays[k];fg=arrays['assembly']>0
        if not np.isfinite(a).all():failures.append({'nonfinite_GT_depth':k,'count':int((~np.isfinite(a)).sum())})
        if np.any(a<0):failures.append({'negative_GT_depth':k})
        if np.any(fg&(a<=0)):failures.append({'foreground_missing_GT_depth':k,'count':int(np.count_nonzero(fg&(a<=0)))})
    if 'depth_input' in arrays and 'depth_valid' in arrays:
        z=arrays['depth_input'];v=arrays['depth_valid']
        if z.shape!=shape or v.shape!=shape:failures.append('sensor_depth_shape_mismatch')
        elif not set(np.unique(v)).issubset({0,1,False,True}):failures.append('depth_valid_not_boolean')
        elif np.any(v.astype(bool)&(~np.isfinite(z)|(z<=0))):failures.append('depth_valid_claims_invalid_measurement')
    return {'status':'passed' if not failures else 'failed','shape':list(shape),'visible_assembly_ids':actual_assemblies,'part_pixel_counts':count_by_part,'foreground_pixels':int(np.count_nonzero(arrays['assembly'])),'junction_pixels':int(np.count_nonzero(arrays['junction'])),'visible_wire_pixels':int(np.count_nonzero(arrays['wire'])),'failures':failures}
