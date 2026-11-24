"""Track-matched cached-object diagnostics; labels never enter predictors."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .latent_motion_data import GRID
from .motion_gt import _box_footprint


def object_mask(box):
    image=Image.new('L',(400,200))
    ImageDraw.Draw(image).polygon([tuple(p) for p in GRID.pixels(_box_footprint(box))*4],fill=1)
    return np.asarray(image).reshape(50,4,100,4).max((1,3)).astype(bool)


def extract_instances(a, selected_rows, output):
    lookup={(str(a['scene'][i]),int(a['anchor'][i])):int(i) for i in selected_rows}
    records=[];matched_rows=set();rejected=0;ego_excluded=0
    for path in sorted(Path('outputs/world_model/motion_gt_v1').glob('pilot-*.npz')):
        with np.load(path) as old:
            selected=[(i,lookup[(str(s),int(t))]) for i,(s,t) in enumerate(zip(old['scenario_ids'],old['t_frames']))
                      if (str(s),int(t)) in lookup]
            if not selected:continue
            fields={k:old[k] for k in ('boxes_t','boxes_1s','box_labels_t','box_labels_1s',
                                      'box_ids_t','box_ids_1s','relative_poses')}
        for i,row in selected:
            pose=fields['relative_poses'][i,1]
            expected=a['actions'][row,0,4]
            angle=(pose[2]-expected[2]+np.pi)%(2*np.pi)-np.pi
            if not np.allclose(pose[:2],expected[:2],atol=2e-4) or abs(angle)>2e-4:
                raise ValueError('cached frame mismatch')
            current={int(identity):j for j,identity in enumerate(fields['box_ids_t'][i]) if identity>=0}
            future={int(identity):j for j,identity in enumerate(fields['box_ids_1s'][i]) if identity>=0}
            for identity in sorted(current.keys()&future.keys()):
                j,k=current[identity],future[identity]
                now=fields['boxes_t'][i,j];then=fields['boxes_1s'][i,k]
                category=int(fields['box_labels_t'][i,j])
                if category!=int(fields['box_labels_1s'][i,k]):continue
                if np.linalg.norm(now[:2])<1e-3 and np.linalg.norm(then[:2])<1e-3:
                    ego_excluded+=1;continue
                mask=object_mask(then)
                if not mask.any() or (mask&a['target_occ'][row,0,category].astype(bool)).sum()/mask.sum()<.8:
                    rejected+=1;continue
                corners=_box_footprint(then)
                inside=bool((np.abs(corners[:,0])<29.7).all() and (np.abs(corners[:,1])<14.7).all())
                records.append(dict(row=row,track_id=identity,category=category,speed=float(np.linalg.norm(now[7:9])),
                                    current=now.copy(),future=then.copy(),fully_inside=inside))
            matched_rows.add(row)
    if not records:raise ValueError('no exact cached track matches')
    np.savez_compressed(output,**{k:np.asarray([v[k] for v in records]) for k in records[0]})
    return records,dict(windows=len(matched_rows),scenes=len({str(a['scene'][r]) for r in matched_rows}),
                        objects=len(records),rejected_mask_alignment=rejected,excluded_ego_boxes=ego_excluded,
                        source='legacy cached Motion boxes/IDs, exact scene+anchor and +1s ego pose verified',
                        scope='logged candidate0, objects labeled at both times and within legacy 64-box/height/center filtering',
                        speed_bins_mps=[.5,5.],labels_used_only_for_evaluation=True)


def summarize_instances(records, probabilities):
    """probabilities maps original row -> [4,50,100]; component IoU includes misses.

    Component extraction merges touching predictions. Occupied-pixel recall is
    reported independently to avoid interpreting merged components as motion loss.
    False positives without a GT association cannot be assigned a true speed bin.
    """
    groups={};unassigned=0;per_object={}
    for row in sorted({r['row'] for r in records}):
        p=probabilities[row]>=.5
        for category in range(4):
            truth=[r for r in records if r['row']==row and r['category']==category]
            if not truth:continue
            gt=[object_mask(r['future']) for r in truth]
            labels,count=ndimage.label(p[category],structure=np.ones((3,3),dtype=np.uint8))
            predicted=[labels==j for j in range(1,count+1)]
            ious=np.zeros((len(gt),len(predicted)))
            for i,g in enumerate(gt):
                for j,v in enumerate(predicted):ious[i,j]=(g&v).sum()/max((g|v).sum(),1)
            matches={}
            if predicted:
                rows,cols=linear_sum_assignment(1-ious)
                matches={int(i):int(j) for i,j in zip(rows,cols) if ious[i,j]>=.05}
            unassigned+=len(predicted)-len(matches)
            for i,record in enumerate(truth):
                hit=i in matches
                center_error=None;raster_error=6.
                if hit:
                    pixels=np.argwhere(predicted[matches[i]])
                    center=np.array([pixels[:,1].mean()*.6-29.7,pixels[:,0].mean()*.6-14.7])
                    expected=record['future'][:2]
                    if record['fully_inside']:center_error=float(np.linalg.norm(center-expected))
                    g=np.argwhere(gt[i]);gc=np.array([g[:,1].mean()*.6-29.7,g[:,0].mean()*.6-14.7])
                    raster_error=min(6.,float(np.linalg.norm(center-gc)))
                value=dict(iou=float(ious[i,matches[i]]) if hit else 0.,
                           pixel_recall=float((p[category]&gt[i]).sum()/gt[i].sum()),
                           miss=not hit,center_error=center_error,capped_error=raster_error)
                speed='stationary' if record['speed']<.5 else 'slow' if record['speed']<5 else 'fast'
                cls=('vehicle','pedestrian','cyclist','other')[category]
                per_object[f"{row}:{record['track_id']}"]=dict(**value,category=cls,speed_bin=speed,
                    speed_mps=record['speed'],fully_inside=record['fully_inside'],row=row)
                for key in ('all','speed/'+speed,'class/'+cls,'class_speed/'+cls+'/'+speed):
                    groups.setdefault(key,[]).append(value)
    result={}
    for key,values in groups.items():
        valid=[r['center_error'] for r in values if r['center_error'] is not None]
        result[key]=dict(objects=len(values),instance_iou_with_misses=float(np.mean([r['iou'] for r in values])),
                         target_pixel_recall=float(np.mean([r['pixel_recall'] for r in values])),
                         miss_rate=float(np.mean([r['miss'] for r in values])),
                         matched_full_inside_center_error_m=float(np.mean(valid)) if valid else None,
                         center_error_evaluated_objects=len(valid),
                         capped_raster_centroid_error_m=float(np.mean([r['capped_error'] for r in values])))
    return dict(groups=result,unassigned_prediction_components=unassigned,per_object=per_object,
                notes=['Instance IoU with misses=0 is not pooled full-grid IoU; unmatched FP have no true speed bin.',
                       'Physical center error only for matched, fully inside GT boxes; report misses alongside it.',
                       'Centroid errors correspond to 1s displacement-vector errors for matched tracks; 6m cap penalizes missing components.',
                       'Touching predicted objects merge into components; target-pixel recall is a complementary non-instance metric.'])
