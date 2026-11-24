"""Actual held-out GT versus shared-WM rasters, with causal inputs above them."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch

from .instance_motion_model import history_features
from .recurrent_instance_motion import RecurrentInstanceMotionPredictor, rollout_boxes, OUTPUT_STEPS, OUTPUT_HORIZONS
from .train_instance_motion import rasterize


PALETTE = np.array([[48,112,190], [224,88,75], [46,166,111], [139,119,163]],np.uint8)


def rgb(occupancy, map_context=None, valid=None):
    if map_context is None:
        result=np.full((*occupancy.shape[1:],3),249,np.uint8)
    else:
        from .visualize_latent_bev import MAP_COLORS
        result=MAP_COLORS[np.asarray(map_context,dtype=int)].copy()
        if valid is not None:
            result[~valid]=[205,205,205]
    for category in range(4):
        result[occupancy[category].astype(bool)] = PALETTE[category]
    return result


def panel(ax, occupancy, title, font, axis_labels=False, map_context=None, valid=None):
    # Native raster rows=y, cols=x. Display forward x vertically and left y to the left.
    ax.imshow(rgb(occupancy,map_context,valid).transpose(1,0,2),origin='lower',extent=(-15,15,-30,30),
              interpolation='nearest',aspect='equal')
    ax.set_xlim(15,-15)
    ax.set_ylim(-30,30)
    ax.add_patch(Rectangle((-1,-2.4),2,4.8,fill=False,lw=1.5,color='#222222'))
    ax.annotate('',xy=(0,6.5),xytext=(0,3),arrowprops=dict(arrowstyle='->',color='#222222',lw=1.3))
    ax.set_title(title,fontproperties=font,fontsize=13,pad=9)
    ax.set_xticks([10,0,-10]);ax.set_yticks([-20,0,20])
    ax.tick_params(labelsize=10)
    ax.grid(alpha=.12)
    if axis_labels:
        ax.set_ylabel('前向 x（米）',fontproperties=font,fontsize=11)
    ax.set_xlabel('横向 y（米），左为正',fontproperties=font,fontsize=10)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instances',type=Path,default=Path('outputs/world_model/instance_motion_v8_20261119/official_instances_dense'))
    p.add_argument('--targets',type=Path,default=Path('outputs/world_model/instance_rollout_v9_20261119/official_targets'))
    p.add_argument('--checkpoint',type=Path,default=Path('outputs/world_model/instance_rollout_v9_20261119/recurrent_seed43/best.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,default=4)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=args.overwrite)
    torch.set_num_threads(4)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    with np.load(args.instances/'instances.npz') as data:
        a={k:data[k] for k in ('history','mask','category','scene','anchor','offsets')}
    with np.load(args.targets/'targets.npz') as data:
        y={k:data[k] for k in ('scene','anchor','offsets','future_ego','future_offsets','future_boxes','future_classes')}
    for key in ('scene','anchor','offsets'):
        np.testing.assert_array_equal(a[key],y[key])
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=RecurrentInstanceMotionPredictor(ckpt['hidden']).eval()
    model.load_state_dict(ckpt['model'])
    scene_order=list(dict.fromkeys(a['scene'].tolist()))
    scene_indices=np.linspace(0,len(scene_order)-1,args.examples,dtype=int)
    rows=[int(np.flatnonzero(a['scene']==scene_order[i])[0]) for i in scene_indices]
    metadata=[]
    with PdfPages(args.output/'GT与共享WM_四场景.pdf') as pdf:
        for example,row in enumerate(rows,1):
            begin,end=a['offsets'][row:row+2]
            tensor=lambda value:torch.as_tensor(value)
            f,b,w,current=history_features(tensor(a['history'][begin:end]).float(),
                                           tensor(a['mask'][begin:end]),tensor(a['category'][begin:end]))
            with torch.inference_mode():
                pred=model(f,b,w)
                boxes=rollout_boxes(current,pred).numpy()
                keep=(pred[3]>0).numpy()
            fig,axes=plt.subplots(3,5,figsize=(18,17),gridspec_kw=dict(hspace=.33,wspace=.27))
            fig.subplots_adjust(left=.05,right=.97,top=.89,bottom=.145)
            for k,time in enumerate((-1.5,-1.,-.5,0.)):
                occupancy=rasterize(a['history'][begin:end,k],a['category'][begin:end],np.zeros(3),
                                     a['mask'][begin:end,k].astype(bool))
                panel(axes[0,k],occupancy,f'输入历史  t{time:+g}秒' if time else '输入历史  当前 t',font,k==0)
            legend=axes[0,4]
            legend.axis('off')
            handles=[Patch(color=color/255,label=name) for color,name in zip(PALETTE,('车辆','行人','骑行者','其他对象'))]
            handles.append(Patch(facecolor='none',edgecolor='#222222',label='参考原点示意框'))
            legend.legend(handles=handles,loc='upper left',prop=font,frameon=False,labelspacing=1.2)
            legend.text(.05,.46,'上：WM使用的四帧关联状态\n\n中：真实未来 GT\n\n下：同一个共享 WM 的预测\n\n关联过程使用过去16帧，\n采样这四帧状态用于预测。',
                        transform=legend.transAxes,fontproperties=font,fontsize=12,va='top',linespacing=1.5)
            metrics=[]
            predictions=[]
            truths=[]
            for column,(step,time) in enumerate(zip(OUTPUT_STEPS,OUTPUT_HORIZONS)):
                pose=y['future_ego'][row,step]
                start,stop=y['future_offsets'][row*10+step:row*10+step+2]
                gt=rasterize(y['future_boxes'][start:stop],y['future_classes'][start:stop],pose,np.ones(stop-start,bool))
                wm=rasterize(boxes[:,step],a['category'][begin:end],pose,keep[:,step])
                union=(gt.any(0)|wm.any(0)).sum()
                iou=float((gt.any(0)&wm.any(0)).sum()/union) if union else None
                label=f'IoU = {iou:.3f}' if iou is not None else 'GT与预测均无框占用'
                panel(axes[1,column],gt,f'GT  |  未来 {time:g} 秒',font,column==0)
                panel(axes[2,column],wm,f'共享 WM  |  未来 {time:g} 秒\n{label}',font,column==0)
                metrics.append(dict(seconds=time,occupancy_iou=iou))
                predictions.append(wm);truths.append(gt)
            scene=str(a['scene'][row])
            fig.suptitle(f'历史输入与真实未来 / 共享世界模型对比\n示例 {example}  ·  场景 {scene}  ·  锚点 {int(a["anchor"][row])}',
                         fontproperties=font,fontsize=19,y=.975,linespacing=1.6)
            fig.text(.055,.025,'输入：Motion 精确历史框，显示当前查询对象的已关联历史，统一到 t 时刻自车坐标。\n'
                     '输出：每列 GT / WM 使用相同的对应未来自车坐标；二维框占用，不含地图，空白不表示已知自由空间。\n'
                     '独立验证场景按源顺序等间距选取，未按预测好坏筛选；图示使用开发集选定的 seed43 / step3000。',
                     fontproperties=font,fontsize=11,linespacing=1.6)
            name=f'{example:02d}_{scene}_input_GT_WM.png'
            fig.savefig(args.output/name,dpi=170)
            pdf.savefig(fig)
            plt.close(fig)
            np.savez_compressed(args.output/f'{example:02d}_{scene}_rasters.npz',
                                target=np.stack(truths),prediction=np.stack(predictions),
                                horizons=np.asarray(OUTPUT_HORIZONS),history=a['history'][begin:end],mask=a['mask'][begin:end])
            metadata.append(dict(file=name,row=row,scene=scene,anchor=int(a['anchor'][row]),horizons=metrics))
    manifest=dict(checkpoint=str(args.checkpoint),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  selection='Four equally spaced source-order independent-validation scenes; earliest cached anchor per scene; no quality filtering',
                  input='Four associated precise historical object states; 16 causal detection frames for association',
                  coordinates='Inputs: current ego. GT/prediction pairs: same logged future ego per horizon.',examples=metadata)
    (args.output/'index.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
