"""QD-inspired vector presentation of unchanged instance-WM predictions."""
import argparse
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon, Circle, Patch
from matplotlib.lines import Line2D
import numpy as np
import torch

from .instance_motion_model import history_features
from .motion_gt import scenario_map_features, global_to_ego, ego_pose_from_state
from .protos import womd_pb2
from .recurrent_instance_motion import RecurrentInstanceMotionPredictor, rollout_boxes, OUTPUT_STEPS, OUTPUT_HORIZONS

COLORS=('#ff8a00','#eb3b5a','#20bf6b','#8d6e63')


def local_xy(points,pose):
    delta=points[:,:2]-pose[:2]
    c,s=np.cos(pose[2]),np.sin(pose[2])
    return np.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]),-1)


def draw_map(ax,features,pose):
    for kind,subtype,points in features:
        xy=local_xy(points,pose)
        if not len(xy) or np.any(xy.min(0)>[33,18]) or np.any(xy.max(0)<[-33,-18]):
            continue
        view=np.c_[-xy[:,1],xy[:,0]]
        if kind in (8,9,10) and len(view)>=3:
            color={8:'#4bb7ae',9:'#f9ad6a',10:'#8bc34a'}[kind]
            ax.add_patch(Polygon(view,facecolor=color,edgecolor=color,alpha=.16,lw=.6,zorder=1,clip_on=True))
        elif kind in (3,4,5):
            color={3:'#c3cdd7',4:'#ffbf00' if subtype>=4 else '#aebbc8',5:'#e76145'}[kind]
            style=(0,(3,3)) if kind==3 or (kind==4 and subtype in (1,4,5)) else '-'
            ax.plot(view[:,0],view[:,1],color=color,lw=.65 if kind==3 else .95,
                    ls=style,alpha=.8,zorder=1,solid_capstyle='round',clip_on=True)


def draw_boxes(ax,boxes,classes,keep,pose):
    centers=local_xy(boxes,pose)
    for i in np.flatnonzero(keep):
        x,y=centers[i];length,width=boxes[i,2:4];angle=boxes[i,4]-pose[2]
        if abs(x)>32+length or abs(y)>17+width:
            continue
        vertices=np.array([[length/2,width/2],[length/2,-width/2],[-length/2,-width/2],[-length/2,width/2]])
        c,s=np.cos(angle),np.sin(angle)
        vertices=vertices@np.array([[c,s],[-s,c]])+centers[i]
        vertices=np.c_[-vertices[:,1],vertices[:,0]]
        color=COLORS[int(classes[i])]
        ax.add_patch(Polygon(vertices,facecolor=color,edgecolor='none',alpha=.18,zorder=3,clip_on=True))
        ax.add_patch(Polygon(vertices,facecolor='none',edgecolor=color,lw=1.05,zorder=4,joinstyle='round',clip_on=True))
        front=vertices[:2].mean(0);center=np.array([-y,x])
        ax.plot([center[0],front[0]],[center[1],front[1]],color=color,lw=.85,zorder=4,clip_on=True)


def panel(ax,features,boxes,classes,keep,pose,title,font):
    ax.set_facecolor('#fbfcfd')
    ax.set(xlim=(-15,15),ylim=(-30,30),aspect='equal',xticks=[],yticks=[])
    for spine in ax.spines.values():
        spine.set_color('#d7dde4');spine.set_linewidth(.7)
    for radius in (10,20):
        ax.add_patch(Circle((0,0),radius,fill=False,edgecolor='#e8edf2',lw=.55,zorder=0))
    draw_map(ax,features,pose)
    draw_boxes(ax,boxes,classes,keep,pose)
    ego=np.array([[-.92,-2.3],[-.92,1.4],[-.5,2.3],[.5,2.3],[.92,1.4],[.92,-2.3]])
    ax.add_patch(Polygon(ego,facecolor='white',edgecolor='#27313b',lw=1.,zorder=6))
    ax.annotate('',xy=(0,5),xytext=(0,2.7),arrowprops=dict(arrowstyle='-|>',color='#27313b',lw=1.),zorder=6)
    ax.set_title(title,fontproperties=font,fontsize=12,color='#2f3437',pad=9)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/world_model/instance_rollout_v9_20261119'))
    p.add_argument('--instances',type=Path,default=Path('outputs/world_model/instance_motion_v8_20261119/official_instances_dense'))
    args=p.parse_args();torch.set_num_threads(4)
    old=args.root/'input_gt_wm_comparisons'
    manifest=json.loads((old/'index.json').read_text())
    metadata=json.loads((args.root/'official_targets/dataset.json').read_text())
    records={r['scene']:r for r in metadata['records']}
    with np.load(args.instances/'instances.npz') as data:
        a={k:data[k] for k in ('history','mask','category','offsets')}
    with np.load(args.root/'official_targets/targets.npz') as data:
        y={k:data[k] for k in ('future_offsets','future_boxes','future_classes','future_ego')}
    ckpt=torch.load(manifest['checkpoint'],map_location='cpu',weights_only=False)
    model=RecurrentInstanceMotionPredictor(ckpt['hidden']).eval()
    model.load_state_dict(ckpt['model'])
    out=args.root/'qd_style_comparisons';out.mkdir(exist_ok=True)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    exported=[]
    with PdfPages(out/'QD风格_历史输入与五时刻对比.pdf') as pdf:
        for number,entry in enumerate(manifest['examples'],1):
            record=records[entry['scene']]
            with Path(record['path']).open('rb') as handle:
                handle.seek(record['offset']);size=struct.unpack('<Q',handle.read(8))[0]
                handle.read(4);payload=handle.read(size)
            scenario=womd_pb2.Scenario();scenario.ParseFromString(payload)
            assert scenario.scenario_id==entry['scene']
            current_pose=ego_pose_from_state(scenario.tracks[scenario.sdc_track_index].states[entry['anchor']])
            features=[(f.kind,f.subtype,global_to_ego(f.points,current_pose)) for f in scenario_map_features(scenario) if len(f.points)]
            row=entry['row'];begin,end=a['offsets'][row:row+2]
            f,b,w,current=history_features(torch.tensor(a['history'][begin:end]),torch.tensor(a['mask'][begin:end]),torch.tensor(a['category'][begin:end]))
            with torch.inference_mode():
                pred=model(f,b,w);boxes=rollout_boxes(current,pred).numpy();keep=(pred[3]>0).numpy()
            fig,axes=plt.subplots(3,5,figsize=(17,15),gridspec_kw=dict(hspace=.26,wspace=.13))
            fig.subplots_adjust(left=.035,right=.97,top=.90,bottom=.12)
            for k,time in enumerate((-1.5,-1.,-.5,0.)):
                title=f'历史输入  {time:g} s' if time else '历史输入  当前'
                panel(axes[0,k],features,a['history'][begin:end,k],a['category'][begin:end],
                      a['mask'][begin:end,k].astype(bool),np.zeros(3),title,font)
            axes[0,4].axis('off')
            legend=[Patch(facecolor=c,edgecolor=c,alpha=.7,label=n) for c,n in zip(COLORS,('车辆','行人','骑行者','其他'))]
            legend += [Line2D([0],[0],color='#ffbf00',lw=1.5,label='道路标线'),Line2D([0],[0],color='#e76145',lw=1.5,label='道路边缘')]
            axes[0,4].legend(handles=legend,loc='upper left',prop=font,frameon=False,labelspacing=1.2)
            axes[0,4].text(.05,.35,'输入置顶\nGT 与 WM 上下对齐\n\n朝前为上，左侧为左\n淡圆环：10 / 20 米\n框内短线：对象朝向',
                           fontproperties=font,fontsize=11,color='#606b77',va='top',linespacing=1.8)
            for column,(step,time) in enumerate(zip(OUTPUT_STEPS,OUTPUT_HORIZONS)):
                pose=y['future_ego'][row,step];start,stop=y['future_offsets'][row*10+step:row*10+step+2]
                panel(axes[1,column],features,y['future_boxes'][start:stop],y['future_classes'][start:stop],np.ones(stop-start,bool),pose,
                      f'GT   +{time:g} s',font)
                iou=entry['horizons'][column]['occupancy_iou']
                panel(axes[2,column],features,boxes[:,step],a['category'][begin:end],keep[:,step],pose,
                      f'共享 WM   +{time:g} s\nIoU  {iou:.3f}',font)
            fig.suptitle(f'历史输入与未来对象运动预测\n{entry["scene"]}  ·  共享递推模型',fontproperties=font,fontsize=17,
                         color='#2f3437',y=.97,linespacing=1.6)
            fig.text(.05,.055,'底图为 Motion 静态地图参考，未由 WM 预测、也未输入本版 WM；未添加点云或相机照片。',
                     fontproperties=font,fontsize=10,color='#697480')
            fig.text(.05,.032,'轮廓来自原始标注 / 模型预测框；IoU 仍按原 0.6 米占用网格计算，模型、数值与场景选择均未更改。',
                     fontproperties=font,fontsize=10,color='#697480')
            name=f'{number:02d}_{entry["scene"]}_qd_style.png'
            fig.savefig(out/name,dpi=180);pdf.savefig(fig);plt.close(fig)
            exported.append(dict(scene=entry['scene'],image=name,source_record=record,original_raster_metrics=entry['horizons']))
    (out/'index.json').write_text(json.dumps(dict(style_reference='outputs/visualizations/perception/demo/4d0d1ccbb1035a90.png',
        map='Full-scene Motion vector labels for display only; not WM input/output',box_source=manifest['checkpoint'],
        numeric_predictions_unchanged=True,examples=exported),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps([r['image'] for r in exported]),flush=True)


if __name__=='__main__':
    main()
