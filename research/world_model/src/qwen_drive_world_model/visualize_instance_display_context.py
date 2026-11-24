"""Show that a reference map changes appearance without changing a WM prediction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F

from .latent_motion_model import warp_current
from .visualize_instance_rollout import panel, PALETTE
from .visualize_latent_bev import MAP_COLORS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/world_model/instance_rollout_v9_20261119'))
    p.add_argument('--parent',type=Path,default=Path('outputs/world_model/instance_motion_v8_20261119/official_validation'))
    args=p.parse_args()
    previous=args.root/'input_gt_wm_comparisons'
    index=json.loads((previous/'index.json').read_text())
    entry=index['examples'][0]
    with np.load(args.parent/'windows.npz') as a:
        row=entry['row']
        assert str(a['scene'][row])==entry['scene'] and int(a['anchor'][row])==entry['anchor']
        current_map=a['history_map'][row]
    with np.load(args.root/'official_targets/targets.npz') as a:
        pose=a['future_ego'][row,1]
    with np.load(previous/f"01_{entry['scene']}_rasters.npz") as a:
        target=a['target'][1]
        prediction=a['prediction'][1]
    state=F.one_hot(torch.tensor(current_map,dtype=torch.long),8).permute(2,0,1).float()[None]
    warped,valid=warp_current(state,torch.tensor(pose)[None])
    background=warped[0].argmax(0).numpy()
    valid=valid[0].numpy()
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(1,3,figsize=(13,9))
    fig.subplots_adjust(left=.07,right=.96,top=.83,bottom=.25,wspace=.4)
    panel(axes[0],prediction,'共享 WM：仅对象框占用',font,True)
    panel(axes[1],prediction,'完全相同的预测框\n＋当前地图投影参考背景',font,True,background,valid)
    panel(axes[2],target,'真实未来对象框 GT\n＋同一地图参考背景',font,True,background,valid)
    iou=float((prediction.any(0)&target.any(0)).sum()/(prediction.any(0)|target.any(0)).sum())
    fig.suptitle(f'同一场景、未来1秒：显示内容差异，不是重新训练\n{entry["scene"]}  |  左图与中图的对象占用 IoU 均为 {iou:.3f}',
                 fontproperties=font,fontsize=15,y=.96,linespacing=1.5)
    legend=[Patch(color=PALETTE[0]/255,label='车辆框占用')]
    legend += [Patch(color=MAP_COLORS[i]/255,label=name) for i,name in [(1,'车道中心线'),(2,'道路标线'),(3,'道路边缘'),(4,'人行横道')]]
    legend.append(Patch(color=np.array([205,205,205])/255,label='当前裁剪地图未覆盖区域'))
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.13),ncol=3,prop=font,frameon=False)
    fig.text(.07,.035,'中、右图地图来自 Motion 当前标注地图，按自车位姿作坐标变换，仅供显示参考；\n'
             '不是共享 WM 的地图预测，也没有作为本版 WM 的输入。加上背景不会改变对象框和 IoU。\n'
             '模型仍需另外解决检测误差、地图约束和长时预测问题，不能靠换画法获得这些能力。',
             fontproperties=font,fontsize=11,linespacing=1.6)
    fig.savefig(previous/'display_content_diagnostic.png',dpi=170)
    plt.close(fig)
    (previous/'display_content_diagnostic.json').write_text(json.dumps(dict(
        scene=entry['scene'],row=row,horizon=1.,occupancy_iou=iou,
        prediction_unchanged=True,map_source='current Motion GT map, geometrically projected for display only',
        map_predicted_by_wm=False,map_used_as_wm_input=False),indent=2)+'\n')


if __name__=='__main__':
    main()
