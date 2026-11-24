"""Decode map and class occupancy into metric semantic BEV comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .latent_motion_model import LatentConfig,StateAutoencoder,LatentPredictor,pack_state,warp_current


MAP_COLORS=np.array([[245,245,240],[125,180,220],[245,199,65],[100,110,120],
                     [225,207,167],[210,66,66],[178,140,190],[210,215,210]],dtype=np.uint8)
OBJECT_COLORS=np.array([[40,108,190],[220,60,65],[235,139,30],[145,75,170]],dtype=np.uint8)


def rgb_bev(maps,occupancy,valid=None):
    image=MAP_COLORS[maps.astype(int)].astype(float)
    occupied=occupancy.max(0)>=.5
    labels=occupancy.argmax(0)
    if valid is not None:
        image[~valid]=[205,205,205]
    image[occupied]=OBJECT_COLORS[labels[occupied]]
    return image.astype(np.uint8)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--scenes',type=int,default=6)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device(args.device)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    config=LatentConfig(**ckpt['config'])
    ae=StateAutoencoder(config).to(device).eval();ae.load_state_dict(ckpt['autoencoder'])
    wm=LatentPredictor(config).to(device).eval();wm.load_state_dict(ckpt['predictor'])
    with np.load(args.data/'windows.npz') as payload:a={k:payload[k] for k in payload.files}
    _,first=np.unique(a['scene'],return_index=True)
    first=np.sort(first);indices=first[np.linspace(0,len(first)-1,min(args.scenes,len(first)),dtype=int)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle,Patch
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    titles=['当前标注','真实未来标注（target）','WM 预测','静态变换（warp）','GT 速度匀速外推（CV）','真实未来压缩重建']
    metadata=[]
    for number,index in enumerate(indices):
        maps=torch.tensor(a['history_map'][index],device=device)[None,None].expand(1,4,-1,-1)
        occ=torch.tensor(a['history_occ'][index],device=device)[None]
        h=ae.encoder(pack_state(maps,occ).flatten(0,1)).reshape(1,4,config.channels,config.token_h,config.token_w)
        action=torch.tensor(a['actions'][index,0],device=device)[None]
        status=torch.tensor(a['status'][index],device=device)[None]
        prediction=ae.decoder(wm(h,action,status))[0]
        target_map=torch.tensor(a['target_map'][index,0],device=device)[None]
        target_occ=torch.tensor(a['target_occ'][index,0],device=device)[None]
        reconstruction=ae(pack_state(target_map,target_occ))[0]
        packed=pack_state(maps[:,0],occ[:,-1]);warped,valid=warp_current(packed,action[:,4])
        warp_map=warped[0,:8].argmax(0).cpu().numpy();warp_occ=warped[0,8:].cpu().numpy()
        cells=[(a['history_map'][index],a['history_occ'][index,-1],None),
               (a['target_map'][index,0],a['target_occ'][index,0],None),
               (prediction[:8].argmax(0).cpu().numpy(),prediction[8:].sigmoid().cpu().numpy(),None),
               (warp_map,warp_occ,valid[0].cpu().numpy()),
               (warp_map,a['cv_occ'][index,0],valid[0].cpu().numpy()),
               (reconstruction[:8].argmax(0).cpu().numpy(),reconstruction[8:].sigmoid().cpu().numpy(),None)]
        fig,axes=plt.subplots(1,6,figsize=(20,5))
        for col,(m,o,v) in enumerate(cells):
            # Display ego-forward up, ego-left left. Native array remains [y,x].
            image=rgb_bev(m,o,v).transpose(1,0,2)[::-1,::-1]
            ax=axes[col];ax.imshow(image,extent=(15,-15,-30,30),origin='upper',interpolation='nearest')
            ax.add_patch(Rectangle((-1,-2.4),2,4.8,fill=False,edgecolor='black',linewidth=1.3))
            ax.annotate('',xy=(0,4),xytext=(0,0),arrowprops=dict(arrowstyle='->',color='black',lw=1.2))
            ax.set_title(titles[col],fontproperties=font,fontsize=10)
            ax.set_xlabel('y 左向 / m',fontproperties=font);ax.set_ylabel('x 前向 / m',fontproperties=font)
            ax.set_aspect('equal')
        legend=[Patch(color=OBJECT_COLORS[i]/255,label=name) for i,name in enumerate(['车辆','行人','骑行者','其他对象'])]
        legend += [Patch(color=MAP_COLORS[i]/255,label=name) for i,name in [(1,'车道中心线'),(2,'道路标线'),(3,'道路边缘'),(4,'人行横道')]]
        fig.legend(handles=legend,loc='lower center',ncol=8,prop=font,fontsize=9)
        fig.suptitle(f"场景 {a['scene'][index]} / anchor {int(a['anchor'][index])} / 未来 1 秒 / 占用阈值 0.5",fontproperties=font)
        fig.tight_layout(rect=(0,.12,1,.92))
        path=args.output/f'bev_{number:02d}.png';fig.savefig(path,dpi=170);plt.close(fig)
        metadata.append(dict(scene=str(a['scene'][index]),anchor=int(a['anchor'][index]),candidate=0,path=str(path)))
    result=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                data=str(args.data.resolve()),examples=metadata,threshold=.5,
                notes=['Semantic map and object-footprint BEV, not future camera photographs or dense physical occupancy.',
                       'No target labels are used for the WM prediction panel.',
                       'GT reconstruction encodes true future labels; it is a representation diagnostic.',
                       'CV uses privileged GT object velocity; map uses warped current map.',
                       'Map background is not certified free/drivable space.'])
    (args.output/'index.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
