"""Export ablation figures from completed, preserved experiment logs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    args=p.parse_args();root=args.suite
    summary=json.loads((root/'development_summary.json').read_text())
    labels=dict(control='Matched continuation',no_latent='No latent loss',raw_smooth='Raw Smooth L1',raw_mse='Raw MSE',
                map02='Map weight 0.2',no_velocity='No velocity loss',decoder='Unfreeze decoder',
                decoder_recon='Decoder + reconstruction',encoder_decoder='Online encoder + decoder')
    overfit=[json.loads(s) for s in (root/'overfit32_seed43/metrics.jsonl').read_text().splitlines()]
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    fit=[s for s in overfit if s['fit_training'] is not None]
    ax.plot([r['step'] for r in fit],[r['fit_training']['occupancy_iou'] for r in fit],label='32 training windows',marker='o')
    ax.plot([r['step'] for r in overfit],[r['val']['occupancy_iou'] for r in overfit],label='107 development scenes',marker='o')
    ax.set_ylim(0,1);ax.set_xlabel('Optimizer steps');ax.set_ylabel('Occupancy IoU');ax.legend();ax.grid(alpha=.2)
    ax.set_title('Frozen encoder/decoder: fits the small training set, loses generalization')
    fig.savefig(root/'01_overfit.png',dpi=180);plt.close(fig)
    names=list(labels)
    fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
    mean=[summary[n]['mean_last_iou'] for n in names]
    ax.barh(np.arange(len(names)),mean,color=['#467ea9' if n=='control' else '#739d8b' for n in names])
    for i,name in enumerate(names):
        ax.scatter(summary[name]['last_ious'],[i]*len(summary[name]['last_ious']),c='black',s=14,zorder=3)
        ax.text(.5392,i,f'{mean[i]:.4f}',va='center',ha='right',fontsize=8)
    ax.axvline(.5253199553,color='#aa6547',linestyle='--',label='Original V4')
    ax.set_yticks(range(len(names)),[labels[n] for n in names]);ax.invert_yaxis();ax.set_xlim(.50,.54)
    ax.set_xlabel('Development occupancy IoU, mean of 2 training seeds');ax.legend(fontsize=8)
    ax.set_title('Matched 2400-step continuations; dots = individual seeds, NOT best checkpoints')
    fig.savefig(root/'02_development.png',dpi=180);plt.close(fig)
    names=['control','decoder','decoder_recon','encoder_decoder'];x=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axes[0].bar(x-.18,[summary[n]['mean_last_iou'] for n in names],width=.36,label='Own decoder')
    axes[0].bar(x+.18,[summary[n]['last_old_decoder_iou'] for n in names],width=.36,label='Original decoder probe')
    axes[0].set_ylim(.49,.54);axes[0].set_title('Predicted latent decoding');axes[0].legend(fontsize=8)
    axes[1].bar(x,[summary[n]['last_target_reconstruction_iou'] for n in names],color='#bf9562')
    axes[1].set_ylim(.78,.88);axes[1].set_title('Original teacher target latent reconstruction')
    for ax in axes:
        ax.set_xticks(x,[labels[n] for n in names],rotation=23,ha='right');ax.set_ylabel('Occupancy IoU')
    fig.suptitle('Decoder adaptation and retention of original latent interpretation')
    fig.savefig(root/'03_decoder_effect.png',dpi=180);plt.close(fig)
    path=root/'confirmation/metrics.json'
    if path.exists():
        conf=json.loads(path.read_text());names=list(conf['mean_metrics']);labels['original_v4']='Original V4'
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
        means=[conf['mean_metrics'][n]['occupancy_iou'] for n in names]
        ax.bar(range(len(names)),means,color='#6f9d8f')
        for i,n in enumerate(names):
            ax.scatter([i]*len(conf['per_seed'][n]),[s['occupancy_iou'] for s in conf['per_seed'][n]],c='black',s=16)
            ax.text(i,means[i]+.001,f'{means[i]:.4f}',ha='center',fontsize=9)
        ax.set_xticks(range(len(names)),[labels[n] for n in names],rotation=20,ha='right')
        ax.set_ylim(.45,.60);ax.set_ylabel('Occupancy IoU')
        ax.set_title('128 reserved Motion scenes / preselection uses development only / 2 training seeds')
        fig.savefig(root/'04_confirmation.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
