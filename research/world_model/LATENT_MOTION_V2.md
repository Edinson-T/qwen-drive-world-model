# Motion latent WM v2：首轮训练记录

更新：已从首轮 step 2000 续训到总计 9120 step（30 epoch），保持数据、网络、损失和冻结 autoencoder 不变。
开发集按总 loss 选出的最佳 checkpoint 在 step 9000：占用 IoU 从 0.2673 升至 0.3440，map mIoU 从 0.2722 升至 0.3611，仍低于 ego-warp 的 0.4089。
达到预算上限，未触发早停；没有重新查看官方 validation 子集。详细记录：`outputs/world_model/motion_latent_v2/continue_30ep/README.md`。
下文保留首轮设置与结果，避免混淆两次实验。

本轮已经完成数据准备、小样本检查、GPU 训练和独立官方 validation 子集评估。
结论：流水线可以训练，但首轮未来预测尚未超过按候选自车位姿变换当前占用的基线。
这不是“已解决 WM”的结果，也不是已证明 latent WM 路线无效。训练预算固定，最后一步仍是最佳 checkpoint，尚未证明充分收敛。

## 结构与边界

```
当前地图 + t−1.5/t−1/t−0.5/t 的对象占用
    → Motion 状态编码器（先重建训练，随后冻结）
    → 四帧 BEV latent（每帧 128×8×16）
    → 时序融合 + action token → 两层 Transformer
    → 未来 1 秒 latent
        ├→ 冻结语义 decoder：地图/四类对象占用辅助监督
        └→ 与冻结编码器生成的未来 target latent 做表示预测监督
```

Transformer 处理 128 个空间 token 加 1 个 action token，通道数 128。
没有把普通特征通道直接当成语义 logits，没有使用旧的“原始输入通道 + 语义残差”输出。
这是带冻结重建编码器的动作条件 latent 预测实验，不称为完整标准 JEPA。

本轮使用 Motion 特有的状态编码器，**还没有与 QD 感知 latent 对齐**。这个 checkpoint 不能直接当成可部署的 QD WM。最终接口仍计划使用 QD BEV latent；相机迁移和 reward/RFS 尚未开展。

## 数据协议

- 来源：本地 WOMD v1.3.1 原始 training TFRecords，重建 512 场景，每场景 4 个窗口。
- 开发切分：405 场景/1620 窗口训练，107 场景/428 窗口验证；同场景全部窗口和候选在同一边。scenario 级切分不能单独保证原始行驶片段的所有重叠窗口不跨边，因此另做官方 validation 评估。
- 独立评估：另外读取官方 validation 的 128 场景/512 窗口，未参与权重更新和选 checkpoint，场景 ID 与开发数据无交集。
- 输入网格：x∈[-30,30]m、y∈[-15,15]m，50×100，行=y、列=x，两个轴均递增。
- 历史占用全部直接从原始对象状态栅格化到当前 ego frame，地图只保留当前一份。
- 标签为车辆、行人、骑行者、other 的二维框覆盖；排除 SDC。零值表示无标注框覆盖，不表示可见、可行驶或自由空间。
- 先按 0.15m 画框，再 max 聚合到 0.6m，避免双线性稀疏标签降采样删除车辆。
- action 时间为 0,0.25,...,5s，含 t=0 的 21 个点。heading 来自原始朝向，连续角度插值。
- 每窗口三候选：日志轨迹、两条平滑横向扰动。未来其他车辆保持日志演化，用候选 1s 位姿定义目标坐标系；不模拟其他车辆交互，也不保证扰动候选合法可行驶。
- 自车状态为当前 vx/vy、由过去速度后向差分得到的 ax/ay，不使用未来推导的导航或加速度。
- 训练随机屏蔽少量历史帧，当前帧始终存在。模型显式接收历史有效 mask。

## 固定预算训练

GPU：0，RTX PRO 4500 Blackwell。没有改动旧实验目录、旧 checkpoint 或旧训练器。

| 实验 | 重建编码器步数 | latent WM 步数 | 用途 |
|---|---:|---:|---|
| overfit | 300 | 300 | 4 个训练窗口上的梯度/拟合检查，验证集也用相同窗口，不能作泛化结果 |
| pilot | 1500 | 2000 | 场景开发切分，按验证总 loss 保存 best |

小样本重建占用 IoU=0.660，WM IoU=0.628，证明可以学习，但不代表细节已经充分重建。
开发集重建 IoU=0.588，WM IoU=0.267；最佳 WM checkpoint 为 step 2000。

## 独立官方 validation 子集结果

固定 pilot/best.pt，128 场景、512 窗口、1536 个候选目标，统一网格，固定概率阈值 0.5：

| 方法 | 二维框占用 union IoU |
|---|---:|
| 直接复制当前 | 0.1751 |
| 按候选 ego 位姿变换当前占用 | **0.3752** |
| 物体匀速外推 | 0.8419 |
| 本轮 latent WM | **0.2713** |
| 编码器+decoder 重建真实未来 | 0.5898 |
| WM 换错候选 action | 0.1190 |
| 同一 WM 只保留当前帧 | 0.2565 |

匀速基线额外使用 GT 对象当前速度，属于 privileged baseline；其余输入并不完全相同。
换错 action/只保留当前帧是推理时消融，不是分别训练的严格消融模型。
重建真实未来只是表示/解码器诊断，不是部署成绩或数学上的性能上界。

WM−ego-warp IoU 差的 scenario bootstrap 95% CI 为 [-0.1445,-0.0610]。
这是按场景重采样的 pooled union IoU 差，不是平均每场景 IoU 差。
仅覆盖本轮固定子集、单 seed、单时刻，不能代替完整 benchmark 或最终独立测试。

可视化显示：重建 decoder 已经使一些车辆边缘变模糊，WM 的未来输出进一步扩散。它能利用 action，但还没有充分学习相对于自车坐标变化和其他车辆运动的精确位置。
“历史可能有帮助”只能作初步观察，不能仅凭推理屏蔽历史的差值声称正式收益。

## 下一轮应针对的内容

1. 在开发集检查更保留空间细节的 latent 重建，目标是减少车辆边缘损失。不能把现有 0.59 重建质量视为足够好。
2. 比较显式自车位姿对齐后的特征残差预测，避免让 Transformer 独自学习全部静态场景坐标变换。残差应位于有语义约束的特征空间。
3. 在同一网格与切分上确认 WM 超过 ego-warp，再讨论更多训练数据和更长 horizon。匀速基线应保留，帮助判断模型是否真正学到动力学。
4. 官方 validation 子集本次已查看；若据此改模型，它以后属于已知评估集。最终需要另一份未参与选择的评估子集。
5. 首轮不接 Perception/QD 或 RM，避免把当前预测误差传入下游，再误判相机迁移或选轨路线。

## 文件与复现

新代码集中在 `research/world_model/src/qwen_drive_world_model/`：

- `latent_motion_data.py`：统一网格、原始 Motion 数据准备、历史/候选/匀速标签。
- `latent_motion_model.py`：状态编码器、冻结 target、latent predictor、独立语义 decoder、warp。
- `train_latent_motion.py`：两阶段训练、检查点、开发评估和可视化。
- `evaluate_latent_motion.py`：固定 checkpoint 的独立评估与场景 bootstrap。
- 测试：`research/world_model/tests/test_latent_motion.py`。6 个新增测试连同旧测试共 **28 个通过**，覆盖坐标、平移/旋转、保留车辆、自车排除、时间轴、因果状态、冻结 teacher 的梯度。

训练与指标位于 `outputs/world_model/motion_latent_v2/`：

- `dataset/`、`official_validation/`：两个数据集及版本 manifest。
- `overfit/metrics.json`：小样本检查。
- `pilot/config.json`：参数、数据协议、训练时源代码 SHA256。
- `pilot/autoencoder_best.pt`：冻结的表示编码器与解码器。
- `pilot/best.pt`、`latest.pt`：完整 predictor/autoencoder，含训练协议。
- `pilot/metrics.jsonl`、`metrics.json`、`status.json`：训练曲线数据、最终结果、完成状态。
- `pilot/training_curves.png`、`validation_comparison.png`：开发集图。
- `pilot/official_validation/metrics.json`、`per_scene.json`、`comparison.png`：独立结果与图。
- 根目录同级 `motion_latent_v2_*.log`：准备、训练、评估日志。

设置环境（GPU 命令需在能访问 NVIDIA 设备的环境执行）：

```bash
cd /home/fractal/qwen-drive
export PYTHONPATH="$PWD/src:$PWD/research/world_model/src:$PWD/tools/waymo_e2e"
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="$PWD/outputs/world_model/motion_latent_v2/.mpl"
```

训练器拒绝覆盖已有 run，请使用新的输出路径：

```bash
/home/fractal/miniconda3/envs/qwen-drive/bin/python -u \
  -m qwen_drive_world_model.train_latent_motion \
  --data outputs/world_model/motion_latent_v2/dataset \
  --output outputs/world_model/motion_latent_v2/new_run \
  --ae-steps 1500 --wm-steps 2000 --eval-every 100 --batch-size 16
```

首轮是任务监督的 Motion 表示学习与未来预测实验，不是 QD 模型微调，不是 RFS 实验，也不是完整 Waymo 训练集结果。
