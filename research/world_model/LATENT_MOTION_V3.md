# Motion latent WM v3：细节保留、几何残差和数据扩充

本轮完成了三组实验和一次新的官方 validation 子集评估。保持 1 秒未来预测任务。
最优版本是 `outputs/world_model/motion_latent_v3/expanded2048/best.pt`。

**结果：固定开发集占用 IoU 从 v2 的 0.3440 提升到 0.4827；新官方验证子集上 WM=0.4677，ego-warp=0.3838。**
这证明本轮组合改进能超过静态场景变换基线。它仍明显低于使用 GT 对象速度的匀速外推，不能称为已解决动力学或可部署到 QD。

## 实际改动

1. 保留 latent WM 主线，未来状态仍在 latent 空间预测。
2. 编码器减少一次 stride 下采样，latent 网格从 8×16 改为 16×32，通道数维持 128。
3. 每条候选用其 1s 位姿，把四帧历史 latent 变换到候选 future ego frame，再预测特征残差。残差位于已训练的特征空间，初始为零；不把任意 latent 通道当语义 logits。
4. 高分辨率特征经过时序融合后，池化为 8×16 送入 Transformer，因此注意力仍是 128 个空间 token + 1 个 action token。Transformer 输出上采样后，与局部高分辨率卷积分支结合，输出 16×32 的未来 latent。
5. Motion 原始数据扩展到 2048 场景，CPU 栅格化改为 8 worker，保持输出顺序确定。已用归档原始场景验证串行/并行结果与旧缓存各字段完全一致。
6. 固定开发集：严格校验原验证场景、anchor、地图、历史占用、目标占用、候选 action、ego status、CV 标签逐项相等，并检查它们不进入扩充训练集。
7. 新训练保存源码快照、参数、数据 hash、父 checkpoint/teacher hash 和指标。旧模型未被覆盖。

## 实验与开发集结果

所有开发指标均在原 107 场景、428 窗口、1284 个候选目标上，网格和阈值 0.5 不变。

| 版本 | 实际训练场景 | Encoder 重建 IoU | WM 占用 IoU | map mIoU |
|---|---:|---:|---:|---:|
| v2，30 epoch | 405 | 0.5882 | 0.3440 | 0.3611 |
| A：复用旧 encoder，仅几何残差 | 405 | 0.5882 | 0.3481 | 0.3838 |
| B：细节 encoder + 几何残差 | 405 | 0.8649 | 0.4392 | 0.6220 |
| C：冻结 B encoder，扩充训练数据 | **1623** | 0.8649 | **0.4827** | **0.6330** |

同一开发集：复制当前=0.1714，ego-warp=0.4089，GT 速度匀速外推=0.8435。

A 复用 v2 冻结 encoder/cache，仅重新训练 predictor。A 最佳 step 2100，step 4500 早停。
B 的 encoder 训练 4500 step，WM 最佳 step 2400，step 4800 早停。B 同时改变分辨率、下采样方式和表示训练量，因此不能把它的全部收益单独归因于 token 数。

C 的数据共有 2048 场景/8192 窗口：1623 场景/6492 窗口用于训练，共 19476 个候选样本；425 个 hash 验证场景中仅使用原来的 107 场景，其他验证场景不加入训练。每 epoch 1218 个 batch，batch size=16。

C 从 B 最佳 predictor 初始化，encoder/decoder 逐字节不变，重建扩充数据的冻结 latent cache；AdamW 重置、学习率 1e-4。原验证 IoU 在初始化时成功复现。
上限 24000 step，每 600 step 检查开发集，连续 8 次未改善 0.001 时早停。
实际新增训练至 step 7200（扩充数据的 5.91 epoch）早停，最佳 checkpoint 在 step 2400（扩充数据的 1.97 epoch）。这些是扩充阶段的计数，不是从头训练总 epoch。

A/B 按总 loss 选 checkpoint。C 在训练开始前指定按占用 IoU 选 checkpoint，因为对象占用预测是本阶段主要目标；loss 和地图指标继续记录。C 同时改变数据量、学习率、优化器初始化和选择标准，所以不能把 C−B 全部归因为纯数据量效应。

## 新的独立评估

本轮事先从官方 Motion validation 中跳过已看过的前 128 个合格场景，取后续 128 个合格场景：512 窗口、1536 候选目标。
场景 ID 与旧官方验证子集及全部训练/开发数据均无交集。
先依据固定开发集保存 `selected_model.json` 和模型 SHA256，再运行新子集评估；这次选中 C。没有根据该子集的结果改权重或阈值。

| 方法 | 对象占用 IoU |
|---|---:|
| 复制当前 | 0.1658 |
| ego-warp | 0.3838 |
| **选定 WM** | **0.4677** |
| GT 对象速度匀速外推 | 0.8298 |
| 真实未来编码后重建（诊断） | 0.8677 |
| WM 换错候选 action（推理消融） | 0.0762 |
| 同一 WM 只保留当前帧（推理消融） | 0.3494 |

WM precision=0.6200、recall=0.6556、map mIoU=0.6326。
按场景 bootstrap 的 pooled union IoU 差 `WM−ego-warp` 95% CI=[+0.0642,+0.1039]。
只保留当前帧与换错 action 均为推理屏蔽/替换，不是独立重新训练的消融；不能当作正式单因素训练实验。
匀速基线使用 privileged GT 对象速度，不能冒充当前相机系统部署结果。

这份新子集现在已经查看过，若下一轮根据它调整设计，它不再是未参与选择的最终评估集。

## 判断与下一步

本轮明显改善了表示重建和预测质量，且在新子集上超过 ego-warp。
可视化仍能看到移动对象位置偏差、消失或扩散。重建质量已有约 0.87，而 WM 约 0.47，后续重点应是动态运动预测，不能继续把主要问题都归因于 decoder 模糊。
静态场景对齐可以解析处理；动态对象的历史关联与运动外推需要进一步改善。当前结果尚不足以证明会提升 RFS。

依然适用的边界：

- 仅 1s Motion 二维框占用，非完整体素、free-space 或 visibility GT。
- 多候选采用日志其他车辆未来和候选坐标变换，不是反事实交互模拟。
- 单 seed，有限子集；还未与 QD latent 对齐、未接相机或 RM、未跑 RFS。
- 单元测试共 29 个通过，包含 residual 初始严格等于几何 warp、可训练梯度和高分辨率输出检查；另完成串行/并行数据与旧缓存一致性检查。

## 入口和产物

代码仍集中在 `latent_motion_data.py`、`latent_motion_model.py`、`train_latent_motion.py`、`evaluate_latent_motion.py`，通过配置区分实验，没有再复制一套训练器。

核心产物位于 `outputs/world_model/motion_latent_v3/`：

- `expanded2048/best.pt`：本轮选定模型。
- `expanded2048/config.json`、`metrics.json`、`source/`：参数、曲线数据、冻结时的源码。
- `selected_model.json`：看新评估结果之前的模型选择记录。
- `selected_evaluation/metrics.json`、`per_scene.json`、`comparison.png`：新官方验证结果。
- `improvement_summary.png`：开发结果、重建质量和新验证基线的分面图。
- `expanded_training_curves.png`：扩充阶段训练曲线。
- `summary.json`：汇总。

环境：

```bash
cd /home/fractal/qwen-drive
export PYTHONPATH="$PWD/src:$PWD/research/world_model/src:$PWD/tools/waymo_e2e"
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="$PWD/outputs/world_model/motion_latent_v2/.mpl"
```

C 的训练命令（重跑必须改成新的 output）：

```bash
/home/fractal/miniconda3/envs/qwen-drive/bin/python -u -m qwen_drive_world_model.train_latent_motion \
  --data outputs/world_model/motion_latent_v3/dataset2048 \
  --output outputs/world_model/motion_latent_v3/expanded2048 \
  --finetune-from outputs/world_model/motion_latent_v3/detail_geometry/best.pt \
  --val-scenes-from outputs/world_model/motion_latent_v2/dataset/windows.npz \
  --detail-encoder --token-height 16 --token-width 32 --ego-warp-residual \
  --wm-lr 0.0001 --wm-steps 24000 --eval-every 600 \
  --early-stop-patience 8 --early-stop-min-delta 0.001 \
  --selection-metric occupancy_iou --batch-size 16
```
