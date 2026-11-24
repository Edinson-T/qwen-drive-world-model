# Motion latent WM v4：运动范围、速度辅助监督与语义 BEV

本轮继续了两组训练。**仍未达到 GT 速度 CV 的强参考水平，不称为训练验收通过。**
最新模型由开发集选定：`outputs/world_model/motion_latent_v4/velocity_aux/best.pt`。

## target 与原 teacher 图

- target：原始真实未来地图和对象框占用。
- 原图的 teacher：真实未来标签经冻结 encoder→decoder 后的重建结果。已在新的语义图中改名为“真实未来压缩重建”。
- prediction：只用历史 latent、自车状态、候选轨迹预测未来 latent，再解码。没有输入真实未来。
- warp：变换当前状态到候选未来自车坐标系，假设其他对象世界位置不变。
- CV：用 GT 对象速度匀速外推。这个强基线同时能使用全局标注中的当前对象，并非只限于 WM 当前裁剪视野，因此与 WM 的输入信息不完全相同。

原 target 和 GT 重建的差别反映表示压缩损失；WM 和 target 的差别才是最终未来预测误差。GT 重建不是未来预测器，也不是数学上的模型上界。

## 训练改进

1. 新增宽空间运动分支，使用 dilation=1/2/4 的卷积，扩大关联历史移动对象的范围。保留原局部分支、冻结表示和 128-token Transformer。新增分支末层零初始化，初始预测与 v3 严格一致。
2. 用 Motion 当前对象 GT 速度做训练期辅助监督。新增小速度头从历史 latent 估计速度，辅助 loss 更新运动表示。
3. **对象 GT 速度只作为监督标签，没有作为 predictor 输入。** predictor 推理接口仍为历史 latent、候选 action、当前自车状态、历史有效 mask。GT 速度没有被用于生成预测 panel。
4. 辅助标签在当前 ego 坐标系，按候选位姿变换并旋转到相应坐标；只监督可用对象框区域，排除 SDC 和重叠不确定区域。
5. 继续使用 1623 个训练场景/6492 窗口/19476 候选样本，固定原 107 个开发场景/428 窗口；复用已校验的冻结 latent cache。

## 固定开发集

| 版本 | 占用 union IoU |
|---|---:|
| v3 扩充数据 | 0.4827 |
| v4 宽运动分支 | 0.5211 |
| v4 宽分支 + 速度辅助监督 | **0.5253** |

宽分支从 v3 最佳模型开始，AdamW 重置，lr=1e-4；最多 24000 新 step，每 600 step 验证，IoU 最小提升 0.001、patience=10。实际最佳 step=12000，step=18000 早停。
速度辅助版从宽分支最佳模型开始，辅助 loss 权重=0.5（速度按 15m/s 归一化），lr=1e-4，最多 18000 新 step；最佳 step=3000，step=9000 早停。
所有 step 均为各续训阶段的新增计数。没有改变占用阈值 0.5。

## 新的官方 validation 子集

跳过前 256 个已使用的合格场景，取后续 128 个：512 窗口、1536 候选目标。
先依据开发集写入 `selected_model.json`，再评估；选中速度辅助版。

| 方法 | 占用 IoU |
|---|---:|
| WM | **0.5352** |
| 复制当前 | 0.1673 |
| ego-warp | 0.4216 |
| GT 速度 CV，直接栅格化 | **0.8537** |
| CV 经过同一个冻结 encoder/decoder | 0.7730 |
| 真实未来编码重建 | 0.8638 |
| 替换为另一候选 action（推理消融） | 0.0896 |
| 只保留当前帧（推理消融） | 0.4037 |

`WM−ego-warp` pooled union IoU 的场景 bootstrap 95% CI=[+0.0873,+0.1391]。
这里是新的一份子集，不能把 v3 的另一份官方子集 0.4677 与本表 0.5352 当作配对差值。固定开发集的提升才可直接对照。

CV 重建诊断使用 CV 占用和变换后的当前地图，没有使用真实未来地图。它用于量化同一表示链路造成的部分损失。
即使考虑该损失，WM=0.5352 仍明显落后，不能归咎于 decoder 后就宣称过关。

GT 速度 CV 是重要强参考，但不是完全同信息输入的严格门槛。还应建立无 GT 速度、同历史观测条件的运动外推对照。本轮没有把 GT 速度添加到 WM 推理输入以追分。

## 更直观的 BEV 图

新增 `visualize_latent_bev.py`，将预测 latent 解码成地图和分类对象占用，以米为坐标，前方朝上、左侧朝左，标出自车位置，并加入中文图例。
六列分别为：当前标注、真实未来标注、WM 预测、静态变换、GT 速度 CV、真实未来压缩重建。
灰色为静态变换地图的越界区域；地图背景不代表已确认 free/drivable space。
这是由 Motion 地图和对象框构成的语义 BEV，不是未来相机照片或真实三维 LiDAR 体素。

产物：

- 最新语义图：`outputs/world_model/motion_latent_v4/semantic_bev/bev_00.png` 到 `bev_05.png`；`index.json` 记录场景、anchor、checkpoint SHA256。
- v3 对照语义图：`outputs/world_model/motion_latent_v4/semantic_bev_v3/`。
- 指标：`selected_evaluation/metrics.json`、`per_scene.json`。
- 模型：`velocity_aux/best.pt`；训练源码快照与配置：`velocity_aux/source/`、`config.json`。
- 当前速度辅助监督的构建脚本：`build_motion_velocity_targets.py`，sidecar JSON 记录目标用途和数据 hash。

单元测试 31 个通过，包括：升级分支保留预训练预测、可训练梯度、90 度旋转下速度标签方向，以及辅助头不需要 GT 速度作为 forward 输入。

后续主要问题仍是移动对象位置预测。当前仅 1s Motion 框占用任务，未接 QD latent、相机迁移、完整 3D occupancy 或 RFS。
