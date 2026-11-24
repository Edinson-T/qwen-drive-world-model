# Anchor-conditioned 轨迹生成方法详解

> 文档版本：2026-11-09  
> 本次更新补充了 QD-SFT、QD-RL、KM、SKM 和 RL+anchor 在完整
> 478 个 WOD-E2E validation 场景上的统一 `N=20` 评测，并详细解释
> RFS、ADE、DiffusionDriveV2 Div、Mean candidate RFS、Oracle RFS 和
> RFS gap 的计算方法。之后又补做了 SFT direct+anchor，与 RL
> direct+anchor 使用完全相同的 anchor bank、偏移比例和评测协议。
> 本轮评测没有修改 Qwen-Drive 的原始 `src/`、
> `scripts/`、`docs/` 或模型权重。

本文描述当前代码真正使用的生成器，不包含已经删除的 WTA、plain DD、
recursive endpoint 和 absolute-anchor residual 等失败路线。

## 1. 自动驾驶模型要输出什么

模型在当前时刻接收：

- 3 个前向相机的多帧图像；
- 自车过去约 1.5 秒的位姿、速度和加速度；
- 导航指令，例如直行、左转或右转；
- 当前 ego 状态。

它需要输出自车未来 5 秒的轨迹。轨迹由 50 个 waypoint 组成，频率为
10 Hz。每个 waypoint 包含：

```text
x：向前行驶的距离
y：向左或向右偏离中心的距离
heading：车辆朝向
```

所有轨迹都表达在当前 ego 坐标系中：

```text
(0, 0) = 当前车辆位置
+x     = 车头前方
+y     = 车辆左侧
```

因此模型输出本质上是一条从原点出发、未来 5 秒会去往哪里、途中如何转弯
的三维折线。

## 2. 什么是 anchor

Anchor 可以理解成“预先总结出来的典型未来路线”。

训练集有 37,643 个未来轨迹。如果每条轨迹都单独保存，模型面对的选择太多。
因此先用 K-Means 把相似的未来轨迹合并成 20 个代表轨迹，每个代表轨迹就叫
一个 anchor。

```text
37,643 条训练轨迹
  -> K-Means 聚类
  -> 20 个典型轨迹中心
  -> 每个中心就是一个 anchor
```

Anchor bank 是在训练集上离线得到的，训练 Planner 时 anchor 本身不再移动。
模型训练的不是“anchor 坐标”，而是如何根据场景修正 anchor。

当前 anchor bank：

```text
文件：outputs/anchored_flow/anchors_k20.npz
数量：K=20
每条 anchor：50 个 waypoint
```

## 3. Anchor 是怎么聚类出来的

首先将训练轨迹按 Qwen-Drive 的尺度归一化：

```text
x_norm = x / 165 m
y_norm = y / 25 m
heading_norm = heading / 1.5703125 rad
```

然后把一条 `50 × 3` 的轨迹展开成一个长度为 150 的向量，对 37,643 个向量做
确定性 K-Means++。训练完成后，再把 20 个中心还原成物理坐标和 heading。

每个 anchor 记录它覆盖了多少训练窗口。当前 20 个 anchor 的样本数差异很大，
最少的只有 124 个，最多的超过 10,000 个。

## 4. 为什么不能直接把 anchor 当最终轨迹

全局 K-Means anchor 不知道当前场景。一个高速直行 anchor 不可能适合所有
低速十字路口，因此不能让 anchor 直接成为最终输出。

最直接的 `anchor + 小修正` 也不够好。实测直接输出原始 anchor 时：

```text
first preference ADE：9.337 m
mean preference ADE：15.985 m
```

虽然早期自定义的 ADE Headroom 很高，但大部分候选都是不可用轨迹。这叫
“用坏候选制造虚假多样性”，也是后来不再把 Headroom 作为正式论文指标的原因。

当前方法因此采用：

```text
场景基础轨迹
+ 小比例 anchor 模式差异
+ 有小幅上限的 learned residual
```

## 5. Scene base：先预测一条场景基础轨迹

模型先将 20 个 anchor 的平均轨迹作为 neutral trajectory，输入冻结的
Qwen-Drive Planner，得到一条场景基础轨迹：

```text
base = Planner(scene, neutral_trajectory)
```

这条 `base` 已经包含相机、历史、导航和 ego 状态中的场景信息。它负责保证
候选轨迹整体上适应当前场景。

Base 不是某一个 anchor 的输出，而是所有模式的共同起点。

## 6. Centered anchor：提取 anchor 的“方向差异”

首先计算 20 个 anchor 的平均轨迹：

```text
anchor_center = mean(anchor_0 ... anchor_19)
```

然后：

```text
centered_anchor_k = anchor_k - anchor_center
```

`centered_anchor_k` 不再表示绝对路线，而表示第 k 个 anchor 相对全部 anchor
平均形状的差异，例如更靠左、更靠右、更早转弯或走得更远。

## 7. 最终候选轨迹公式

第 k 条候选为：

```text
candidate_k =
    base
    + lambda * centered_anchor_k
    + delta_k(scene, anchor_k)
```

各项含义：

- `base`：共享的场景基础轨迹；
- `lambda * centered_anchor_k`：保留 anchor 提供的模式差异；
- `delta_k`：网络根据场景为第 k 个 anchor 学习的小幅修正。

`lambda` 是可训练标量，但必须位于：

```text
0.04 <= lambda <= 0.08
初始值 = 0.045
```

保留下界的原因是：如果 lambda 学到 0，20 条轨迹会重新变成同一条 base，
再次发生 mode collapse。

早期曾使用 `modes@1m` 做快速诊断。它的规则是：只取每条轨迹 5 秒后的
终点，如果新终点与已有代表终点的距离小于 1 米，就认为属于同一个模式。
这个规则是本项目自己定义的工程诊断量，不是 WOD-E2E 官方指标，也不是
DiffusionDrive 的 diversity 指标，而且结果会受候选顺序影响。因此现在的
正式评测不再用它判断方法好坏，而改用论文中有明确定义的 DiffusionDriveV2
`Div` 指标。

历史上 lambda 与自定义 endpoint modes 的扫描结果只保留作参数背景：

```text
lambda 0.035 -> 平均 4 个端点模式
lambda 0.040 -> 平均 5 个端点模式
lambda 0.045 -> 平均 5 个端点模式
lambda 0.060 -> 平均 7 个端点模式
```

当前训练中 lambda 基本保持在 0.045。lambda 的主要作用是防止 anchor 模式
差异在训练中收缩为零；最终多样性以第 16 节定义的 DDv2 `Div` 为准。

## 8. Learned residual 的物理限制

`delta_k` 由一个零初始化的线性头预测。训练开始时：

```text
delta_k = 0
```

所以模型一开始不会破坏 anchor 先验。

Residual 被限制为：

```text
每个 waypoint 的 XY 修正 <= 12 m
每个 waypoint 的 heading 修正 <= 0.35 rad
```

限制使用平滑的 tanh，不是直接截断，因此超过上限后仍保留梯度。

## 9. Anchor identity 如何进入 Transformer

完整架构图：

```text
outputs/visualizations/anchored_flow/architecture/planner_base_residual_architecture.png
outputs/visualizations/anchored_flow/architecture/planner_base_residual_architecture.pdf
```

每条候选还会输入：

- `anchor_k` 的完整 waypoint 坐标；
- anchor ID 的 embedding；
- 当前时间 `t=0.9`；
- 场景 K/V；
- 历史位姿、速度、加速度；
- navigation command 和 ego status。

Anchor ID 经过 embedding 和 projection 后，以加和方式加入 adaptive layer
normalization condition：

```text
condition = time + navigation + ego + anchor_embedding
```

完整 anchor 几何则直接进入 waypoint input 和 Fourier features。

## 10. 为什么还要加 low-frequency noise

每条 anchor 会加入少量低频噪声：

```text
noise_std = 0.02
noise_modes = 6
```

它不是每个 waypoint 独立随机抖动，而是在 6 个低频余弦方向上生成平滑噪声，
避免轨迹出现锯齿，同时让不同候选不完全相同。

## 11. 训练标签如何分配

对每个训练场景，计算 GT future 与 20 个 anchor 的平均 XY 距离：

```text
error_k = mean_t ||GT_t - anchor_k,t||_2
gt_anchor_id = argmin_k error_k
```

这个 anchor 叫 GT-nearest anchor。

只有它能获得直接的 GT 轨迹回归 loss；所有 20 条候选都参与分类和排序 loss。

## 12. 总 loss

当前总 loss：

```text
total =
    trajectory_weight * (
        regression_weight * L1_positive
        + classifier_weight * focal
        + ranking_weight * ranking
    )
```

当前权重：

```text
trajectory_weight = 12
regression_weight = 8
classifier_weight = 1
ranking_weight = 5
```

因此展开为：

```text
total = 12 * (8 * L1_positive + 1 * focal + 5 * ranking)
```

### 12.1 Positive trajectory L1

```text
L1_positive = mean |candidate_(gt_anchor_id) - GT|
```

它只修正 GT-nearest anchor 对应的轨迹，使最近模式尽可能贴近真实未来。

### 12.2 Focal classification

20 条候选的 score head 输出 20 个 logit。GT-nearest anchor 是正类，其余是
负类。Focal loss 参数：

```text
alpha = 0.25
gamma = 2.0
```

Focal loss 会降低大量简单负样本的权重，使训练更关注难以区分的 anchor。

### 12.3 Candidate ranking

对每条候选计算 smooth L1 误差 `e_k`，然后用：

```text
p_k = softmax(-e_k / 0.02)
```

得到“理论最优选择分布”。再要求 score logits 学习该分布：

```text
ranking = -sum_k p_k * log_softmax(score)_k
```

它试图让 score head 直接排序候选质量，而不是只分类 nearest anchor。

### 12.4 为什么总 loss 看起来一直在 180

`ranking` 的随机基线大约为：

```text
ln(20) = 2.9957
```

仅 ranking 对总 loss 的贡献就是：

```text
12 * 5 * 2.9957 = 179.74
```

因此总 loss 看起来一直在 180 附近。真正应观察的是分项：

- positive trajectory L1；
- focal loss；
- ranking loss；
- first/median ADE；
- DDv2 Div、Mean candidate RFS 和 RFS gap。

实际训练中 ranking loss 长期保持在 2.99，说明它没有学会候选质量排序。最终
selector 应由 WoTE 的 future-BEV reward model 替代。

## 13. 训练配置

当前全量 run：

```text
训练窗口：37,643
epoch：1
gradient_accumulation：8
optimizer steps：4,705
learning rate：1e-5
AdamW weight decay：1e-4
gradient clip：1.0
LoRA rank：16
LoRA alpha：32
```

VLM 和原始 Planner 冻结，主训练对象为：

- Planner LoRA；
- anchor embedding 和 score head；
- scene-base residual head；
- mode offset scale。

## 14. 当前保留的 checkpoint 和候选集

训练每 500 steps 保存一次 checkpoint。48-scene sweep 后，KM 的 step 500
生成器指标最好；完整 478-scene 也确认 step 500 略优于 step 4705。

当前主要实验对象：

| 名称 | 含义 |
| --- | --- |
| QD-SFT-reasoning | 原始 Qwen-Drive SFT Planner，启用 reasoning |
| QD-RL-reasoning | 原始 Qwen-Drive RL Planner，启用 reasoning |
| KM-step500 | 普通全局 K-Means anchor + BaseResidual，step 500 |
| SKM-step250 | Stratified K-Means anchor + BaseResidual，step 250 |
| SFT direct+anchor | 冻结 SFT base，直接叠加 K=20 centered anchor，不再跑 Planner flow |
| RL direct+anchor | 冻结 RL base，直接叠加 K=20 centered anchor，不再跑 Planner flow |

冻结的已训练 checkpoint：

```text
KM:
outputs/anchored_flow/k20_base_residual_full1/checkpoints/step_000500

SKM:
outputs/anchored_flow/k20_sft_skm_pilot1000/checkpoints/step_000250
```

两个 direct+anchor 方法都没有训练新的 residual head。它们分别使用冻结的
SFT 或 RL 单轨迹作为 base，再按照相同公式生成候选：

```text
candidate_k =
    base
    + 0.045 * centered_anchor_k
```

因为候选差异是在 Planner 之后直接加入的，所以不再经过 Planner 重写。这样
可以公平隔离 SFT base 和 RL base 的影响。

## 15. 统一评测协议

之前不同实验存在三类不一致：

1. QD 有 1、6、16 等不同候选数量；
2. QD-SFT 有的结果使用 direct planning，有的使用 reasoning planning；
3. 部分结果使用早期自定义指标，难以和 DiffusionDrive 比较。

本次正式结果统一为：

```text
验证场景：完整 478 个 WOD-E2E validation scenes
候选数量：N=20
规划模式：QD 使用 reasoning_planning
时间网格：Waymo 官方 5 秒、4 Hz、20 个位置
```

仓库 `docs/evaluation.md` 对“reported numbers 使用 normalized grid 还是
absolute 4 Hz grid”的表述与发布数值存在冲突。实测发现论文中的 QD
`RFS=7.95/8.45` 和 `ADE=2.31/1.27` 对应 absolute 4 Hz 网格，因此本文件
和统一评测脚本采用 absolute 4 Hz 网格，保证结果可复现论文数值。

Qwen-Drive 的 `N=20` 候选文件由原始推理脚本重新生成：

```text
outputs/waymo_val/sft_20sample_reasoning_fullval.jsonl
outputs/waymo_val/rl_20sample_reasoning_fullval.jsonl
```

注意：table 中的 QD 实际 `RFS/ADE` 使用单独一次 reasoning 采样的成绩，
因为 Qwen-Drive 论文中的 WOD-E2E 数值是可部署的单轨迹结果。20 条候选只用于
计算多样性和 Oracle 上限，不能把 20 条中的最好轨迹冒充可部署成绩。

各方法代表“实际会执行的轨迹”的选择方式：

| 方法 | 实际轨迹如何选择 |
| --- | --- |
| QD-SFT-reasoning | 单独一次 reasoning 采样 |
| QD-RL-reasoning | 单独一次 reasoning 采样 |
| KM-step500 | 20 个 candidate scores 中取最大 |
| SKM-step250 | 20 个 candidate scores 中取最大 |
| SFT direct+anchor | 没有 learned selector，取第一条候选 |
| RL direct+anchor | 没有 learned selector，取第一条候选 |

统一评测代码：

```text
research/anchored_flow/src/qwen_drive_anchored/evaluate_paper_metrics.py
```

完整报告：

```text
outputs/anchored_flow/paper_metrics_n20_reasoning_2026-11-08/summary.md
outputs/anchored_flow/paper_metrics_n20_reasoning_2026-11-08/per_scene.csv

outputs/anchored_flow/paper_metrics_sft_vs_rl_direct_anchor_2026-11-09/summary.md
outputs/anchored_flow/paper_metrics_sft_vs_rl_direct_anchor_2026-11-09/per_scene.csv
```

## 16. 每个指标是什么意思，具体怎么算

### 16.1 RFS：最贴近最终驾驶偏好的指标

RFS 的全称是 **Rater Feedback Score**，范围是 0 到 10，越高越好。它不是简单
比较预测轨迹和真实驾驶员轨迹，因为同一个场景可能存在几条都合理的开法。
Waymo 让人类评分员给多条候选轨迹打分，然后检查模型轨迹是否落在某条合理
轨迹附近。

计算过程：

```text
1. 将模型 50 个 waypoint 的 10 Hz 轨迹
   重采样为 20 个 waypoint 的 4 Hz 轨迹；
2. 在 3 秒和 5 秒两个时刻，分别计算模型轨迹相对人类偏好轨迹的
   横向和纵向偏差；
3. 用当前车速缩放允许误差；
4. 如果模型轨迹落入某条人类偏好轨迹的 trust region，
   就认为它实现了一个合理驾驶行为；
5. 从所有合理匹配中取得分，并考虑偏差越大分数衰减越多；
6. 完全不落入任何 trust region 时，最低分压到 4 分；
7. 对 478 个场景的 RFS 求平均。
```

公式上可以写成：

```text
RFS_scene = official_rater_feedback(
    prediction,
    human_preference_trajectories,
    human_preference_scores,
    initial_speed
)

RFS = mean(RFS_scene over all validation scenes)
```

官方工具在 3 秒和 5 秒使用的速度相关阈值：

```text
基础阈值：[1.0 m, 1.8 m]
横向阈值乘数：1
纵向阈值乘数：4
最终阈值还会根据初始车速在 0.5 到 1.0 倍之间缩放
```

通俗地说，车辆低速时可以容忍的偏差更小，高速时容忍范围会适当放大。与真实
行驶轨迹完全一致并不是必要条件；只要模型选择的是人类认为合理的另一种开法，
RFS 仍然可以较高。

### 16.2 ADE：轨迹和真实未来有多接近

ADE 的全称是 **Average Displacement Error**，单位是米，越低越好。

对一条被选中的预测轨迹和一个真实未来轨迹：

```text
ADE =
  mean over all waypoints
  sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)
```

例如轨迹共有 20 个评测点，就先计算每个点的二维距离，再求平均。

RFS 衡量“是否像人类会做出的合理选择”，ADE 衡量“是否贴近实际记录下来的
具体轨迹”。一个模型可以因为选择了另一条同样安全、合理的道路而 RFS 较高，
同时 ADE 不低。WOD-E2E 以 RFS 为主，ADE 作为轨迹精度和 tie-breaker。

### 16.3 DDv2 Div：候选轨迹到底有多不一样

`Div` 是 DiffusionDriveV2 定义的轨迹多样性指标。它不是数终点模式，也不是
简单的候选数量，而是比较所有候选在每个未来时刻是否真的拉开了距离。

对某个场景、某个未来时刻 `t`：

```text
Div_raw(t) =
  所有候选两两之间二维距离的平均值

scale(t) =
  所有候选 waypoint 到 ego 原点距离的平均值

Div(t) =
  min(1, Div_raw(t) / (scale(t) + 1e-6))
```

然后：

```text
Div_scene = 100% * mean(Div(t) over all waypoints)
Div = mean(Div_scene over all validation scenes)
```

通俗理解：

- 20 条轨迹几乎重合时，候选之间距离接近 0，Div 接近 0%；
- 20 条轨迹分成直行、左转、右转和不同速度时，Div 明显增大；
- Div 高只表示“候选不一样”，不表示每条都合理；
- 故意生成一条冲出道路的极端轨迹也能提高 Div，因此必须同时报告 RFS、
  ADE 和 Oracle RFS。

所有横向比较必须使用相同候选数量。本项目正式表统一使用 `N=20`，因此
QD、KM、SKM 和 RL+anchor 的 Div 可以直接放在同一张表里。

### 16.4 Mean candidate RFS：20 条候选平均有多好

先对每条候选分别运行官方 RFS，再对 20 条求平均：

```text
Mean candidate RFS =
  mean(RFS of candidate 1 ... candidate 20)
```

它不表示最终会选哪条轨迹，只表示整个候选集合的平均质量。一个模型可能发现
一条特别好的轨迹，但另外 19 条都很差，这时 Mean candidate RFS 会下降。

### 16.5 Oracle RFS：候选集合的理论上限

对每个场景，分别计算 20 条候选的官方 RFS，然后挑 RFS 最大的一条：

```text
Oracle RFS_scene =
  max(RFS_candidate_1 ... RFS_candidate_20)

Oracle RFS =
  mean(Oracle RFS_scene over all validation scenes)
```

Oracle RFS 表示“如果已经知道哪条候选最好，理论上能达到多少分”。它使用了
validation rater label，因此**严重数据泄漏，只能做上限分析，绝不能作为部署
成绩**。

### 16.6 RFS gap：生成器和 selector 之间还差多少

```text
RFS gap = Oracle RFS - 实际选中轨迹的 RFS
```

它越低越好，直接使用 RFS 分数单位：

| 现象 | 含义 |
| --- | --- |
| RFS gap 很小 | 实际选择已经接近候选集合上限 |
| RFS gap 很大但 Oracle 高 | 候选里有好轨迹，主要是 selector 不会选 |
| RFS gap 小但 Oracle 也低 | 候选集合本身缺少足够好的轨迹 |

这比早期的自定义 `Headroom` 更直接，因为它完全使用官方 RFS，而不是用 ADE
作为 RFS 的代理。

## 17. 完整 478 场景的统一结果

| Method | N | RFS↑ | ADE↓ | DDv2 Div↑ | Mean candidate RFS↑ | Oracle RFS↑ | RFS gap↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-SFT-reasoning | 20 | 7.9548 | 2.3167 | 3.96% | 7.9615 | 8.0928 | 0.1380 |
| SFT direct+anchor | 20 | 7.7928 | 2.3657 | 27.33% | 7.7210 | 8.7703 | 0.9775 |
| QD-RL-reasoning | 20 | 8.4432 | 1.2807 | 3.25% | 8.4510 | 8.5433 | 0.1001 |
| KM-step500 | 20 | 7.4854 | 2.5139 | 20.81% | 7.6236 | 8.6552 | 1.1698 |
| SKM-step250 | 20 | 7.5319 | 2.6305 | 21.85% | 7.6686 | 8.6606 | 1.1287 |
| RL direct+anchor | 20 | 8.2372 | 1.4260 | 27.43% | 8.1610 | 8.9623 | 0.7251 |

必须注意：

- QD 的 `RFS/ADE` 是单条可部署推理结果；
- QD 的 `Div/Mean/Oracle` 使用同样的模型、reasoning 模式，但来自 20 条
  独立采样的诊断候选集；
- 因为 QD 的实际单轨迹和 20 候选诊断集来自两次采样，QD 的 RFS gap 除了
  selector 差距外，还包含少量随机采样差异；KM、SKM 和 RL+anchor 的 gap
  则是在同一个候选集内部计算的；
- QD-RL 在 validation 上使用过 RFS reward，因此它的 validation 分数是
  in-sample，不代表在未见场景上的泛化；
- KM/SKM 的 `RFS` 是 candidate score head 的 argmax 结果；
- SFT/RL direct+anchor 都没有 selector，当前实际 RFS 取第一条候选；
- `Oracle RFS` 和 `RFS gap` 都使用 validation 标签，只能用于分析。

## 18. 这些结果说明了什么

### 18.1 原版 QD 的候选多样性确实很低

QD-SFT 和 QD-RL 的 Div 分别只有 `3.96%` 和 `3.25%`。这表示 20 次独立
采样虽然每次噪声不同，但最终学到的未来路线仍高度接近。

QD-RL 的 RFS 比 QD-SFT 高 `0.4884`，ADE 也明显更低，说明 RL 提升了轨迹
质量和人类偏好匹配度；但 RL 并没有解决 mode collapse，反而使 Div 更低。

### 18.2 SFT direct+anchor 证明差异来自 Planner 更新

SFT direct+anchor 的 Div 是 `27.33%`，RL direct+anchor 是 `27.43%`。两者
候选的逐时刻两两距离几乎完全相同，平均差只有约 `8 x 10^-17 m`。这是因为
两者使用同样的 anchor bank，并在同一个 base 后直接加入相同偏移。

这说明 SFT base 本身完全能保留 anchor 多样性。此前 SFT anchor-init（A2）
多样性较低，是因为 20 条候选又被 SFT Planner 从 `t=0.8` 更新到 `1.0`，
网络把不同起点重新拉向相似结果，而不是 anchor 本身不够分开。

SFT direct+anchor 的弱点在轨迹质量：

```text
SFT direct+anchor Oracle RFS：8.7703
RL  direct+anchor Oracle RFS：8.9623
```

两者的实际 RFS 都低于各自单轨迹基线，因为当前只是取第一条候选，没有 selector。

### 18.3 KM 和 SKM 解决了“候选不一样”，但没有解决“选得对”

KM 和 SKM 把 Div 提高到 `20.81%` 和 `21.85%`，说明 anchor 先验确实阻止了
所有候选重新坍缩到同一条轨迹。

但两者的实际 RFS 只有 `7.4854` 和 `7.5319`，甚至低于原版 QD-SFT 的
`7.9548`。同时 RFS gap 分别高达 `1.1698` 和 `1.1287`。

通俗地说：

```text
候选集合里确实已经开始包含不同走法，
但当前 score head 经常把坏候选选成最终输出。
```

SKM 相比 KM：

```text
RFS：+0.0465
Div：+1.04 个百分点
ADE：变差 0.1166 m
Oracle RFS：+0.0054
```

这是一个轻微优势，不足以单独证明 SKM 已经取得决定性提升。SKM 的主要价值是
用导航命令、速度分层和真实 medoid 改善长尾覆盖；完整 anchor 消融应放到
WM 主线之后。

### 18.4 RL direct+anchor 的候选覆盖最好，但缺少可靠 selector

RL+anchor 的 Div 为 `27.43%`，大约是 QD-RL 的 8.4 倍。Oracle RFS 达到
`8.9623`，说明 20 条候选中确实存在非常符合人类偏好的轨迹。

然而当前实际只取第一条候选，RFS 是 `8.2372`，反而低于单条 QD-RL 的
`8.4432`。RFS gap 为 `0.7251`。

这说明：

```text
生成器已经提供了高价值候选；
当前核心瓶颈是 candidate selection；
把 Oracle 上限转化成 top-1 收益，需要一个真正理解未来场景的选择器。
```

## 19. Flow 时间区间和实际推理速度

不同实验的 flow time 起点不同，不能把它们都叫成“从 t=0.8 开始”或
“从 t=0.9 开始”：

| 方法 | Flow 预测时间 | 候选侧 flow 步数 |
| --- | --- | ---: |
| 原始 QD | 在 `0.0, 0.1, ..., 0.9` 调用网络，最终更新到 `1.0` | 10 |
| SFT anchor init（A2） | `0.8 -> 0.9 -> 1.0` | 2 |
| 当前 BaseResidual | base 查询和候选查询都在 `t=0.9`，没有显式 Euler 迭代 | 1 次候选 endpoint 查询 |
| Direct base+anchor | 候选不再跑 flow，直接做向量加法 | 0 |

从名义 flow 步数看：

```text
原始 QD：10
A2 anchor init：2
BaseResidual：1 次 t=0.9 候选 endpoint 查询
Direct anchor：0 个额外候选 flow step
```

BaseResidual 不是从 `t=0.8` 开始，也没有执行两步 Euler 更新。它先在
`t=0.9` 预测 scene base，再用同一时间条件为 20 条候选做一次批量查询，最后
由 residual head 直接输出最终轨迹。因此网络实际有 base 和 candidate 两次
前向调用，但只有一次候选 endpoint 查询。

因此 BaseResidual 的候选 flow 比原版 QD 少约 10 倍，A2 少 2 倍。但总延迟
不会按同样倍数下降，因为相机编码、VLM、reasoning 和数据预处理占很大比例。
BaseResidual 还会额外计算一次 scene base，所以网络调用次数不等于表面的
flow 步数。

`Direct base+anchor` 本身没有自己的 `t` 区间。它的 base 通常来自已经用原始
QD 或冻结 Planner 生成的单条轨迹；之后只执行：

```text
candidate_k = base + 0.045 * centered_anchor_k
```

所以 RL direct+anchor 的完整表示应当是：

```text
RL base：原始 QD flow，t=0.0 ... 0.9 共 10 步，最终到 1.0
anchor expansion：无 flow，只做向量加法
```

不能把它写成 `t=0.9 -> 1.0`，也不能写成从 `t=0.8` 开始的两步推理。

在同一块 GPU、同一批 16 个 RL reasoning 场景上，使用原始 QD 10-step 推理：

| 候选数 | 总时间 | 每场景时间 |
| ---: | ---: | ---: |
| 1 | 13.8 s | 0.86 s |
| 5 | 15.7 s | 0.98 s |
| 20 | 24.9 s | 1.55 s |

结果是：

- `1 -> 5`：每场景增加约 14%；
- `5 -> 20`：增加约 58%；
- `1 -> 20`：增加约 80%，速度约为原来的 `1/1.8`。

VLM 每场景只运行一次，expert 候选是一批并行生成，所以耗时不会按候选数线性
增加。但候选数仍会增加 expert 计算、显存和批处理压力。

Direct anchor 的候选扩展几乎不花 GPU：在 478 个场景上读取、生成、写出
20 候选文件约需 3.2 秒，包含文件 I/O 后平均约 6.7 ms/scene。在线系统中，
它相当于先生成一条 QD base，再做一次很小的向量运算。

后续 BEV WM 的情况不同。如果每条候选都要单独预测未来 BEV，WM 成本会随
候选数明显增加；如果共享场景编码并批量处理候选，增加会更小。因此
`N=20 -> N=5` 的压缩是否值得，必须同时测量 Oracle RFS 损失和 WM 延迟。

## 20. 最终轨迹选择路径

当前 simple score head 只作为 control，不是论文最终 selector。

论文主线应保持：

```text
20 candidate trajectories
  -> action-conditioned BEV world model
  -> 为每条候选预测未来 BEV
  -> reward model 根据未来 BEV 打分
  -> argmax reward
  -> final trajectory
```

WM/RM 必须做对照实验，不能只证明“它能选”，还要证明“未来 BEV 是必要的”：

| 对照 | 目的 |
| --- | --- |
| random selection | 最低基线 |
| fixed first candidate | 无选择器基线 |
| trajectory-only learned selector | 不使用未来 BEV |
| reward without future BEV | 检查只看当前状态是否足够 |
| action-conditioned BEV WM + reward | 完整方法 |

如果完整方法没有优于 trajectory-only 或 no-future-BEV reward，就不能声称
world model 对最终选择有贡献。

## 21. 当前结论和下一步

当前已经证明：

- 原版 QD 在 WOD-E2E 上存在明显 mode collapse；
- anchor-conditioned BaseResidual 能显著提高候选多样性；
- SFT/RL direct+anchor 证明差异来自 Planner 后续更新，而不是 anchor 本身；
- RL direct+anchor 在固定 20 候选上提供最高的 Oracle RFS；
- KM/SKM 的 simple score head 无法把候选覆盖转化成可信的 top-1 结果；
- SFT/RL direct+anchor 也都缺少 learned selector，实际成绩不能代表候选上限。

因此下一步不是继续增加 generator 训练步数，也不是继续调 lambda，而是：

1. 冻结当前候选生成器和 `N=20` 评测协议；
2. 实现 action-conditioned BEV world model；
3. 用未来 BEV 训练 reward model；
4. 与 random、first、trajectory-only selector、无未来 BEV reward 做对照；
5. 只有当 WM+RM 的 top-1 RFS 明显超过基线时，再继续做更多 anchor 数量、
   KM/SKM 和 SFT/RL 的系统消融。
