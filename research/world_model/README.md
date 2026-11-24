# 世界模型实验索引

项目入口按预测任务分组。占用 IoU 只有在输入、网格、测试集和阈值一致时才能作严格对比。

| 要找的结果 | 指标与范围 | 报告 / 文件 |
|---|---|---|
| **显式对象运动 V8，1 秒** | 独立集 128 场景，框占用 IoU **0.8902**（两种子均值），同集 GT 速度匀速基线 0.8492 | [中文实验报告](INSTANCE_MOTION_V8.md) · [独立集原始指标](../../outputs/world_model/instance_motion_v8_20261119/official_evaluation/metrics.json) · [选定权重](../../outputs/world_model/instance_motion_v8_20261119/forecast_dense/seed43/best.pt) |
| **共享递推 V9，0.5、1、2、3、5 秒** | 同一独立集上框占用 IoU **0.9325 / 0.8742 / 0.7110 / 0.5627 / 0.4009**（两种子均值） | [原始指标](../../outputs/world_model/instance_rollout_v9_20261119/official_evaluation/metrics.json) · [选定模型](../../outputs/world_model/instance_rollout_v9_20261119/selected_model.json) · [五时刻图](../../outputs/world_model/instance_rollout_v9_20261119/official_horizon_comparison.png) |
| **经典 latent V4，1 秒** | 固定开发集 0.5253，独立验证子集 0.5352 | [训练与验证](LATENT_MOTION_V4.md) · [权重](../../outputs/world_model/motion_latent_v4/velocity_aux/best.pt) |
| **V7 运动传播，1 秒** | 同批场景上纯传播原生 0.5598，经 latent 再解码 0.5484，V4 为 0.5288 | [同场景对照](MOTION_TRANSPORT_V7.md) |
| **早期 Motion GT 语义监督** | 同时监督地图、框占用和框。1 秒固定阈值测试 IoU，优化共享版 0.354，独立模型 0.362 | [占用优化](../../outputs/world_model/motion_gt_v1/OCCUPANCY_OPTIMIZATION.md) · [框优化](../../outputs/world_model/motion_gt_v1/BOX_OPTIMIZATION.md) · [基础测试指标](../../outputs/world_model/motion_gt_v1/VALIDATION_METRICS.md) |
| **QD 检测框 → V9 → 轨迹排序** | 同批 200 条记录，当前 BEV 8.6848 RFS，加入预测未来 8.6776，原规划基线 8.6782。该实验未观察到选轨收益 | [RM 指标](../../outputs/selector_instance_v9_20261119/reward_model/metrics.json) · [配对差值及区间](../../outputs/selector_instance_v9_20261119/reward_model/paired_comparisons.json) |

V8/V9 的高分来自 **Waymo Motion 精确历史标注框输入、未来对象状态监督、二维框占用评测**。这项指标衡量物体足迹，地图和三维占用属于另外的预测任务。QD 轨迹排序使用的是感知头预测框，结果在表中单列。

## 旧实验与复现依赖

- [旧版运行手册](LEGACY_RUNBOOK.md)收录早期原型、语义监督、QD 训练与评测命令。[V2](LATENT_MOTION_V2.md)、[V3](LATENT_MOTION_V3.md)、[V5](LATENT_MOTION_V5.md)、[V6](ABLATION_V6.md)保留各自的实验记录。
- `outputs/world_model/motion_latent_v2/dataset/` 定义 V8/V9 共同使用的固定开发集。训练脚本通过场景与锚点读取它。
- `outputs/world_model/motion_latent_v3/dataset2048/` 提供 V8/V9 的窗口、候选和训练划分。`motion_latent_v3/expanded2048/best.pt` 另用于 V3 对照与部分旧图的复现。
- `motion_gt_v1` 属于独立的语义监督实验。它的命名与 latent 迭代版本不是同一条序列，部分早期适配实验还引用其权重。
- `outputs/` 由 Git 忽略。表中的本地指标、权重和图不随源码仓库分发。

训练与评测源码在 [src/qwen_drive_world_model](src/qwen_drive_world_model)，检查在 [tests](tests)。
