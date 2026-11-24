# Motion WM 第五轮

2026-11-19勘误：V5优化器状态复制共享Adam动量，已修复并重跑。请以[审计与修正结果](../../outputs/world_model/audit_20261119/diagnosis.md)为准；原V5数值仅作历史记录。

本轮训练与新增128场景评估已完成。详细结果见 [实验报告](../../outputs/world_model/motion_latent_v5/README.md)。

未来latent降低了1秒端点风险误差；联合回传相比冻结WM尚无显著额外收益，并略降占用IoU。原V4仍保留。该实验不是RFS或完整WoTE复现。
