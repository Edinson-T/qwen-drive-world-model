# V6 受控消融

9配置×2种子及32窗口拟合诊断已完成，报告见[结果与原因判断](../../outputs/world_model/ablation_v6_20261119/README.md)。

冻结E/D可拟合到0.945；更换latent损失未改善；解冻D对续训对照有小幅收益，但未显著超过原V4。原V4保留。

入口：`src/qwen_drive_world_model/ablate_motion_training.py`；确认：`confirm_motion_ablations.py`；同输入基线：`raster_tracking_baseline.py`。
