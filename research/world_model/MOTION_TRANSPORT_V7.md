# V7 显式运动传播

6次正式训练、48项检查及新128场景确认已完成。详见[实验报告](../../outputs/world_model/motion_transport_v7_native_20261119/README.md)。

纯传播新场景原生IoU0.5598，latent再解码0.5484；原V4为0.5288。传播加残差的3候选1秒端点代理后悔值为0.00743，原V4为0.01627。原V4保留，未宣称RFS或QD域提升。

训练：`src/qwen_drive_world_model/train_motion_transport.py`；模型：`motion_transport.py`；诊断：`motion_instance_diagnostics.py`；确认：`confirm_motion_transport.py`。
