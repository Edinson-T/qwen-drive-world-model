# V8：实例状态预测超过 GT 速度 CV（2026-11-19）

本轮找到一个超过指定基线的方法，但必须准确命名：这是新的**实例状态/token 预测器 + 解析 BEV 栅格化**，不是原 V4 的四帧裁剪 BEV latent 模型已经达到 0.89。旧模型和权重保留。

## 结果与评测口径

预测未来 1 秒的标注框二维占用；50×100、0.6m 网格，三条候选 ego 轨迹，沿用缓存中的 target_occ 和 cv_occ。不是三维完整占用、自由空间或可见性指标。

| 数据 | GT 速度 CV | seed43 | seed44 | 两次训练 IoU 均值 |
|---|---:|---:|---:|---:|
| 固定开发集：107 场景、428 窗口 | 0.84346 | 0.89563 | 0.89286 | 0.89424 |
| 新取官方 validation：128 场景、512 窗口 | 0.84923 | 0.89095 | 0.88941 | 0.89018 |

这里的均值是两次独立模型评测值的算术均值，不是集成预测。官方验证子集未参与 checkpoint 选择；其场景 ID 与整个训练/开发缓存核对无交集。按开发集选定 seed43 第4000步、seed44 第3500步并保存 SHA 后，才进行该独立评测。

独立集相对 CV 增加 0.04095（4.10 个百分点）；按场景配对 bootstrap 的95%区间为 [0.03263, 0.05042]。此区间描述场景采样不确定性，条件是这两个已训练种子，不代表充分覆盖训练随机性。

同一独立集：历史估速 CV 0.82374，历史加速度/转向率 0.85666，历史转向积分加纵向加速度 0.86558。代码中的 history_ctrv 实际包含纵向加速度，不能称为纯 CTRV。

## 改了什么，以及为什么有效

旧路线先把框栅格化、压缩为 latent，再预测未来 latent。新路线保留每个物体的精确位置、长宽、朝向和类别，先进行历史关联，再预测未来框，最后画成同一口径的 BEV。这样避免在压缩栅格里重新发现对象身份、速度和转向。

1. 用过去1.5秒、10Hz共16帧检测框做几何 Hungarian 关联，不使用 GT 对象 ID 或 GT 速度；关联后抽取 -1.5、-1、-0.5、0 秒四个状态。
2. 从历史位置估算速度、加速度、转向率，构造5个运动候选。
3. 小型 MLP（43→128→128→11，23563参数）学习候选权重、位置残差、朝向/尺寸修正和未来存在概率。
4. 预测未来实例状态，经0.15m多边形绘制并最大池化到0.6m，得到评测用 BEV。没有额外训练 AE 编码器/解码器。

**输入条件发生了变化。** 使用 Motion 的精确 GT 历史框，以及当前所有有效源对象；原 V4 只有四帧裁剪栅格 latent。新方案的历史关联还用了16个检测时刻。GT 速度 CV 本身也拥有完整当前对象列表。这里将源对象范围对齐基线，不能宣传为原输入接口不变时的提升。

未来标签按 GT ID 分配，这是训练监督；预测器特征和几何关联均不读取数值 ID、GT 速度。纯关联推理函数只接受框和类别，另有标签包装用于核对关联准确率。保存的 GT 速度只供特权基线计算。

## 训练数据、预算与 Loss

父缓存2048场景8192窗口；实际 SGD 使用1623训练场景6492窗口、225699个对象窗口样本。固定开发集107场景428窗口，其余缓存保留数据没有加入 SGD。

两个随机种子43、44，各6000次更新；每步随机有放回抽取2048对象样本，AdamW学习率3e-4、weight_decay=1e-4、梯度裁剪5，每500步验证。每次完整运行约54.4个对象样本遍历当量（有放回抽样，不是严格 epoch）。开发集选中步分别约36.3和31.8遍历当量。

Loss 针对未来对象状态，而不是原 latent 或像素 BCE：

`L = area_weighted_mean(L_center + 0.25*(1-cos(2*heading_error)) + 0.25*L_log_size) + 0.1*BCE(existence)`

center 是按当前长宽（最小1m）归一化后的 Smooth L1；log_size 也是 Smooth L1。几何项只监督未来有效对象，用 sqrt(length*width) 截到[1,8]加权。未来存在项监督所有当前查询对象。

## 原因定位的证据与边界

- 只在四个稀疏检测时刻关联的版本开发集约0.824，未超过基线；使用稠密历史关联后明显改善。数据总量没有扩成数十倍，单纯继续原模型训练也不是本次成功的原因。
- 在开发集对已训练模型做推理干预：用历史 CV 中心替换学习中心，IoU 从0.89424降至0.81549；替换回当前朝向为0.89053；当前尺寸为0.89439。主要收益来自中心运动预测，尺寸修正没有稳定收益。
- 强制保留所有当前对象为0.89424，几乎不变，收益不是依赖预测“删掉难对象”。
- 限制当前源对象到原60×30m裁剪，学习模型降至0.80839，GT速度CV降至0.76342。说明输入边界与未来驶入视野对象很重要。此干预只限制源对象，没有重新计算历史关联，**不是严格复现原 V4 输入的消融**。
- 上述干预均是开发集事后分析，没有重新训练，不用于修改独立验证选择。
- 独立集同一4174个未来有效且在目标裁剪内的对象，中心误差 seed43/44为0.205/0.207m，GT速度CV为0.300m；其中当前速度≥5m/s的845个对象为0.460/0.460m，对比CV 0.733m。统计不按模型存在概率筛掉难例。
- 独立集骑行者类别 IoU 没有改善，不能声称所有类别都优于 CV。

因此已经证明：精确实例历史下，学习运动预测可以超过 GT 速度 CV；尚未证明原 V4 latent 结构能够做到。没有把问题归咎于某一个 Loss，也没有证据声称原模型存在单一确定的代码错误。

## 复现、文件与下一步接口

代码在 `research/world_model/src/qwen_drive_world_model/{build_instance_motion,instance_motion_model,train_instance_motion,evaluate_instance_motion}.py`。纯关联接口 `track_dense_boxes` 只读检测框和类别。

本轮根目录：`outputs/world_model/instance_motion_v8_20261119/`。

- `forecast_dense/protocol.json`：实际配置、数据溯源；`forecast_dense/source/`：训练源码快照。
- `forecast_dense/selection_before_official_test.json`：独立测试前固定的模型、步数和 SHA。
- `official_evaluation/metrics.json`：独立集指标、场景统计、配对区间、无交集核查。
- `official_evaluation/position_errors.json`：对象位置误差。
- `component_audit/metrics.json`：开发集推理干预。
- `official_comparison.png`、`training_curves.png`、`bev_examples.png`：指标、训练曲线、真实栅格对照；示例按固定等间距索引选择。
- `selected_model.json`：按开发集选定 seed43 的实验模型入口；不兼容原 V4 接口。

训练命令（输出目录须不存在，保留历史实验）：

```bash
PYTHONPATH=src:research/world_model/src:tools/waymo_e2e /home/fractal/miniconda3/envs/qwen-drive/bin/python -m qwen_drive_world_model.train_instance_motion \
  --instances outputs/world_model/instance_motion_v8_20261119/dataset_dense \
  --data outputs/world_model/motion_latent_v3/dataset2048 \
  --val-reference outputs/world_model/motion_latent_v2/dataset \
  --output outputs/world_model/instance_motion_v8_reproduction \
  --steps 6000 --batch-size 2048 --eval-every 500 --lr 0.0003 --seeds 43 44
```

部署到 QD 仍须提供检测框及足够频率的历史关联，并评测检测噪声。当前没有验证原12张图像约束、真实20条候选轨迹排序/RFS、5秒预测。可以将本分支作为显式运动基线或融合分支；不能把 checkpoint 直接加载到原 latent WM，或据此声称相机闭环已完成。

## 原始盘状态

用户完成只读挂载，已实际确认 `/dev/sdc2`、exfat、`ro`。新数据入口：`/home/fractal/qwen-drive/.waymo_readonly/waymo_motion_v1_3_1/{training,validation}`。旧 `/mnt/hdd` 挂载失效不是改 Python 字符串即可恢复；现已用新可读入口构建本轮缓存。原始数据没有写入。
