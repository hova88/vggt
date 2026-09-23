# NuScenes 连续前视图像道路类型识别：开发记录

记录日期：2026-09-23
状态：代码与预训练 head-only smoke test 已跑通；真实监督训练等待道路标签。

## 目标与边界

输入连续四帧 `CAM_FRONT` 图像，使用 VGGT aggregator 的多帧 patch feature，输出五类道路状态的 raw logits 和概率。五类依次为 `elevated_up`、`elevated_down`、`main_road`、`side_road`、`intersection`。概率将作为后续 HMM 的视觉 observation，因此实验记录 accuracy、macro F1、NLL、Brier Score 和 ECE；当前阶段没有 lane、depth、地图或 HMM 过滤训练。

NuScenes 不提供这五类对应的真值。训练只读取外部 `road_labels.jsonl` 中有标注的 target sample；标签对应序列最后一帧。支持整数 `label` 和五维 `soft_label`，不自动推断或生成道路标签。

## 仓库核查与实现选择

- 当前 [`Aggregator`](../vggt/models/aggregator.py) 输入 `[B,T,3,H,W]`、像素范围 `[0,1]`，返回缓存层列表和 `patch_start_idx`。最后缓存层拼接 frame/global 中间特征。
- 当前 [`VGGT`](../vggt/models/vggt.py) 的 camera/depth/point/track 接口保持原样；道路任务通过独立 wrapper 只调用 aggregator。
- 特殊 token 包含一个 camera token 和四个 register token，实测 `patch_start_idx=5`。默认 head 只读后面的 patch tokens。
- 复用 [`load_and_preprocess_images`](../vggt/utils/load_fn.py)：NuScenes 原图 `1600×900` 先按 VGGT 方式缩至宽 518、高 294；小实验随后统一缩至 `280×154`，高宽均可被 patch size 14 整除。配置可将 `image_width` 改回 518。默认无翻转、随机裁剪、颜色扰动。
- 现有 [`training/`](../training/) 主要服务于几何多任务和分布式训练；本实验增加独立的小规模单机入口，复用仓库模型与预处理，不改原训练入口。

## 新增文件

| 文件 | 作用 |
| --- | --- |
| [`vggt/heads/road_probability_head.py`](../vggt/heads/road_probability_head.py) | patch 投影、每帧 ego cross attention、时间 Transformer、road query 池化、分类与概率 wrapper |
| [`training/data/nuscenes_road_dataset.py`](../training/data/nuscenes_road_dataset.py) | NuScenes version 检测、manifest 校验、同 scene 连续帧、scene 划分所需记录 |
| [`configs/road_head_nuscenes_small.yaml`](../configs/road_head_nuscenes_small.yaml) | T、stride、样本上限、冻结策略、AMP、校准等配置 |
| [`scripts/build_nuscenes_road_manifest.py`](../scripts/build_nuscenes_road_manifest.py) | 生成不含标签的人工标注模板 |
| [`scripts/train_road_head.py`](../scripts/train_road_head.py) | dataset debug、shape test、dummy smoke、tiny overfit、正式训练与 best checkpoint |
| [`scripts/eval_road_head.py`](../scripts/eval_road_head.py) | 验证指标、混淆矩阵、可靠性图、注意力图 |
| [`scripts/calibrate_road_head.py`](../scripts/calibrate_road_head.py) | 验证集单标量 temperature scaling |
| [`scripts/infer_road_head.py`](../scripts/infer_road_head.py) | 从 sample token 或有序图片推理，返回概率、熵、可选 prior correction |
| [`scripts/road_common.py`](../scripts/road_common.py) | 共享模型加载、loss、指标和可视化 |

## 实测数据与 shape

使用 `/data/nuscenes`，自动选择 `v1.0-mini`。官方 devkit 找到 10 个 scene、404 个 target sample；默认四帧、stride 1 时，30 个 target 因历史不足被跳过，剩下 374 个可用序列，缺失图像 0。仅用于 smoke test 的 scene 划分为 train 299 条 / 8 个 scene、val 75 条 / 2 个 scene。当前真实 annotation 文件缺失，404 个 target 均未标注；因此**没有真实类别分布**。

预训练 VGGT、GPU、`B=1,T=4,image_width=280` 的运行时 shape：

| 张量 | shape |
| --- | --- |
| `images` | `[1,4,3,154,280]` |
| `aggregated_tokens_list[-1]` | `[1,4,225,2048]` |
| `patch_start_idx` | `5` |
| `patch_tokens` | `[1,4,220,2048]` |
| token projection | `[1,4,220,256]` |
| frame road features | `[1,4,256]` |
| temporal features | `[1,4,256]` |
| road embedding | `[1,256]` |
| road logits / probability | `[1,5]` |

最后特征维度在运行时由 `tokens.shape[-1]` 读取并校验，没有写死 2048。参数总量 912,061,189；road head 2,948,869；head-only 可训练参数 2,948,869。

## 验证记录

1. 生成了 [`outputs/road_head/road_labels_template.jsonl`](../outputs/road_head/road_labels_template.jsonl)，包含 374 行 sample token、scene、timestamp、CAM_FRONT 路径，**没有填入标签**。
2. hard/soft manifest parser、混合 target batch、head 梯度、metrics 与图表保存均通过代码测试。
3. 使用缓存的官方 `facebook/VGGT-1B` 权重进行 GPU head-only smoke test：预训练 aggregator 前向、概率和为 1 的断言、一次 head 梯度更新均通过；峰值 PyTorch GPU allocated 约 **1861 MiB**。本地生成的 `dummy_smoke_spatial.png` 和 `dummy_smoke_temporal.png` 只验证可视化代码，不代表学到了道路关注区域；两张调试图未纳入 PR。
4. `last_blocks=2` 解冻模式已通过随机权重、CPU、`image_width=70` 的前向与一次更新；对应可训练参数 53,342,981。本机 GPU 上该模式在 AdamW 状态初始化时发生 CUDA OOM，尚无 GPU 部分 SFT 结果。
5. 正常训练命令在缺少 `/data/nuscenes/road_labels.jsonl` 时明确停止，不会保存伪造的训练 checkpoint。

因此，**tiny overfit 未运行，小规模 train/val 的 accuracy、macro F1、NLL、Brier、ECE 均未产生**。dummy 标签的类别计数和 loss 不能视为实验性能。

## 训练与概率处理

- 默认 `head_only`：冻结整个 aggregator，并在前向时禁用其梯度；冻结权重以 bf16 驻留 GPU。官方 safetensors 的 aggregator 权重逐 tensor 装载，避免整体 CPU→GPU 搬运时的内存峰值。
- 可选 `last_blocks`：只解冻最后两个 `frame_blocks` 和 `global_blocks`，保持 patch embedding 冻结，head 与 block 使用不同学习率。部分解冻 GPU 训练仍需进一步验证显存。
- hard label 使用 cross entropy；soft label 使用 soft cross entropy；混合 batch 中 hard label 转 one-hot，损失数学上等价。默认 label smoothing 为 0。
- best checkpoint 按 validation NLL 选择。验证集拟合一个正温度 `tau`，推理采用 `softmax(logits/tau)`；同时输出 entropy、normalized entropy、entropy confidence。
- 默认关闭 HMM prior correction。开启时，使用训练集 class prior 计算归一化的 `posterior / prior` 分数；这只是 pseudo emission，不是 HMM 本体。

## 复现与后续步骤

```bash
conda run -n uniad2.0 python scripts/build_nuscenes_road_manifest.py \
  --nuscenes-root /data/nuscenes \
  --output /data/nuscenes/road_labels_template.jsonl --max-samples 500
```

人工标注后，将完成文件保存为 `/data/nuscenes/road_labels.jsonl`，再依次执行：

```bash
conda run -n uniad2.0 python scripts/train_road_head.py --config configs/road_head_nuscenes_small.yaml --tiny-overfit
conda run -n uniad2.0 python scripts/train_road_head.py --config configs/road_head_nuscenes_small.yaml
conda run -n uniad2.0 python scripts/eval_road_head.py --config configs/road_head_nuscenes_small.yaml --checkpoint outputs/road_head/best.pt
conda run -n uniad2.0 python scripts/calibrate_road_head.py --config configs/road_head_nuscenes_small.yaml --checkpoint outputs/road_head/best.pt
conda run -n uniad2.0 python scripts/infer_road_head.py --config configs/road_head_nuscenes_small.yaml --checkpoint outputs/road_head/best.pt --sample-token YOUR_TARGET_SAMPLE_TOKEN
```

最优先的下一步是跨多个 scene 获取真实道路标签，先确认 16–32 条样本可被 head-only 模式明显过拟合，再运行 scene-level 验证与温度校准。小验证集上的 ECE 和温度估计需要谨慎解释。

## 2026-09-23：千问多帧标注尝试与视觉审核

新增 [`scripts/label_road_with_qwen.py`](../scripts/label_road_with_qwen.py)。它从 `~/.bashrc` 导出的 `QWEN_API_KEY` 读取密钥，通过千问 OpenAI 兼容接口将每条 target 对应的四帧图像按时间顺序提交给 `qwen3.8-flash`。输出使用 `proposed_label`，**不使用训练 parser 接受的 `label` 字段**。请求可续跑，记录模型、规则版本、图像路径、证据、理由和 token usage，不保存密钥或图像 base64。

Flash 对 374 条模板均返回有效响应，保存在 [`road_labels_qwen_v3_proposals.jsonl`](../outputs/road_head/road_labels_qwen_v3_proposals.jsonl)：主路 282、辅路 2、路口中 67、无法归入五类 23、高架上/下均 0。这里的数量是**模型提议分布，不是真实类别分布**。全部 23 条 `null` 都位于 `scene-0916` 的停车场/内部通道片段。

根据后续要求，用 `qwen3.8-max` 对这 23 条四帧序列重新判断。一次无效 JSON 响应重试后，23 条全部有效，且 **23/23 仍返回 `null`**。逐条结果见 [`qwen_flash_vs_max_null23.json`](../outputs/road_head/qwen_flash_vs_max_null23.json)，原始 Max 提议见 [`road_labels_qwen_max_null23_proposals.jsonl`](../outputs/road_head/road_labels_qwen_max_null23_proposals.jsonl)。`View Image` 抽查其中 5 条，均呈现停车场或园区内部通道；按照当前固定五类规则，保留未标注比强行塞入主路更合适。两个模型一致不等于有真实 GT，也不能计算分类准确率。

视觉审核同时发现 Flash 的路口提议存在明显误报。例如 `scene-1100` 有 32/37 条被提议为“路口中”，但抽查画面显示自车停在路口线前；官方 ego pose 显示该 scene 的 37 条 target 在约 18 秒内总位移约 1 米。另有若干 scene 的“路口中”提议只显示前方斑马线、尚未驶入路口。15 条有选择的视觉审核记录见 [`qwen_view_image_audit.jsonl`](../outputs/road_head/qwen_view_image_audit.jsonl)，其样本并非随机抽取，不能推算总体准确率。**这批提议不能直接转成 `/data/nuscenes/road_labels.jsonl` 训练。** 当前 mini 片段还没有得到高架上/下样本，不能据此完成五类监督训练。

复现模型提议和对比：

```bash
bash -ic 'source ~/.bashrc; /root/miniconda3/envs/uniad2.0/bin/python scripts/label_road_with_qwen.py --template outputs/road_head/road_labels_template.jsonl --output outputs/road_head/road_labels_qwen_v3_proposals.jsonl --model qwen3.8-flash'
bash -ic 'source ~/.bashrc; /root/miniconda3/envs/uniad2.0/bin/python scripts/label_road_with_qwen.py --template outputs/road_head/road_labels_flash_null23_template.jsonl --output outputs/road_head/road_labels_qwen_max_null23_proposals.jsonl --model qwen3.8-max'
conda run -n uniad2.0 python scripts/compare_qwen_road_labels.py
```

## 2026-09-24：v1.0-mini CAM_FRONT 全量 sample 关键帧

本轮“全量”按当前训练任务的 `sample_token` 语义覆盖 `v1.0-mini` 全部 **404 张 CAM_FRONT sample 关键帧**。数据目录另有 1,938 张 `sweeps/CAM_FRONT` 非关键帧；它们没有一一对应的 sample target，不属于本轮 manifest。训练 dataset 默认仍跳过每个 scene 开头历史不足的前三帧，本轮标注请求则重复最早可用帧补齐四帧上下文，并显式记录 `history_padded_count`。

新增 [`road_labels_v1mini_all_samples_template.jsonl`](../outputs/road_head/road_labels_v1mini_all_samples_template.jsonl) 和 [`road_labels_v1mini_all_samples_qwen_flash_proposals.jsonl`](../outputs/road_head/road_labels_v1mini_all_samples_qwen_flash_proposals.jsonl)。后者沿用之前已完成的 374 条 `qwen3.8-flash` v3 提议，另外请求 30 条 scene 开头帧。用 [`validate_road_proposals.py`](../scripts/validate_road_proposals.py) 检查后，404 个 token 均唯一、404 条响应均成功，10 个 scene 的当前帧路径和四帧序列均为 CAM_FRONT；完整统计见 [`v1mini_flash_labeling_summary.json`](../outputs/road_head/v1mini_flash_labeling_summary.json)。

| Flash 提议 | 数量 |
| --- | ---: |
| elevated_up | 0 |
| elevated_down | 0 |
| main_road | 305 |
| side_road | 2 |
| intersection | 71 |
| null（五类无法合理覆盖） | 26 |

30 条补历史样本中，每个 scene 的前三条分别补 3、2、1 帧。主路、辅路和路口提议的类别分布高度不平衡；`scene-1100` 的 40 条里有 34 条被标为路口中，而此前 `View Image` 和 ego pose 核查已指出该 scene 有系统性误报。因此文件字段为 `proposed_label` 和 `review_status=unverified_model_proposal`，**不是 `label` 真值，也不能直接用于 SFT 或概率校准**。保留的 Max 对比只覆盖先前的 23 条 Flash `null`；新补的 30 条中另有 3 条 `null`，尚未用 Max 复核。

复现关键帧模板、Flash 补标和覆盖率校验：

```bash
conda run -n uniad2.0 python scripts/build_nuscenes_road_manifest.py \
  --nuscenes-root /data/nuscenes --version v1.0-mini \
  --output outputs/road_head/road_labels_v1mini_all_samples_template.jsonl \
  --max-samples 500 --include-short-history
bash -ic 'source ~/.bashrc; /root/miniconda3/envs/uniad2.0/bin/python scripts/label_road_with_qwen.py --template outputs/road_head/road_labels_v1mini_all_samples_template.jsonl --output outputs/road_head/road_labels_v1mini_all_samples_qwen_flash_proposals.jsonl --model qwen3.8-flash --pad-history --workers 4'
conda run -n uniad2.0 python scripts/validate_road_proposals.py
```

`requirements_road_head.txt` 列出 NuScenes、指标绘图、YAML 和 API 请求的附加依赖。`tests/test_road_labeling.py` 覆盖短历史补帧、时间顺序、scene 校验和模型响应类别校验。

### Fork PR 归档

代码、关键帧模板、模型提议、Max 对比与视觉审核结果提交到本人 fork 的 [hova88/vggt#1](https://github.com/hova88/vggt/pull/1)，base 为 `hova88/vggt:main`。官方 `facebookresearch/vggt` 不是此 PR 的 base，也未向官方仓库推送。本地 `requirements.txt` 的既有修改和 `scripts.py` 属于另一项工作，未纳入该 PR。
