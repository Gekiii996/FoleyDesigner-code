# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

基于扩散模型（DiT Transformer）的空间音频生成系统（Stable Audio Tools 分支）。核心创新点：在预训练模型主干冻结的前提下，通过轻量级交叉注意力注入层添加空间位置条件（方位 + 距离），实现带有空间感的立体声音频生成。

## 常用命令

```bash
# 环境安装
pip install -r requirements.txt && pip install -e .

# 训练
python train.py \
  --model_config stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json \
  --dataset_config stable_audio_tools/configs/dataset_configs/filmstereo_train.json \
  --pretrained_ckpt_path /path/to/pretrained.ckpt \
  --num_gpus 1 --batch_size 4 --precision 16-mixed

# 推理
python infer.py --model_config ... --ckpt_path ...

# Gradio Web 界面
python run_gradio.py

# VAE 评估
python vae_eval.py

# 从 Lightning checkpoint 导出裸模型权重
python unwrap_model.py

# DeepSpeed checkpoint 转换
python scripts/ds_zero_to_pl_ckpt.py
```

训练超参数默认值在 `defaults.ini`（batch_size=4，precision="16-mixed"，num_gpus=1 等）。

## 架构概览

### 数据流

```
音频文件 (WAV, 文件名含位置信息)
         ↓ CondPosWave (data/pos.py)
位置矩阵 [T, 3] = [angle, depth, mask]
         ↓ PosConditioner → PosEmbd (models/posembd.py, conditioners.py)
位置 latent [B, 64, T_down]
         ↓ 注入 DiT 第 3/7/11/15/19/23 层（交叉注意力）
扩散模型去噪 → VAE Decoder → 生成立体声音频
```

### 关键模块

| 模块 | 路径 | 作用 |
|------|------|------|
| 位置矩阵生成 | `stable_audio_tools/data/pos.py` | `CondPosWave`：将文件名中的位置字符串（如 `left!near.wav`）解码为位置矩阵 |
| 位置 Embedding | `stable_audio_tools/models/posembd.py` | `FourierFeatureEncoder`（Fourier 特征）+ `PosDownsampleEncoder`（5层卷积下采样） |
| 条件编码器 | `stable_audio_tools/models/conditioners.py` | `PosConditioner` 封装 PosEmbd，输出 `[latent, mask]`；还有 T5 文本和数值时长条件 |
| 注入层 | `stable_audio_tools/models/transformer.py` | `injection_layers`：在主干 6 个位置插入的轻量级交叉注意力（`zero_init_output=True`） |
| DiT 主干 | `stable_audio_tools/models/dit.py` | 24 层 Transformer，embed_dim=1536，24 头；位置注入在 forward 中按层号触发 |
| 训练循环 | `stable_audio_tools/training/diffusion.py` | `DiffusionCondTrainingWrapper`：梯度监控、参数冻结、EMA |
| 采样算法 | `stable_audio_tools/inference/sampling.py` | DPM++ 3M-SDE、Discrete Euler 等 |

### 参数高效微调设计

- 主干 VAE、T5、Transformer 权重全部冻结，仅训练 `pos_embd` + `injection_layers`（约 1M 参数）
- LR 极小（3e-7），防止灾难性遗忘
- 是否可训练由模型配置中的 `"finetune": true` 字段控制

### 位置编码规则

文件名格式控制位置（由 `CondPosWave` 解析）：
- 静态：`left!near.wav`
- 动态：`left!near2right!far.wav`
- 角度：1=left, 2=left_front, 3=front, 4=right_front, 5=right
- 距离：1=near, 2=medium, 3=far

### 配置文件

- **模型配置**：`stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json`（含位置条件）；`*_extraBlock.json` 为无位置条件的 Baseline
- **数据配置**：`stable_audio_tools/configs/dataset_configs/filmstereo_*.json`；`motion_type` 字段区分 `static`/`dynamic`，音频路径的文件名即为 `movement_str`

## 技术债与注意事项

- 数据配置中 `start_points`/`end_points` 单位为采样点（44100Hz），而推理 API 使用 `start_seconds`/`end_seconds`，注意换算
- `injection_layers` 的注入层号（3/7/11/15/19/23）硬编码在 `transformer.py` forward 中，修改深度时需同步修改
