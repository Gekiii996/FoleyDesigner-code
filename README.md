# Spatial Audio Generation

[![Project Page](https://img.shields.io/badge/Project%20Page-Open%20Site-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white)](https://gekiii996.github.io/FoleyDesigner/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.05731)
[![Licence](https://img.shields.io/badge/Licence-Repo%20Entry-6e7781?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Gekiii996/FoleyDesigner-code)

给定音源的方位（左/前/右）和距离（近/中/远），以及在音频时间轴上的起止采样点，模型能生成与之匹配的空间化音频。

## Links

- Project page: https://gekiii996.github.io/FoleyDesigner/
- Paper: https://arxiv.org/abs/2604.05731
- Licence: repository entry on GitHub

---

## 项目结构

```
archive/
├── train.py                    # 训练入口
├── infer.py                    # 推理入口（带位置条件）
├── run_gradio.py               # Gradio Web 界面
├── vae_eval.py                 # VAE 评估
├── unwrap_model.py             # 从 Lightning checkpoint 导出模型权重
├── defaults.ini                # 训练默认参数（batch_size、精度、GPU 数等）
├── requirements.txt            # Python 依赖
├── setup.py                    # 包定义
├── scripts/
│   ├── gen_api.py              # Gradio API 批量生成
│   ├── build_local_data.py     # 构建 film_stereo 训练配置
│   └── ds_zero_to_pl_ckpt.py  # DeepSpeed checkpoint 转换
└── stable_audio_tools/
    ├── data/
    │   ├── pos.py              # CondPosWave（位置矩阵生成）
    │   ├── dataset.py          # 数据集类（含 film_stereo 类型）
    │   └── utils.py
    ├── models/
    │   ├── posembd.py          # PosEmbd（位置 embedding）
    │   ├── conditioners.py     # 含 PosConditioner
    │   ├── transformer.py      # 含 injection_layers
    │   ├── dit.py              # DiT，pos_embd 注入
    │   ├── diffusion.py        # pos_embd 路由
    │   └── ...
    ├── training/
    │   ├── diffusion.py        # 训练 wrapper
    │   └── ...
    ├── inference/
    └── configs/
        ├── model_configs/
        │   └── txt2audio/
        │       ├── stable_audio_2_0.json          # 带位置条件的模型配置
        │       └── stable_audio_2_0_extraBlock.json  # 无位置条件（baseline）
        └── dataset_configs/
            ├── filmstereo_train.json  # 训练集（film_stereo 格式，实际数据）
            └── local_training_example.json  # audio_dir 格式示例
```

---

## 环境安装

```bash
# 推荐 Python 3.10
pip install -r requirements.txt
pip install -e .
```

---

## 数据格式

本项目使用 `film_stereo` 数据集类型，每个音频样本需包含位置标注。

**dataset_config JSON 格式：**

```json
{
    "dataset_type": "film_stereo",
    "datasets": [
        {
            "id": "my_dataset",
            "path": "/path/to/audio/left!near.wav",
            "prompt": "a sound of footsteps, coming from the left at a distant near, in reverb like ...",
            "motion_type": "static",
            "movement_str": "left!near.wav",
            "start_points": [1000],
            "end_points": [44100]
        },
        {
            "id": "my_dataset",
            "path": "/path/to/audio/left!near2right!far.wav",
            "prompt": "a car passing by, moving from left near to right far ...",
            "motion_type": "dynamic",
            "movement_str": "left!near2right!far.wav",
            "start_points": [0],
            "end_points": [220500]
        }
    ],
    "random_crop": false
}
```

**位置编码规则：**

| 字段 | 可选值 | 说明 |
|------|--------|------|
| `motion_type` | `"static"` / `"dynamic"` | 音源是否移动 |
| `movement_str`（static） | `"{angle}!{depth}.wav"` | 如 `"left!near.wav"` |
| `movement_str`（dynamic） | `"{angle}!{depth}2{angle}!{depth}.wav"` | 如 `"left!near2right!far.wav"` |
| angle | `left` / `left_front` / `front` / `right_front` / `right` | 方位（5 档） |
| depth | `near` / `medium` / `far` | 距离（3 档） |
| `start_points` / `end_points` | 采样点索引列表（44100 sr） | 音源活跃区间，可多段 |

**构建训练配置：**

```bash
# 从原始 JSONL 构建 film_stereo 配置
python scripts/build_local_data.py
```

---

## 训练

### 参数说明（defaults.ini 中可配置）

```ini
batch_size = 4           # 每 GPU batch size
num_gpus = 1             # GPU 数量
precision = "16-mixed"   # 训练精度
num_workers = 6          # DataLoader 工作进程数
checkpoint_every = 10000 # 每多少步保存 checkpoint
save_top_k = 1           # 保留最优 K 个 checkpoint
```

### 启动训练（冻结主干，只训练位置相关参数）

```bash
python train.py \
    --model_config stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json \
    --dataset_config stable_audio_tools/configs/dataset_configs/filmstereo_train.json \
    --pretrained_ckpt_path /path/to/stable_audio_2_0.ckpt \
    --save_dir ./checkpoints \
    --batch_size 4 \
    --num_gpus 1 \
    --precision 16-mixed \
    --name pos_finetune
```

训练时只有 `pos_embd`（位置 embedding）和 `injection_layers`（交叉注意力注入层）参与梯度更新，主干权重全部冻结。参见 `infer.py: create_model()` 中的参数冻结逻辑。

### 多卡训练

```bash
python train.py \
    --model_config ... \
    --dataset_config ... \
    --num_gpus 4 \
    --strategy ddp \
    --batch_size 2 \
    --accum_batches 4
```

---

## 推理

```bash
python infer.py \
    --model_config stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json \
    --pretrained_ckpt_path /path/to/pretrained.ckpt \
    --ckpt_path /path/to/pos_finetune.ckpt
```

推理时在 `infer.py` 底部的 `conditioning_list` 中填写生成条件：

```python
conditioning_list = [
    {
        "prompt": "a dog barking, coming from the right",
        "seconds_total": 10,
        "motion_type": "static",
        "movement_str": "right!near.wav",
        "start_seconds": [1.0],
        "end_seconds": [8.0],
    }
]
```

---

## 模型配置说明

模型配置文件 `configs/model_configs/txt2audio/stable_audio_2_0.json` 中，位置条件相关字段：

```json
{
    "id": "pos",
    "type": "pos",
    "config": {
        "fourier_M": 64,       // Fourier 特征维度（输出 2*M=128 维）
        "latent_dim": 64,      // 位置 latent 维度
        "max_length": 2097152, // 支持的最大音频采样点数（约 47 秒 @ 44100 Hz）
        "finetune": true       // true = 冻结 pos_embd，只训练 injection_layers
    }
}
```

`injection_layers` 注入位置固定在 Transformer 的第 3/7/11/15/19/23 个 block（共 6 处，共 24 层），每处为一个独立的轻量交叉注意力模块。

---

## 核心数据流

```
音频文件 (WAV)
    │
    ├── [VAE Encoder] → latent [B, 64, T/2048]
    │
    └── 位置标注 (movement_str + start/end_points)
            │
            └── CondPosWave → pos_matrix [T, 3]  (angle, depth, mask)
                    │
                    └── PosEmbd (Fourier + 下采样卷积) → pos_embd [B, 64, T']
                            │
                            └── PosConditioner → [pos_embd, mask]
                                    │
                                    ↓
                            DiT Transformer
                            (injection_layers 在第 3/7/11/15/19/23 block
                             通过交叉注意力将 pos_embd 注入主干)
                                    │
                                    ↓
                            生成 latent → [VAE Decoder] → 音频
```
