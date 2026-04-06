import torch
import torchaudio
from stable_audio_tools.data.dataset import CondPosWave
from stable_audio_tools.training import create_training_wrapper_from_config
from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
import random

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict
from stable_audio_tools.models.factory import create_pretransform_from_config
from stable_audio_tools.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler


def generate_audio(
        model,
        conditioning_list: list,
        output_dir: str = "./outputs",
        demo_steps: int = 250,
        cfg_scales: list = [3, 5, 7],
        sample_rate: int = 44100
):
    """
    生成音频的推理函数
    """
    model.eval()

    with torch.no_grad():
        for cfg_scale in cfg_scales:
            print(f"Generating with CFG scale: {cfg_scale}")

            # 准备条件
            demo_conds = []
            for cond_info in conditioning_list:
                # 检查并转换秒数为采样点
                if "start_seconds" in cond_info and "end_seconds" in cond_info:
                    start_samples = [int(s * sample_rate) for s in cond_info["start_seconds"]]
                    end_samples = [int(e * sample_rate) for e in cond_info["end_seconds"]]
                else:
                    start_samples = cond_info.get("start_points")
                    end_samples = cond_info.get("end_points")


                pos_wave = CondPosWave.conditioning_pos_wave(
                    movement_str=cond_info.get("movement_str"),
                    motion_type=cond_info.get("motion_type"),
                    start_samples=start_samples,
                    end_samples=end_samples,
                    num_samples_target=int(cond_info.get("seconds_total", 2.0) * sample_rate),
                    ori_sr=sample_rate,
                    target_sr=sample_rate,
                )

                demo_conds.append({
                    "prompt": cond_info.get("prompt", ""),
                    "pos": torch.tensor(pos_wave, dtype=torch.float32),
                    "sample_rate": sample_rate,
                    "motion_type": cond_info.get("motion_type"),
                    "seconds_start": cond_info.get("seconds_start", 0),
                    "seconds_total": cond_info.get("seconds_total", 2.0),
                })

            # 获取条件输入
            conditioning = model.diffusion.conditioner(demo_conds, model.device)
            cond_inputs = model.diffusion.get_conditioning_inputs(conditioning)

            # 准备噪声
            demo_samples = len(demo_conds[0]["pos"])
            if model.diffusion.pretransform is not None:
                demo_samples = demo_samples // model.diffusion.pretransform.downsampling_ratio

            noise = torch.randn([len(demo_conds), model.diffusion.io_channels, demo_samples]).to(model.device)

            # 采样
            with torch.cuda.amp.autocast():
                sampling_model = model.diffusion_ema.model if model.diffusion_ema is not None else model.diffusion.model

                if model.diffusion_objective == "v":
                    fakes = sample(sampling_model, noise, demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale,
                                   batch_cfg=True)
                elif model.diffusion_objective == "rectified_flow":
                    fakes = sample_discrete_euler(sampling_model, noise, demo_steps, **cond_inputs, cfg_scale=cfg_scale,
                                                  batch_cfg=True)
                else:
                    raise ValueError(f"Unknown objective: {model.diffusion_objective}")

                # 如果有预变换，解码
                if model.diffusion.pretransform is not None:
                    fakes = model.diffusion.pretransform.decode(fakes)

                    print(fakes.shape)

            # 保存每个音频
            for i, audio in enumerate(fakes):
                audio = audio.squeeze().cpu()
                print(audio.shape)
                audio = audio / torch.max(torch.abs(audio)) * 0.9  # 归一化
                audio = (audio * 32767).to(torch.int16)

                filename = f"{output_dir}/generated_cfg{cfg_scale}_{i:02d}.wav"
                torchaudio.save(filename, audio, sample_rate)
                print(f"Saved: {filename}")

def create_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda')
    injection_sd = {}
    for i in ckpt['state_dict']:
        if 'pos' in i or 'injection' in i:
            new_k = i.replace('diffusion.', '')
            injection_sd[new_k] = ckpt['state_dict'][i]



    # 加载训练好的模型
    args = get_all_args()

    with open(args.model_config) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)  # create_diffusion_cond_from_config(model_config)
    # model.load_state_dict(ckpt['state_dict'])

    if args.pretrained_ckpt_path:
        pretrained_dict = load_ckpt_state_dict(args.pretrained_ckpt_path)

        print(f"\n🔧 正在合并 injection 权重...")
        original_keys_count = len(pretrained_dict)
        pretrained_dict.update(injection_sd)
        print(f"原始权重: {original_keys_count}, 合并后总权重: {len(pretrained_dict)}")


        model_dict = model.state_dict()

        pretrained_keys = set(pretrained_dict.keys())
        for i in pretrained_keys:
            if 'seconds' in i:
                print(i)
        model_keys = set(model_dict.keys())

        # 统计匹配情况
        matched_keys = []
        shape_mismatch = []
        missing_from_model_keys = []  # 权重中有但模型中没有的key

        for key in pretrained_keys:
            if key in model_dict:
                if pretrained_dict[key].shape == model_dict[key].shape:
                    matched_keys.append(key)
                else:
                    shape_mismatch.append(key)
            else:
                missing_from_model_keys.append(key)

        # 模型中有但权重中没有的key (未加载的参数)
        unloaded_keys = list(model_keys - pretrained_keys)

        print(f"✅ 成功加载 {len(matched_keys)}/{len(model_keys)} 个模型权重")
        print(f"⚠️ Shape不匹配: {len(shape_mismatch)} 个")
        print(f"⚠️ 权重文件存在但模型不存在的Key: {len(missing_from_model_keys)} 个")
        print(f"⚠️ 模型中未被加载的参数: {len(unloaded_keys)} 个")


        if shape_mismatch:
            print(f"\nShape不匹配的keys (前5个): {shape_mismatch[:5]}")

        if unloaded_keys:
            print(f"\n模型中未被加载的参数 (前10个):")
            for key in unloaded_keys[:10]:
                print(f"  - {key}")

        copy_state_dict(model, pretrained_dict)

    # 查看不参与训练的参数
    for param in model.parameters():
        param.requires_grad = False

    target_layers = ['pos', 'injection_layers']

    for name, param in model.named_parameters():
        for target_layer in target_layers:
            if target_layer in name:
                param.requires_grad = True
                # print(f"✅ 解冻层: {name}")
                break

    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\n📊 参数训练状态统计:")
    print(f"🔓 可训练参数: {len(trainable_params)} 个")
    print(f"🔒 冻结参数: {len(frozen_params)} 个")
    print(f"总参数: {len(trainable_params) + len(frozen_params)} 个")

    # if frozen_params:
    #     print(f"\n🔒 不参与训练的参数:")
    #     for name in frozen_params:
    #         print(f"  - {name}")
    #
    # if trainable_params:
    #     print(f"\n 参与训练的参数:")
    #     for name in trainable_params:
    #         print(f"  @ {name}")

    # 可选: 显示参数量统计
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"\n💾 参数量统计:")
    print(f"可训练参数量: {trainable_param_count:,}")
    print(f"总参数量: {total_param_count:,}")
    print(f"训练参数占比: {trainable_param_count / total_param_count * 100:.2f}%")

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:

        pretransform_dict = load_ckpt_state_dict(args.pretransform_ckpt_path)
        pretransform_model_dict = model.pretransform.state_dict()

        pretransform_keys = set(pretransform_dict.keys())
        pretransform_model_keys = set(pretransform_model_dict.keys())

        # 统计匹配情况
        matched_keys = []
        shape_mismatch = []
        missing_from_model_keys = []

        for key in pretransform_keys:
            if key in pretransform_model_dict:
                if pretransform_dict[key].shape == pretransform_model_dict[key].shape:
                    matched_keys.append(key)
                else:
                    shape_mismatch.append((key, pretransform_dict[key].shape, pretransform_model_dict[key].shape))
            else:
                missing_from_model_keys.append(key)

        unloaded_keys = list(pretransform_model_keys - pretransform_keys)

        print(f"\n🔧 Pretransform 权重加载统计:")
        print(f"✅ 成功匹配: {len(matched_keys)}/{len(pretransform_model_keys)} 个权重")
        print(f"⚠️ Shape不匹配: {len(shape_mismatch)} 个")
        print(f"⚠️ 权重文件存在但模型不存在的Key: {len(missing_from_model_keys)} 个")
        print(f"⚠️ Pretransform模型中未被加载的参数: {len(unloaded_keys)} 个")


        if shape_mismatch:
            print(f"\nShape不匹配详情:")
            for key, pretrained_shape, model_shape in shape_mismatch[:5]:
                print(f"  - {key}: 预训练={pretrained_shape}, 模型={model_shape}")

        if unloaded_keys:
            print(f"\nPretransform模型中未被加载的参数 (前10个):")
            for key in unloaded_keys[:10]:
                print(f"  - {key}")

        # 加载权重

    model = create_training_wrapper_from_config(model_config, model)
    model = model.cuda()
    return model


if __name__ == "__main__":

    # 创建模型
    model = create_model(ckpt_path="/workspace/stable-audio-tools/dit/dit_injection/02yersh2/checkpoints/epoch=9-step=100000.ckpt")

    # 定义生成条件
    conditioning_list = [
        {
            "prompt": "a woman crying and screaming",
            "seconds_total": 10,
            "motion_type": "static",
            "movement_str": "front!near.wav",
            "start_seconds": [0.0,3.0,],
            "end_seconds": [2.0,6.0,]
        },
        # 你也可以添加另一个事件，例如从第6秒到第8秒
        {
            "prompt": "a woman screaming",
            "seconds_total": 10,
            "motion_type": "static",
            "movement_str": "front!near.wav",
            "start_seconds": [0.0,3.0,],
            "end_seconds": [2.0,6.0,]
        },

                {
            "prompt": "a woman crying, yawning, with a sad expression",
            "seconds_total": 10,
            "motion_type": "static",
            "movement_str": "front!near.wav",
            "start_seconds": [0.0,1.0,2.0],
            "end_seconds": [1.0,2.0,3.0]
        },


    ]


    # 生成音频
    generate_audio(
        model=model,
        conditioning_list=conditioning_list,
        output_dir="./generated_audio",
        demo_steps=100,  # 可以调整步数
        cfg_scales=[5,]  # 只生成这两个CFG尺度
    )
