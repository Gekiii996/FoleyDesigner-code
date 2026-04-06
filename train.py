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

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def main():

    args = get_all_args()

    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    print(f'dky batch size: {args.batch_size}, num_workers: {args.num_workers}')
    train_dl = create_dataloader_from_config(
        dataset_config, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    # validate_dataloader(train_dl, num_batches_to_check=5708)

    model = create_model_from_config(model_config)  # create_diffusion_cond_from_config(model_config)


    if args.pretrained_ckpt_path:
        if args.pretrained_ckpt_path:
            pretrained_dict = load_ckpt_state_dict(args.pretrained_ckpt_path)
            model_dict = model.state_dict()



            # 统计匹配情况
            matched_keys = []
            shape_mismatch = []
            missing_keys = []

            for key in pretrained_dict:
                if key in model_dict:
                    if pretrained_dict[key].shape == model_dict[key].shape:
                        matched_keys.append(key)
                    else:
                        shape_mismatch.append(key)
                else:
                    missing_keys.append(key)

            print(f"✅ 成功加载 {len(matched_keys)}/{len(pretrained_dict)} 个权重")
            print(f"⚠️ Shape不匹配: {len(shape_mismatch)} 个")
            print(f"⚠️ Key不存在: {len(missing_keys)} 个")

            if shape_mismatch:
                print(f"Shape不匹配的keys: {shape_mismatch[:5]}")  # 显示前5个

            copy_state_dict(model, pretrained_dict)




        # copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))
    # 查看不参与训练的参数

    # for param in model.parameters():
    #     param.requires_grad = False

    # target_layers = ['pos', 'injection_layers']

    # for name, param in model.named_parameters():
    #     for target_layer in target_layers:
    #         if target_layer in name:
    #             param.requires_grad = True
    #             # print(f"✅ 解冻层: {name}")
    #             break


    for param in model.parameters():
        param.requires_grad = True

    target_layers = ['seconds']

    for name, param in model.named_parameters():
        for target_layer in target_layers:
            if target_layer in name:
                param.requires_grad = False
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
    if trainable_params:
        print(f"\n 参与训练的参数:")
        for name in trainable_params:
            print(f"  @ {name}")

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

        # 统计匹配情况
        matched_keys = []
        shape_mismatch = []
        missing_keys = []

        for key in pretransform_dict:
            if key in pretransform_model_dict:
                if pretransform_dict[key].shape == pretransform_model_dict[key].shape:
                    matched_keys.append(key)
                else:
                    shape_mismatch.append((key, pretransform_dict[key].shape, pretransform_model_dict[key].shape))
            else:
                missing_keys.append(key)

        print(f"\n🔧 Pretransform 权重加载统计:")
        print(f"✅ 成功匹配: {len(matched_keys)}/{len(pretransform_dict)} 个权重")
        print(f"⚠️ Shape不匹配: {len(shape_mismatch)} 个")
        print(f"⚠️ Key不存在: {len(missing_keys)} 个")

        if shape_mismatch:
            print(f"\nShape不匹配详情:")
            for key, pretrained_shape, model_shape in shape_mismatch[:5]:
                print(f"  - {key}: 预训练={pretrained_shape}, 模型={model_shape}")

        if missing_keys:
            print(f"\nKey不存在的权重 (前5个): {missing_keys[:5]}")

        # 加载权重
        model.pretransform.load_state_dict(pretransform_dict, strict=False)
    
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(training_wrapper)

    exc_callback = ExceptionCallback()
    
    if args.save_dir and isinstance(wandb_logger.experiment.id, str):
        checkpoint_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
    else:
        checkpoint_dir = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    # args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2, 
                                        contiguous_gradients=True, 
                                        overlap_comm=True, 
                                        reduce_scatter=True, 
                                        reduce_bucket_size=5e8, 
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True
                                        )
        # else:
        #     strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    strategy = "auto"

    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu",
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="value",  # 改这里！！
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        # callbacks=[ckpt_callback, exc_callback, save_model_config_callback],
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        reload_dataloaders_every_n_epochs = 0
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    # main()
    # with open('/workspace/stable-audio-tools/stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json') as f:
    #     model_config = json.load(f)
    #
    # pretransform = model_config.get('model').get("pretransform", None)
    # pre = create_pretransform_from_config(pretransform,44100)
    # print(pre.scale)
    # pos = torch.randn(1, 2, 321411)
    # latnet = pre.encode(pos)
    # print(latnet.shape)
    with open('/workspace/stable-audio-tools/stable_audio_tools/configs/dataset_configs/filmstereo_test.json') as f:
        dataset_config = json.load(f)
    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=1,
        num_workers=1,
        sample_rate=44100,
        sample_size=2097152,
        audio_channels=2,
    )
    for batch_idx, batch in enumerate(train_dl):
        reals, metadata = batch
        # print(reals.shape)
        # print(metadata[0]['pos'].shape)

        for i in metadata[0]['pos']:
            print(i)
        break
    

"""
WANDB_MODE=offline
python train.py \
  --model-config /workspace/stable-audio-tools/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json \
  --dataset-config /workspace/stable-audio-tools/stable_audio_tools/configs/dataset_configs/filmstereo.json \
  --name vae_finetune \
  --batch-size 4 \
  --num-workers 8 \
  --num-nodes 1 \
  --seed 42 \
  --checkpoint-every 2000 \
  --save-dir /workspace/stable-audio-tools/vae-finetune \
  --num-gpus 1
  --ckpt-path /workspace/stable-audio-tools/vae-finetune/vae_finetune/o6djl481/checkpoints/epoch=9-step=546000.ckpt
  
WANDB_MODE=online
python train.py \
  --model-config /workspace/stable-audio-tools/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json \
  --dataset-config /workspace/stable-audio-tools/stable_audio_tools/configs/dataset_configs/filmstereo.json \
  --name vae_finetune \
  --batch-size 4 \
  --num-workers 8 \
  --num-nodes 1 \
  --seed 42 \
  --checkpoint-every 1000 \
  --save-dir /workspace/stable-audio-tools/vae-finetune \
  --pretrained-ckpt-path /workspace/stable-audio-tools/model/vae/pretransform_weights.safetensors  \
  --num-gpus 1
  --ckpt-path /workspace/stable-audio-tools/vae-finetune/vae_finetune/emq4ie00/checkpoints/epoch=7-step=351000.ckpt
  
"""



"""

"""