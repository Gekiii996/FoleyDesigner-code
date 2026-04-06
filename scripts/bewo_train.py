import torch
import json
import os
import pytorch_lightning as pl

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
# from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
model_config ={    "sample_size": 220500,
    "sample_rate": 44100,}
with open('/workspace/stable-audio-tools/stable_audio_tools/configs/dataset_configs/bewo_train.json') as f:
    dataset_config = json.load(f)
train_dl = create_dataloader_from_config(
    dataset_config, 
    batch_size=8, 
    num_workers=6,
    sample_rate=model_config["sample_rate"],
    sample_size=model_config["sample_size"],
    audio_channels=model_config.get("audio_channels", 2),
)
if train_dl is not None:
    try:
        # 尝试取出一个 batch 看看
        batch = next(iter(train_dl))
        print("Successfully loaded a batch!")
        print(f"Batch keys: {batch.keys()}")
    except StopIteration:
        print("ERROR: DataLoader is empty! No audio files found in the specified path.")
    except Exception as e:
        print(f"ERROR during batch loading: {e}")
else:
    print("ERROR: DataLoader is None. Check dataset_type in JSON.")
print("-----------------------------")