import os
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from pathlib import  Path

warnings.filterwarnings('ignore')

# 你原有的配置构建代码
config_tmp = {
    "dataset_type": "film_stereo",
    "datasets": [],
    "random_crop": False
}

base_path = '/dky_text2audio/reverb/train/'


relative_path = 'D:\\dky_data_reverb'

with open('/dky_text2audio/reverb/jsons/train.jsonl', 'r', encoding='utf-8') as f:
    temp_jsons = []
    lines = f.readlines()
    con = 1
    for line in lines:

        if con >40000:
            break
        con += 1
        tmp = {"id": 'my_audio'}

        data = json.loads(line)
        path = data.get('reverb_path', None)
        relative_str = str(path).replace(str(relative_path), "").lstrip("\\/")
        relative = '/'.join(relative_str.split('\\'))
        linux_path = '/dky_text2audio/reverb/' / Path(relative)
        if ((not linux_path.exists())): continue

        linux_path = str(linux_path)

        tmp['path'] = linux_path
        tmp['prompt'] = data['audio_caption']
        tmp['motion_type'] = data['motion_type']
        tmp['movement_str'] = data['audio_path'].split('/')[-1]
        tmp['start_points'] = data['condition']['start_points']
        tmp['end_points'] = data['condition']['end_points']
        config_tmp['datasets'].append(tmp)



