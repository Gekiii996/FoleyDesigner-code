import json
import os
import random
import shutil
from pathlib import Path

# ================= ⚙️ 配置区域 =================
# 1. 原始数据路径
ORIGINAL_JSONL = "/dky_text2audio/reverb/jsons/train.jsonl"
ORIGINAL_AUDIO_ROOT = Path("/dky_text2audio/reverb/")
WINDOWS_ROOT_PREFIX = 'D:\\dky_data_reverb'

# 2. 输出路径 (影子数据集位置)
# 我们将在这里创建软链接，不占硬盘空间
TARGET_DIR = Path("/dky_text2audio/reverb_10k_subset/")
TARGET_JSONL = TARGET_DIR / "filtered_metadata.jsonl"

# 3. 最终生成的训练配置文件路径
FINAL_CONFIG_JSON = "/workspace/stable-audio-tools/stable_audio_tools/configs/dataset_configs/bewo_train.json"
# 你的 custom.py 路径
CUSTOM_MODULE_PATH = "/workspace/BothEars/models/bewo_config/custom.py"

# 4. 采样数量
MAX_SAMPLES = 10000

# ================= 🚀 执行逻辑 =================

def prepare_dataset():
    if TARGET_DIR.exists():
        print(f"清理旧目录: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"正在读取: {ORIGINAL_JSONL}")
    with open(ORIGINAL_JSONL, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    print(f"原始数据共 {len(all_lines)} 条，正在打乱...")
    random.shuffle(all_lines)

    valid_entries = []
    count = 0
    
    print("开始构建影子数据集 (创建软链接)...")
    
    for line in all_lines:
        if count >= MAX_SAMPLES:
            break
            
        try:
            data = json.loads(line)
            win_path = data.get('reverb_path')
            if not win_path: continue

            # 路径转换逻辑 (Windows -> Linux)
            rel_str = str(win_path).replace(WINDOWS_ROOT_PREFIX, "").lstrip("\\/")
            rel_path = '/'.join(rel_str.split('\\'))
            
            src_file = ORIGINAL_AUDIO_ROOT / rel_path
            
            # 检查源文件是否存在
            if not src_file.exists():
                continue
                
            # 构建目标路径 (保持原有目录结构，避免文件名冲突)
            dst_file = TARGET_DIR / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建软链接 (核心步骤)
            # dst_file 指向 src_file
            os.symlink(src_file, dst_file)
            
            # 记录这条有效数据
            valid_entries.append(line)
            count += 1
            
            if count % 2000 == 0:
                print(f"已处理: {count} / {MAX_SAMPLES}")

        except Exception as e:
            print(f"跳过错误数据: {e}")
            continue

    # 保存筛选后的 JSONL (给 custom.py 使用)
    with open(TARGET_JSONL, 'w', encoding='utf-8') as f:
        f.writelines(valid_entries)
    
    print(f"\n✅ 影子数据集构建完成！")
    print(f"位置: {TARGET_DIR}")
    print(f"有效样本数: {count}")
    print(f"过滤后的元数据: {TARGET_JSONL}")
    
    # 生成训练配置文件
    generate_config_file()

def generate_config_file():
    config_data = {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": "bewo_subset_10k",
                # 注意：这里指向我们刚创建的影子目录
                "path": str(TARGET_DIR),
                "custom_metadata_module": CUSTOM_MODULE_PATH
            }
        ],
        "random_crop": True
    }
    
    os.makedirs(os.path.dirname(FINAL_CONFIG_JSON), exist_ok=True)
    with open(FINAL_CONFIG_JSON, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ 训练配置文件已生成: {FINAL_CONFIG_JSON}")

if __name__ == "__main__":
    prepare_dataset()
