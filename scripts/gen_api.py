from gradio_client import Client, handle_file
import torch,json
from torch.utils.data import Dataset
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path

class AudioCaps(Dataset):
    def __init__(self, target_length=None,data_type='train',debug = False):


        self.target_length = target_length
        self.debug = debug


        test_data = pd.read_csv('/workspace/BothEars/matched_rows.csv')
        base_path = Path('/workspace/AudioCaps')

        dic_ls = []
        tmp = 0
        for index, row in test_data.iterrows():
            dic_temp = {}
            caption_id = row['audiocap_id']

            audio_path = base_path / Path(f'{caption_id}.wav')
            if (not audio_path.exists()): continue

            caption = row['caption']

            # if 'talk' in caption or 'speak' in caption:
            #     tmp += 1
            #     print(f'skip : {tmp}')
            #     continue

            dic_temp['path'] = str(audio_path)
            dic_temp['caption'] = caption
            # print(f'test {tmp}')

            dic_ls.append(dic_temp)


        random.seed(42)  # 设置随机种子以确保可重复性
        random.shuffle(dic_ls)

        self.data_type = data_type
        self.lenght = len(dic_ls)
        print(f'data type = {data_type} length = {self.lenght}')
        self.audios = dic_ls

    def __len__(self):
        if self.data_type == 'train':
            return 60000
            # return 4000

        if self.data_type == 'valid':
            return 6000
            # return 400
        if self.data_type == 'test':
            return 200

    def __getitem__(self, idx):
        if self.debug:
            return 1

        # 加载音频文件
        audio = self.audios[idx]

        prompt = audio['caption']


        # 返回字典，键名与 training_step 匹配
        return {
            'mel_audio': torch.zeros(1),
            'prompt': prompt,
        }





def main():
    output_dir = Path("/workspace/stable-audio-tools/Audiocaps_MMA_stableAudio")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ds = AudioCaps(data_type="test")
    
    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    generated_data = {
        "epoch": "test",
        "samples": []
    }
    
    for sample_idx, batch in enumerate(tqdm(test_dataloader, desc="Generating samples")):
        prompt = batch['prompt'][0]

        client = Client("http://127.0.0.1:7860/")
        result = client.predict(
                prompt=prompt,
                negative_prompt=None,
                seconds_start=0,
                seconds_total=10,
                cfg_scale=7,
                steps=100,
                preview_every=0,
                seed="-1",
                sampler_type="dpmpp-3m-sde",
                sigma_min=0.03,
                sigma_max=500,
                cfg_rescale=0,
                use_init=False,
                init_audio=None,
                init_noise_level=0.1,
                api_name="/generate"
        )
        print(result)

        #result[0] 是保存的wave地址，
        # todo 需要加载MMA数据集，生成数据，然后保存为json，也是类似的操作

        sample_info = {
            "sample_index": sample_idx,
            "prompt": prompt,
            "audio_filename": result[0],
        }

        generated_data["samples"].append(sample_info)


    metadata_path = output_dir / "test_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
