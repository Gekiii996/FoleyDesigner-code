from stable_audio_tools.models.autoencoders import AudioAutoencoder
import  torch,os,gc,torchaudio,json
from stable_audio_tools.models.factory import create_model_from_config
from torchaudio import transforms as T


audio,sr = torchaudio.load('/dky_text2audio/reverb/test/Small/dynamic/0150_id_11161/(clear tight room)left!medium2front!near.wav')
resample_tf = T.Resample(sr, 44100)
audio = resample_tf(audio)
print(type(audio))
audio = audio.to('cuda:1')
with open('/workspace/stable-audio-tools/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json') as f:
    model_config = json.load(f)

sd_path = '/workspace/stable-audio-tools/vae-finetune/vae_finetune/jgkutrjy/checkpoints/epoch=11-step=758000.ckpt'
sd = torch.load(sd_path, map_location='cpu')['state_dict']

copy_sd ={}
for k,v in sd.items():
    if 'loss' not in k and 'discriminator' not in k and 'autoencoder_ema' not in k:
        copy_sd[k.replace('autoencoder.','')] = v


sd = copy_sd



model = create_model_from_config(model_config)
model.load_state_dict(sd, strict=True)



model = model.to('cuda:1').eval()




with torch.no_grad():
    lantent,kl = model.encode(audio,return_info=True)
    print(lantent)
    reconstructed = model.decode(lantent)
    reconstructed = reconstructed.squeeze(0)
    print(reconstructed.shape)

torchaudio.save('reconstructed_audio.wav', reconstructed.cpu(), 44100)  # Save the reconstructed audio