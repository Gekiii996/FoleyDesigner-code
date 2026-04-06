import pytorch_lightning as pl
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb
from torch.nn.utils import clip_grad_norm_
import  numpy as np


from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
import auraloss
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from sympy.strategies import condition
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from ..models.diffusion import DiffusionModelWrapper, ConditionedDiffusionModelWrapper
from ..models.autoencoders import DiffusionAutoencoder
from ..models.diffusion_prior import PriorType
from .autoencoders import create_loss_modules_from_bottleneck
from .losses import AuralossLoss, MSELoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config

from time import time
from stable_audio_tools.data.dataset import CondPosWave

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

class DiffusionUncondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training an unconditional audio diffusion model (like Dance Diffusion).
    '''
    def __init__(
            self,
            model: DiffusionModelWrapper,
            lr: float = 1e-4,
            pre_encoded: bool = False
    ):
        super().__init__()

        self.diffusion = model
        
        self.diffusion_ema = EMA(
            self.diffusion.model,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1
        )

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        loss_modules = [
            MSELoss("v",
                     "targets",
                     weight=1.0,
                     name="mse_loss"
                )
        ]

        self.losses = MultiLoss(loss_modules)

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        
        diffusion_input = reals

        loss_info = {}

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        if self.diffusion.pretransform is not None:
            if not self.pre_encoded:
                with torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        loss_info["reals"] = diffusion_input

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas
        targets = noise * alphas - diffusion_input * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_inputs, t)

            loss_info.update({
                "v": v,
                "targets": targets
            })

            loss, losses = self.losses(loss_info)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):

        self.diffusion.model = self.diffusion_ema.ema_model
        
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionUncondDemoCallback(pl.Callback):
    def __init__(self, 
                 demo_every=2000,
                 num_demos=8,
                 demo_steps=250,
                 sample_rate=48000
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step

        demo_samples = module.diffusion.sample_size

        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

        try:
            with torch.cuda.amp.autocast():
                fakes = sample(module.diffusion_ema, noise, self.demo_steps, 0)

                if module.diffusion.pretransform is not None:
                    fakes = module.diffusion.pretransform.decode(fakes)

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)

            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
        
            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            trainer.logger.experiment.log(log_dict)

            del fakes
            
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
        finally:
            gc.collect()
            torch.cuda.empty_cache()

class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''





    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective

        self.loss_modules = [
            MSELoss("output", 
                   "targets", 
                   weight=1.0, 
                   mask_key="padding_mask" if self.mask_padding else None, 
                   name="mse_loss"
            )
        ]

        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def _print_block_summary(self, pos_mean, pos_max, pos_count, inj_stats):
        step = self.global_step
        print(f"\n{'=' * 70}")
        print(f"GRADIENT BLOCK STATS @ Step {step}")
        print(f"{'=' * 70}")

        # POS 整体
        status = "NORMAL" if pos_mean < 2.0 else "HIGH"
        print(f"POS (Total {pos_count} params)")
        print(f"  Mean: {pos_mean:8.4f} | Max: {pos_max:8.4f} | Status: {status}")

        # Injection Blocks
        print(f"\nINJECTION BLOCKS (6 blocks)")
        print(f"{'Block':<6} {'Params':<8} {'Mean':<8} {'Max':<8} {'Status':<8}")
        print("-" * 50)
        for block_idx in sorted(inj_stats.keys()):
            s = inj_stats[block_idx]
            status = "NORMAL" if s["mean"] < 10.0 else "HIGH"
            print(f"{block_idx:<6} {s['count']:<8} {s['mean']:<8.4f} {s['max']:<8.4f} {status}")

        # 警告
        if pos_mean > 5.0 or any(s["mean"] > 20 for s in inj_stats.values()):
            print(f"\nGRADIENT EXPLOSION WARNING! Check LR / init / clip")
        print(f"{'=' * 70}\n")

    def _log_block_summary(self, pos_mean, pos_max, inj_stats):
        self.log("grad/pos_mean", pos_mean, on_step=True)
        self.log("grad/pos_max", pos_max, on_step=True)

        for block_idx, s in inj_stats.items():
            self.log(f"grad/inj_block_{block_idx}_mean", s["mean"], on_step=True)
            self.log(f"grad/inj_block_{block_idx}_max", s["max"], on_step=True)

    def on_after_backward(self):
        if self.global_step % 50 != 0:
            return
        # # 打印每一层的最大和均值梯度
        # for name, param in self.diffusion.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Layer {name} | Mean Grad: {param.grad.mean()} | Max Grad: {param.grad.max()}")

        # === 1. 按 Block 分组统计 injection_layers ===
        inj_block_stats = {}  # {block_idx: [norm1, norm2, ...]}

        for name, param in self.diffusion.model.named_parameters():
            if "injection_layers" in name and param.grad is not None:
                # 提取 block 编号: injection_layers.3.xxx → 3
                try:
                    block_idx = name.split("injection_layers.")[1].split(".")[0]
                    block_idx = int(block_idx)
                except:
                    continue

                if block_idx not in inj_block_stats:
                    inj_block_stats[block_idx] = []
                inj_block_stats[block_idx].append(param.grad.norm(2).item())

        # === 2. pos 整体统计（16 个参数）===
        pos_norms = []
        for name, param in self.diffusion.conditioner.named_parameters():
            if "pos" in name and param.grad is not None:
                pos_norms.append(param.grad.norm(2).item())

        # === 3. 计算统计量 ===
        # pos 整体
        pos_count = len(pos_norms)
        pos_mean = sum(pos_norms) / pos_count if pos_count > 0 else 0
        pos_max = max(pos_norms) if pos_norms else 0
        pos_total_norm = sum(x ** 2 for x in pos_norms) ** 0.5

        # injection 每 block
        inj_block_final = {}
        for block_idx, norms in inj_block_stats.items():
            count = len(norms)
            mean = sum(norms) / count
            max_norm = max(norms)
            total_norm = sum(x ** 2 for x in norms) ** 0.5
            inj_block_final[block_idx] = {
                "count": count,
                "mean": mean,
                "max": max_norm,
                "total": total_norm
            }
        clip_grad_norm_([p for p in self.parameters() if p.requires_grad], max_norm=1.0)
        # === 4. 打印（简洁表格）===
        self._print_block_summary(pos_mean, pos_max, pos_count, inj_block_final)

        # === 5. WandB 日志 ===
        self._log_block_summary(pos_mean, pos_max, inj_block_final)

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        with torch.cuda.amp.autocast():
            conditioning = self.diffusion.conditioner(metadata, self.device)

        # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        #self.mask_padding False。   self.mask_padding_dropout 0.0
        # skip
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout
        # skip

        # Create batch tensor of attention masks from the "mask" field of the metadata array
        if use_padding_mask:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        p.tick("conditioning")

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    p.tick("pretransform")

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    if use_padding_mask:
                        padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
            
        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        p.tick("noise")

        extra_args = {}

        if use_padding_mask:
            extra_args["mask"] = padding_masks

        with torch.cuda.amp.autocast():
            # self.diffusion == ConditionedDiffusionModelWrapper
            p.tick("amp")
            output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
            p.tick("diffusion")

            loss_info.update({
                "output": output,
                "targets": targets,
                "padding_mask": padding_masks if use_padding_mask else None,
            })

            loss, losses = self.losses(loss_info)

            p.tick("loss")

            if self.log_loss_info:
                # Loss debugging logs
                num_loss_buckets = 10
                bucket_size = 1 / num_loss_buckets
                loss_all = F.mse_loss(output, targets, reduction="none")

                sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

                # gather loss_all across all GPUs
                loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

                # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
                loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

                # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
                debug_log_dict = {
                    f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
                }

                self.log_dict(debug_log_dict)


        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        #print(f"Profiler: {p}")

        # if self.global_step > 0:
        #     # 收集所有可训练参数
        #     params_to_clip = []
        #     for name, param in self.diffusion.model.named_parameters():
        #         if param.grad is not None:
        #             params_to_clip.append(param)
        #     for name, param in self.diffusion.conditioner.named_parameters():
        #         if param.grad is not None:
        #             params_to_clip.append(param)
        #
        #     if params_to_clip:
        #         torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)

        return loss


    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model
        
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionCondDemoCallback(pl.Callback):
    def __init__(self, 
                 demo_every=2000,
                 num_demos=8,
                 sample_size=65536,
                 demo_steps=250,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = {},
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
                 demo_cond_from_batch: bool = False,
                 display_audio_cond: bool = False
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales

        # If true, the callback will use the metadata from the batch to generate the demo conditioning
        self.demo_cond_from_batch = demo_cond_from_batch

        # If true, the callback will display the audio conditioning
        self.display_audio_cond = display_audio_cond

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        # 修正：创建两个变量，清晰地区分音频和潜空间的样本长度
        audio_samples = self.demo_samples  # e.g., 65536. 这是最大音频长度
        latent_samples = audio_samples     # 默认与音频长度一致

        demo_cond = self.demo_conditioning

        if self.demo_cond_from_batch:
            # Get metadata from the batch
            demo_cond = batch[1][:self.num_demos]

        if module.diffusion.pretransform is not None:
            # 计算出潜空间的长度
            latent_samples = audio_samples // module.diffusion.pretransform.downsampling_ratio # e.g., 1024

        # Noise 的长度应该是潜空间的长度
        noise = torch.randn([self.num_demos, module.diffusion.io_channels, latent_samples]).to(module.device)

        # === 关键：构建完整的 demo conditioning（包含 pos_wave）===
        demo_conds = []
        for i in range(self.num_demos):
            raw_cond = self.demo_conditioning[i % len(self.demo_conditioning)]

            # 提取必要字段
            prompt = raw_cond.get("prompt", "")
            
            # 修正：seconds_total 的默认值应该基于 *最大音频长度*
            seconds_total = raw_cond.get("seconds_total", audio_samples / self.sample_rate)
            
            movement_str = raw_cond.get("movement_str")
            motion_type = raw_cond.get("motion_type")
            start_points = raw_cond.get("start_points")
            end_points = raw_cond.get("end_points")
            seconds_start = raw_cond.get("seconds_start", 0)

            print(f'[DEBUG] Generating pos_wave for demo {i}: seconds_total={seconds_total}, seconds_start={seconds_start} cal audiolen {audio_samples / self.sample_rate}')

            # 这就是您期望的逻辑
            num_samples_target = int(seconds_total * self.sample_rate)

            # === 生成 pos_wave ===
            pos_wave = CondPosWave.conditioning_pos_wave(
                movement_str=movement_str,
                motion_type=motion_type,
                start_samples=start_points,
                end_samples=end_points,
                num_samples_target=num_samples_target, # 使用上面正确计算的样本数
                ori_sr=self.sample_rate,
                target_sr=self.sample_rate,
            )  # [T, 3]

            # 裁剪和填充逻辑也需要基于音频长度
            offset = int(seconds_start * self.sample_rate)
            
            # 如果生成的 pos_wave 比期望的音频长度长，就从 offset 开始裁剪
            if len(pos_wave) > num_samples_target:
                 pos_wave = pos_wave[offset:offset + num_samples_target]

            # 如果短了，就填充
            if len(pos_wave) < self.demo_samples:
                pad_len = self.demo_samples - len(pos_wave)
                last_valid_pos = pos_wave[-1:] if len(pos_wave) > 0 else np.zeros((1, 3))
                pad = np.repeat(last_valid_pos, pad_len, axis=0)
                if len(pos_wave) > 0:
                    pad[:, 2] = 0.0
                pos_wave = np.concatenate([pos_wave, pad], axis=0)


            pos_wave = torch.tensor(pos_wave, dtype=torch.float32)

            # 构造完整 info
            info = {
                "prompt": prompt,
                "pos": pos_wave,
                "sample_rate": self.sample_rate,
                "motion_type": motion_type,
                "seconds_start": seconds_start,
                "seconds_total": seconds_total,
            }
            print('pos shape:', pos_wave.shape)
            demo_conds.append(info)





        try:
            print("Getting conditioning")
            with torch.cuda.amp.autocast():
                conditioning = module.diffusion.conditioner(demo_conds, module.device)

            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)
            log_dict = {}

            if self.display_audio_cond:
                audio_inputs = torch.cat([cond["audio"] for cond in demo_conds], dim=0)
                audio_inputs = rearrange(audio_inputs, 'b d n -> d (b n)')

                filename = f'demo_audio_cond_{trainer.global_step:08}.wav'
                audio_inputs = audio_inputs.to(torch.float32).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, audio_inputs, self.sample_rate)
                log_dict[f'demo_audio_cond'] = wandb.Audio(filename, sample_rate=self.sample_rate, caption="Audio conditioning")
                log_dict[f"demo_audio_cond_melspec_left"] = wandb.Image(audio_spectrogram_image(audio_inputs))
                trainer.logger.experiment.log(log_dict)

            for cfg_scale in self.demo_cfg_scales:
                print(f"Generating demo for cfg scale {cfg_scale}")

                with torch.cuda.amp.autocast():
                    model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model
                    if module.diffusion_objective == "v":
                        fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale,
                                       batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale,
                                                      batch_cfg=True)
                    else:
                        raise ValueError(f"Unknown objective: {module.diffusion_objective}")

                    if module.diffusion.pretransform is not None:
                        fakes = module.diffusion.pretransform.decode(fakes)

                # 拼接多 track
                fakes = rearrange(fakes, 'b d n -> d (b n)')
                fakes = fakes.div(torch.max(torch.abs(fakes)) + 1e-8).mul(32767).to(torch.int16).cpu()

                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                torchaudio.save(filename, fakes, self.sample_rate)

                log_dict[f'demo_cfg_{cfg_scale}'] = wandb.Audio(
                    filename, sample_rate=self.sample_rate, caption=f'CFG {cfg_scale}'
                )
                log_dict[f'demo_melspec_cfg_{cfg_scale}'] = wandb.Image(audio_spectrogram_image(fakes))

            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
            
            del fakes

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()


class DiffusionAutoencoderTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a diffusion autoencoder
    '''
    def __init__(
            self,
            model: DiffusionAutoencoder,
            lr: float = 1e-4,
            ema_copy = None,
            use_reconstruction_loss: bool = False
    ):
        super().__init__()

        self.diffae = model
        
        self.diffae_ema = EMA(
            self.diffae,
            ema_model=ema_copy,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1,
            include_online_model=False
        )

        self.lr = lr

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        loss_modules = [
            MSELoss("v",
                    "targets",
                    weight=1.0,
                    name="mse_loss"
            )
        ]

        if model.bottleneck is not None:
            # TODO: Use loss config for configurable bottleneck weights and reconstruction losses
            loss_modules += create_loss_modules_from_bottleneck(model.bottleneck, {})

        self.use_reconstruction_loss = use_reconstruction_loss

        if use_reconstruction_loss:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)

            sample_rate = model.sample_rate

            stft_loss_args = {
                "fft_sizes": scales,
                "hop_sizes": hop_sizes,
                "win_lengths": win_lengths,
                "perceptual_weighting": True
            }

            out_channels = model.out_channels

            if model.pretransform is not None:
                out_channels = model.pretransform.io_channels

            if out_channels == 2:
                self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            else:
                self.sdstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

            loss_modules.append(
                AuralossLoss(self.sdstft, 'audio_reals', 'audio_pred', name='mrstft_loss', weight=0.1), # Reconstruction loss
            )

        self.losses = MultiLoss(loss_modules)

    def configure_optimizers(self):
        return optim.Adam([*self.diffae.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        loss_info["audio_reals"] = reals
        
        if self.diffae.pretransform is not None:
            with torch.no_grad():
                reals = self.diffae.pretransform.encode(reals)

        loss_info["reals"] = reals

        #Encode reals, skipping the pretransform since it was already applied
        latents, encoder_info = self.diffae.encode(reals, return_info=True, skip_pretransform=True)

        loss_info["latents"] = latents
        loss_info.update(encoder_info)

        if self.diffae.decoder is not None:
            latents = self.diffae.decoder(latents)
        
        # Upsample latents to match diffusion length
        if latents.shape[2] != reals.shape[2]:
            latents = F.interpolate(latents, size=reals.shape[2], mode='nearest')

        loss_info["latents_upsampled"] = latents

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffae.diffusion(noised_reals, t, input_concat_cond=latents)
            
            loss_info.update({
                "v": v,
                "targets": targets
            })

            if self.use_reconstruction_loss:
                pred = noised_reals * alphas - v * sigmas

                loss_info["pred"] = pred

                if self.diffae.pretransform is not None:
                    pred = self.diffae.pretransform.decode(pred)
                    loss_info["audio_pred"] = pred

            loss, losses = self.losses(loss_info)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': reals.std(),
            'train/latent_std': latents.std(),
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss
    
    def on_before_zero_grad(self, *args, **kwargs):
        self.diffae_ema.update()

    def export_model(self, path, use_safetensors=False):

        model = self.diffae_ema.ema_model
        
        if use_safetensors:
            save_file(model.state_dict(), path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

class DiffusionAutoencoderDemoCallback(pl.Callback):
    def __init__(
        self, 
        demo_dl, 
        demo_every=2000,
        demo_steps=250,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_steps = demo_steps
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionAutoencoderTrainingWrapper, outputs, batch, batch_idx): 
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step

        demo_reals, _ = next(self.demo_dl)

        # Remove extra dimension added by WebDataset
        if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
            demo_reals = demo_reals[0]

        encoder_input = demo_reals
        
        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad() and torch.cuda.amp.autocast():
            latents = module.diffae_ema.ema_model.encode(encoder_input).float()
            fakes = module.diffae_ema.ema_model.decode(latents, steps=self.demo_steps)

        #Interleave reals and fakes
        reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')

        # Put the demos together
        reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')

        log_dict = {}
        
        filename = f'recon_{trainer.global_step:08}.wav'
        reals_fakes = reals_fakes.to(torch.float32).div(torch.max(torch.abs(reals_fakes))).mul(32767).to(torch.int16).cpu()
        torchaudio.save(filename, reals_fakes, self.sample_rate)

        log_dict[f'recon'] = wandb.Audio(filename,
                                            sample_rate=self.sample_rate,
                                            caption=f'Reconstructed')

        log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents)
        log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

        log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reals_fakes))

        if module.diffae_ema.ema_model.pretransform is not None:
            with torch.no_grad() and torch.cuda.amp.autocast():
                initial_latents = module.diffae_ema.ema_model.pretransform.encode(encoder_input)
                first_stage_fakes = module.diffae_ema.ema_model.pretransform.decode(initial_latents)
                first_stage_fakes = rearrange(first_stage_fakes, 'b d n -> d (b n)')
                first_stage_fakes = first_stage_fakes.to(torch.float32).mul(32767).to(torch.int16).cpu()
                first_stage_filename = f'first_stage_{trainer.global_step:08}.wav'
                torchaudio.save(first_stage_filename, first_stage_fakes, self.sample_rate)

                log_dict[f'first_stage_latents'] = wandb.Image(tokens_spectrogram_image(initial_latents))

                log_dict[f'first_stage'] = wandb.Audio(first_stage_filename,
                                            sample_rate=self.sample_rate,
                                            caption=f'First Stage Reconstructed')
                
                log_dict[f'first_stage_melspec_left'] = wandb.Image(audio_spectrogram_image(first_stage_fakes))
                

        trainer.logger.experiment.log(log_dict)

def create_source_mixture(reals, num_sources=2):
    # Create a fake mixture source by mixing elements from the training batch together with random offsets
    source = torch.zeros_like(reals)
    for i in range(reals.shape[0]):
        sources_added = 0
        
        js = list(range(reals.shape[0]))
        random.shuffle(js)
        for j in js:
            if i == j or (i != j and sources_added < num_sources):
                # Randomly offset the mixed element between 0 and the length of the source
                seq_len = reals.shape[2]
                offset = random.randint(0, seq_len-1)
                source[i, :, offset:] += reals[j, :, :-offset]
                if i == j:
                    # If this is the real one, shift the reals as well to ensure alignment
                    new_reals = torch.zeros_like(reals[i])
                    new_reals[:, offset:] = reals[i, :, :-offset]
                    reals[i] = new_reals
                sources_added += 1

    return source

