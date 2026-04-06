import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim=2, M=64, sigma=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.M = M
        self.sigma = sigma
        self.register_buffer("B", torch.randn(M, input_dim) * sigma)

    def forward(self, x):
        """
        x: [B, T, 2]
        return: [B, T, 2M]
        """
        proj = 2 * torch.pi * x.to(self.B.device) @ self.B
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return features


class PosDownsampleEncoder(nn.Module):
    def __init__(self, in_dim=128, base_channels=128, c_mults=[1, 2, 4, 8], strides=[2, 4, 4, 8], latent_dim=64):
        super().__init__()
        layers = []
        in_channels = in_dim
        for mult, stride in zip(c_mults, strides):
            out_channels = base_channels * mult
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
                nn.GroupNorm(9, out_channels),
                nn.SiLU(),
            ]
            in_channels = out_channels

        layers += [nn.Conv1d(in_channels, latent_dim, kernel_size=3, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 1, 2)  # [B, T, C] → [B, C, T]
        return self.layers(x)


class PosEmbd(nn.Module):
    def __init__(self, fourier_M=64, latent_dim=64, mask_hidden_dim=64):
        super().__init__()
        self.fourier = FourierFeatureEncoder(input_dim=2, M=fourier_M)

        # Downsample encoder
        self.down = PosDownsampleEncoder(
            in_dim=128,
            base_channels=64,
            c_mults=[1, 2, 4, 8, 16],
            strides=[2, 4, 4, 8, 8],
            latent_dim=64
        )
    def forward(self, pos):
        """
        pos: [B, T, 3] = [x, y, mask]
        return: [B, latent_dim, T_down]
        """
        xy = pos[..., :2]  # [B, T, 2]
        mask = pos[..., 2:] # [B, T, 1]

        ffeat = self.fourier(xy)  # [B, T, 2M]

        # mask modulation
        gated_features = ffeat * mask  # 广播：(B,T,2M) * (B,T,1)

        # optional residual term (retain structure even if mask=0)
        gated_features = gated_features + 0.1 * ffeat


        latent = self.down(gated_features)
        return latent

