"""Finger-aware hand-wise tactile VAE decoder."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upconv_block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    if stride > 1:
        layer = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=stride - 1,
        )
    else:
        layer = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=pad)
    return nn.Sequential(
        layer,
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class HandWiseTactileDecoder(nn.Module):
    """Decode one hand-wise latent into [B, T, 5, 6]."""

    def __init__(
        self,
        window: int = 16,
        per_finger_dim: int = 6,
        n_fingers: int = 5,
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        latent_dim: int = 256,
        bottleneck_T: int = 4,
        n_strided_blocks: int = 2,
        use_finger_embed: bool = True,
        use_time_embed: bool = True,
    ):
        super().__init__()
        self.window = int(window)
        self.n_fingers = int(n_fingers)
        self.bottleneck_channels = int(bottleneck_channels)
        self.bottleneck_T = int(bottleneck_T)
        self.use_finger_embed = bool(use_finger_embed)
        self.use_time_embed = bool(use_time_embed)

        self.latent_to_tokens = nn.Linear(
            latent_dim,
            n_fingers * bottleneck_channels * self.bottleneck_T,
        )
        self.finger_embed = (
            nn.Embedding(n_fingers, bottleneck_channels) if self.use_finger_embed else None
        )
        self.time_embed = (
            nn.Parameter(torch.randn(1, 1, bottleneck_channels, self.bottleneck_T) * 0.02)
            if self.use_time_embed else None
        )

        cur_T = window
        strides: List[int] = []
        cur_ch_chain = [hidden_channels]
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            strides.append(stride)
            cur_ch_chain.append(
                bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels)
            if stride > 1:
                cur_T //= stride
        if cur_T != self.bottleneck_T:
            raise ValueError(
                f"Decoder bottleneck_T={self.bottleneck_T}, but window={window} and "
                f"n_strided_blocks={n_strided_blocks} imply {cur_T}.")

        blocks: List[nn.Module] = []
        in_ch = bottleneck_channels
        rev_strides = list(reversed(strides))
        rev_ch_chain = list(reversed(cur_ch_chain))
        for i, st in enumerate(rev_strides):
            out_ch = rev_ch_chain[i + 1]
            blocks.append(_upconv_block(in_ch, out_ch, kernel=5, stride=st))
            in_ch = out_ch
        self.up_strided = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_channels, per_finger_dim, kernel_size=5, padding=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] -> recon: [B, T, 5, 6]."""
        B = z.shape[0]
        Fng = self.n_fingers

        x = self.latent_to_tokens(z)
        x = x.reshape(B, Fng, self.bottleneck_channels, self.bottleneck_T)

        if self.finger_embed is not None:
            ids = torch.arange(Fng, device=z.device)
            x = x + self.finger_embed(ids).view(1, Fng, self.bottleneck_channels, 1)
        if self.time_embed is not None:
            x = x + self.time_embed

        x = x.reshape(B * Fng, self.bottleneck_channels, self.bottleneck_T)
        x = self.up_strided(x)
        x = self.head(x)                                           # [B*5, 6, T]

        if x.shape[-1] != self.window:
            if x.shape[-1] > self.window:
                x = x[..., : self.window]
            else:
                x = F.pad(x, (0, self.window - x.shape[-1]))

        x = x.transpose(1, 2).contiguous()                         # [B*5, T, 6]
        x = x.reshape(B, Fng, self.window, -1).permute(0, 2, 1, 3)
        return x.contiguous()                                      # [B, T, 5, 6]

