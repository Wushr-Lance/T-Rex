"""Finger-aware hand-wise tactile VAE encoder."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T]
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)


class FingerAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, C]
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class HandWiseTactileEncoder(nn.Module):
    """Encode [B, T, 5, 6] into VAE posterior parameters [B, latent_dim]."""

    def __init__(
        self,
        window: int = 16,
        per_finger_dim: int = 6,
        n_fingers: int = 5,
        hidden_channels: int = 128,
        bottleneck_channels: int = 256,
        latent_dim: int = 256,
        n_strided_blocks: int = 2,
        temporal_pool: str = "attn",
        use_finger_embed: bool = True,
    ):
        super().__init__()
        if temporal_pool not in ("attn", "flatten_mlp"):
            raise ValueError("temporal_pool must be 'attn' or 'flatten_mlp'")
        self.window = int(window)
        self.n_fingers = int(n_fingers)
        self.latent_dim = int(latent_dim)
        self.temporal_pool = temporal_pool
        self.use_finger_embed = bool(use_finger_embed)

        self.stem = _conv_block(per_finger_dim, hidden_channels, kernel=5, stride=1)
        self.finger_embed = (
            nn.Embedding(n_fingers, hidden_channels) if self.use_finger_embed else None
        )

        blocks: List[nn.Module] = []
        cur_T = self.window
        cur_ch = hidden_channels
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            out_ch = bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels
            blocks.append(_conv_block(cur_ch, out_ch, kernel=5, stride=stride))
            cur_ch = out_ch
            cur_T = cur_T // stride if stride > 1 else cur_T
        self.strided = nn.Sequential(*blocks)
        self._bottleneck_T = cur_T
        self.proj = nn.Conv1d(cur_ch, bottleneck_channels, kernel_size=3, padding=1)

        if temporal_pool == "attn":
            self.temporal_pooler = TemporalAttentionPool(bottleneck_channels)
        else:
            self.temporal_pooler = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(bottleneck_channels * self._bottleneck_T, bottleneck_channels),
                nn.LayerNorm(bottleneck_channels),
                nn.GELU(),
            )

        self.finger_pooler = FingerAttentionPool(bottleneck_channels)
        self.mu_head = nn.Linear(bottleneck_channels, latent_dim)
        self.logvar_head = nn.Linear(bottleneck_channels, latent_dim)

    @property
    def bottleneck_T(self) -> int:
        return self._bottleneck_T

    def forward(self, f6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """f6: [B, T=16, 5, 6] -> (mu, logvar): [B, latent_dim]."""
        B, T, F, D = f6.shape
        if T != self.window:
            raise ValueError(f"Encoder built for window={self.window}, got T={T}")
        if F != self.n_fingers:
            raise ValueError(f"Expected {self.n_fingers} fingers, got {F}")

        x = f6.permute(0, 2, 1, 3).contiguous()       # [B, 5, T, 6]
        x = x.reshape(B * F, T, D).transpose(1, 2)    # [B*5, 6, T]
        x = self.stem(x)                              # [B*5, hidden, T]

        if self.finger_embed is not None:
            ids = torch.arange(F, device=f6.device).repeat(B)
            x = x + self.finger_embed(ids).unsqueeze(-1)

        x = self.strided(x)
        x = self.proj(x)                              # [B*5, bottleneck, T_bn]
        finger_feat = self.temporal_pooler(x)         # [B*5, bottleneck]
        finger_feat = finger_feat.reshape(B, F, -1)   # [B, 5, bottleneck]
        hand_feat = self.finger_pooler(finger_feat)   # [B, bottleneck]
        return self.mu_head(hand_feat), self.logvar_head(hand_feat)

