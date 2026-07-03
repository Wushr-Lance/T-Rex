"""Full hand-wise tactile VAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .decoder import HandWiseTactileDecoder
from .encoder import HandWiseTactileEncoder


@dataclass
class TactileVAEConfig:
    source_window: int = 64
    input_window: int = 16
    subsample_stride: int = 4
    per_finger_dim: int = 6
    n_fingers: int = 5
    hidden_channels: int = 128
    bottleneck_channels: int = 256
    bottleneck_T: int = 4
    latent_dim: int = 256
    n_strided_blocks: int = 2
    temporal_pool: str = "attn"
    use_finger_embed: bool = True
    use_time_embed: bool = True
    beta_kl: float = 1e-3
    use_magnitude_weight: bool = True
    weight_alpha: float = 2.0
    weight_tau: float = 4.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "TactileVAEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TactileVAE(nn.Module):
    def __init__(self, cfg: TactileVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = HandWiseTactileEncoder(
            window=cfg.input_window,
            per_finger_dim=cfg.per_finger_dim,
            n_fingers=cfg.n_fingers,
            hidden_channels=cfg.hidden_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            latent_dim=cfg.latent_dim,
            n_strided_blocks=cfg.n_strided_blocks,
            temporal_pool=cfg.temporal_pool,
            use_finger_embed=cfg.use_finger_embed,
        )
        if self.encoder.bottleneck_T != cfg.bottleneck_T:
            raise ValueError(
                f"Config bottleneck_T={cfg.bottleneck_T}, but encoder implies "
                f"{self.encoder.bottleneck_T}.")
        self.decoder = HandWiseTactileDecoder(
            window=cfg.input_window,
            per_finger_dim=cfg.per_finger_dim,
            n_fingers=cfg.n_fingers,
            hidden_channels=cfg.hidden_channels,
            bottleneck_channels=cfg.bottleneck_channels,
            latent_dim=cfg.latent_dim,
            bottleneck_T=cfg.bottleneck_T,
            n_strided_blocks=cfg.n_strided_blocks,
            use_finger_embed=cfg.use_finger_embed,
            use_time_embed=cfg.use_time_embed,
        )

    def encode(self, f6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(f6)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _recon_weight(self, magnitude: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return 1.0 + cfg.weight_alpha * torch.sigmoid(magnitude / cfg.weight_tau - 1.0)

    def forward(
        self,
        f6: torch.Tensor,
        magnitude: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(f6)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decode(z)

        per_sample_recon = (recon - f6).pow(2).mean(dim=[1, 2, 3])
        if self.cfg.use_magnitude_weight and magnitude is not None:
            w = self._recon_weight(magnitude.to(per_sample_recon.device))
            recon_loss = (per_sample_recon * w).sum() / (w.sum() + 1e-8)
        else:
            recon_loss = per_sample_recon.mean()

        kl_per_sample = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
        kl_loss = kl_per_sample.mean()
        total_loss = recon_loss + self.cfg.beta_kl * kl_loss

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "per_sample_recon": per_sample_recon.detach(),
            "kl_per_sample": kl_per_sample.detach(),
        }

