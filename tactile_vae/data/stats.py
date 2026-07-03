"""T-Rex F6 normalization stats for hand-wise tactile VAE training.

The T-Rex trainers normalize tactile F6 with q01/q99 percentile min-max stats
over all 10 fingers. This module keeps that convention, but also exposes
hand-specific normalization so a [T, 5, 6] one-hand sample uses the correct
30 stats entries.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


_F6_DIM = 60
_HAND_DIM = 30


@dataclass
class TacF6Stats:
    tacf6_min: np.ndarray
    tacf6_max: np.ndarray
    tacf6_mask: np.ndarray
    stats_source: str = "manifest"

    @classmethod
    def from_data_root(cls, data_root: str) -> "TacF6Stats":
        manifest_paths = sorted(
            glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
        )
        if not manifest_paths:
            raise FileNotFoundError(f"No pretrain_manifest.json under {data_root}/*/")

        all_q01, all_q99 = [], []
        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            block = manifest.get("statistics", {}).get("tactile_f6")
            if block:
                all_q01.append(np.array(block["q01"], dtype=np.float32))
                all_q99.append(np.array(block["q99"], dtype=np.float32))

        if all_q01:
            tacf6_min = np.min(np.stack(all_q01), axis=0)
            tacf6_max = np.max(np.stack(all_q99), axis=0)
        else:
            tacf6_min = np.full(_F6_DIM, -1.0, dtype=np.float32)
            tacf6_max = np.full(_F6_DIM, +1.0, dtype=np.float32)

        return cls._checked(tacf6_min, tacf6_max, np.ones(_F6_DIM, dtype=bool), "manifest")

    @classmethod
    def from_lerobot_root(cls, root: str) -> Optional["TacF6Stats"]:
        """Load q01/q99 sidecar from a converted T-Rex LeRobot root if present."""
        sidecar = os.path.join(root, "meta", "trex_norm_stats.json")
        if not os.path.isfile(sidecar):
            return None
        with open(sidecar, "r") as f:
            payload = json.load(f)
        block = payload[next(iter(payload))]["tactile_f6"]
        return cls._checked(
            np.array(block["q01"], dtype=np.float32),
            np.array(block["q99"], dtype=np.float32),
            np.array(block.get("mask", [True] * _F6_DIM), dtype=bool),
            "trex_norm_stats",
        )

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "TacF6Stats":
        """Compute temporary q01/q99 stats from subset samples.

        samples: [..., 10, 6] or [..., 60].
        """
        flat = samples.reshape(-1, _F6_DIM).astype(np.float32, copy=False)
        q01 = np.quantile(flat, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(flat, 0.99, axis=0).astype(np.float32)
        return cls._checked(q01, q99, np.ones(_F6_DIM, dtype=bool), "subset_computed")

    @classmethod
    def _checked(
        cls,
        tacf6_min: np.ndarray,
        tacf6_max: np.ndarray,
        tacf6_mask: np.ndarray,
        stats_source: str,
    ) -> "TacF6Stats":
        if tacf6_min.shape[0] != _F6_DIM or tacf6_max.shape[0] != _F6_DIM:
            raise ValueError(
                f"Expected F6 stats dim {_F6_DIM}, got "
                f"{tacf6_min.shape}/{tacf6_max.shape}")
        return cls(
            tacf6_min=tacf6_min.astype(np.float32, copy=False),
            tacf6_max=tacf6_max.astype(np.float32, copy=False),
            tacf6_mask=tacf6_mask.astype(bool, copy=False),
            stats_source=stats_source,
        )

    def normalize_full(self, x: np.ndarray) -> np.ndarray:
        """Normalize full bimanual F6 with shape [..., 10, 6] or [..., 60]."""
        orig_shape = x.shape
        flat = x.reshape(-1, _F6_DIM).astype(np.float32, copy=False)
        denom = (self.tacf6_max - self.tacf6_min) + 1e-8
        normed = np.clip(2.0 * (flat - self.tacf6_min) / denom - 1.0, -1.0, 1.0)
        out = np.where(self.tacf6_mask, normed, flat)
        return out.reshape(orig_shape)

    def normalize_hand(self, x: np.ndarray, hand: int) -> np.ndarray:
        """Normalize one hand F6 with shape [..., 5, 6] or [..., 30].

        hand=0 uses stats entries 0:30; hand=1 uses entries 30:60.
        """
        if hand not in (0, 1):
            raise ValueError(f"hand must be 0 or 1, got {hand}")
        orig_shape = x.shape
        flat = x.reshape(-1, _HAND_DIM).astype(np.float32, copy=False)
        sl = slice(hand * _HAND_DIM, (hand + 1) * _HAND_DIM)
        vmin = self.tacf6_min[sl]
        vmax = self.tacf6_max[sl]
        mask = self.tacf6_mask[sl]
        denom = (vmax - vmin) + 1e-8
        normed = np.clip(2.0 * (flat - vmin) / denom - 1.0, -1.0, 1.0)
        out = np.where(mask, normed, flat)
        return out.reshape(orig_shape)

    def denormalize_hand(self, x_norm: np.ndarray, hand: int) -> np.ndarray:
        if hand not in (0, 1):
            raise ValueError(f"hand must be 0 or 1, got {hand}")
        orig_shape = x_norm.shape
        flat = x_norm.reshape(-1, _HAND_DIM).astype(np.float32, copy=False)
        sl = slice(hand * _HAND_DIM, (hand + 1) * _HAND_DIM)
        vmin = self.tacf6_min[sl]
        vmax = self.tacf6_max[sl]
        mask = self.tacf6_mask[sl]
        out = (flat + 1.0) * 0.5 * (vmax - vmin) + vmin
        out = np.where(mask, out, flat)
        return out.reshape(orig_shape)

    def to_dict(self) -> dict:
        return {
            "tacf6_min": self.tacf6_min.tolist(),
            "tacf6_max": self.tacf6_max.tolist(),
            "tacf6_mask": self.tacf6_mask.tolist(),
            "stats_source": self.stats_source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TacF6Stats":
        return cls._checked(
            np.array(d["tacf6_min"], dtype=np.float32),
            np.array(d["tacf6_max"], dtype=np.float32),
            np.array(d["tacf6_mask"], dtype=bool),
            d.get("stats_source", "unknown"),
        )

