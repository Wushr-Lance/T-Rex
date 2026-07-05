"""Datasets for hand-wise tactile VAE training."""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .stats import TacF6Stats


def _scan_episodes(data_root: str) -> Tuple[List[str], List[int]]:
    manifest_paths = sorted(
        glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
    )
    if not manifest_paths:
        raise FileNotFoundError(f"No pretrain_manifest.json under {data_root}/*/")

    ep_dirs: List[str] = []
    n_frames: List[int] = []
    for mp in manifest_paths:
        with open(mp, "r") as f:
            manifest = json.load(f)
        for ep in manifest["episodes"]:
            ep_dirs.append(ep["episode_dir"])
            n_frames.append(int(ep["num_frames"]))
    return ep_dirs, n_frames


def _subsample_indices(source_window: int, input_window: int, subsample_stride: int) -> np.ndarray:
    idx = np.arange(input_window, dtype=np.int64) * int(subsample_stride)
    if idx[-1] >= source_window:
        raise ValueError(
            f"input_window={input_window} with subsample_stride={subsample_stride} "
            f"needs frame {idx[-1]}, but source_window={source_window}")
    return idx


def _as_full_f6(arr: np.ndarray) -> np.ndarray:
    """Return tactile F6 as [T, 10, 6] from either [T, 10, 6] or [T, 60]."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (10, 6):
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 60:
        return arr.reshape(arr.shape[0], 10, 6)
    raise ValueError(f"Expected tactile F6 [T,10,6] or [T,60], got {arr.shape}")


class F6ChunkDataset(Dataset):
    """Merged-HDF5 T-Rex tactile chunks.

    Each item reads a raw [source_window, 10, 6] chunk, selects one hand, and
    uniformly subsamples it to [input_window, 5, 6].
    """

    def __init__(
        self,
        data_root: str,
        source_window: int = 64,
        input_window: int = 16,
        subsample_stride: int = 4,
        stride: int = 4,
        stats: Optional[TacF6Stats] = None,
        episodes: Optional[Tuple[List[str], List[int]]] = None,
        drop_short: bool = True,
    ):
        self.data_root = data_root
        self.source_window = int(source_window)
        self.input_window = int(input_window)
        self.subsample_stride = int(subsample_stride)
        self.stride = max(1, int(stride))
        self.subsample_idx = _subsample_indices(
            self.source_window, self.input_window, self.subsample_stride)
        self.stats = stats if stats is not None else TacF6Stats.from_data_root(data_root)

        if episodes is None:
            ep_dirs, n_frames = _scan_episodes(data_root)
        else:
            ep_dirs, n_frames = episodes

        if drop_short:
            kept = [(d, n) for d, n in zip(ep_dirs, n_frames) if n >= self.source_window]
            if not kept:
                raise RuntimeError(
                    f"All episodes are shorter than source_window={self.source_window}.")
            ep_dirs = [d for d, _ in kept]
            n_frames = [n for _, n in kept]

        self._episode_dirs = ep_dirs
        self._n_frames = n_frames
        windows_per_ep = [
            max(0, (n - self.source_window) // self.stride + 1) for n in n_frames
        ]
        self._windows_per_ep = np.array(windows_per_ep, dtype=np.int64)
        self._cum_windows = np.cumsum(self._windows_per_ep)
        self._total_windows_per_hand = int(self._cum_windows[-1]) if len(self._cum_windows) else 0
        self._total = self._total_windows_per_hand * 2

        self._cache_ep_idx = -1
        self._cache_f6: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self._total

    @property
    def num_episodes(self) -> int:
        return len(self._episode_dirs)

    def _load_ep_f6(self, ep_idx: int) -> Optional[np.ndarray]:
        if ep_idx == self._cache_ep_idx:
            return self._cache_f6
        self._cache_ep_idx = ep_idx
        self._cache_f6 = None
        path = os.path.join(self._episode_dirs[ep_idx], "pretrain.hdf5")
        if not os.path.isfile(path):
            return None
        try:
            with h5py.File(path, "r") as f:
                if "tactile_f6" not in f:
                    return None
                self._cache_f6 = f["tactile_f6"][:].astype(np.float32, copy=False)
        except Exception:
            self._cache_f6 = None
        return self._cache_f6

    def _decode_idx(self, idx: int) -> Tuple[int, int, int]:
        hand = idx // self._total_windows_per_hand
        within = idx % self._total_windows_per_hand
        ep_idx = int(np.searchsorted(self._cum_windows, within, side="right"))
        prev = int(self._cum_windows[ep_idx - 1]) if ep_idx > 0 else 0
        frame_start = (within - prev) * self.stride
        return ep_idx, frame_start, hand

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, frame_start, hand = self._decode_idx(idx)
        f6 = self._load_ep_f6(ep_idx)
        if f6 is None or frame_start + self.source_window > f6.shape[0]:
            raw_hand = np.zeros((self.input_window, 5, 6), dtype=np.float32)
        else:
            raw_chunk = f6[frame_start: frame_start + self.source_window]  # [64,10,6]
            raw_hand_64 = raw_chunk[:, hand * 5:(hand + 1) * 5, :]
            raw_hand = raw_hand_64[self.subsample_idx]                    # [16,5,6]

        f6_normed = self.stats.normalize_hand(raw_hand, hand).astype(np.float32, copy=False)
        magnitude = float(np.linalg.norm(raw_hand))
        return {
            "f6": torch.from_numpy(f6_normed),
            "raw_f6": torch.from_numpy(raw_hand.astype(np.float32, copy=False)),
            "magnitude": torch.tensor(magnitude, dtype=torch.float32),
            "ep_idx": ep_idx,
            "frame": frame_start,
            "hand": hand,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        def _scalar(v):
            return int(v.item()) if torch.is_tensor(v) else int(v)

        return {
            "f6": torch.stack([b["f6"] for b in batch], dim=0),
            "raw_f6": torch.stack([b["raw_f6"] for b in batch], dim=0),
            "magnitude": torch.stack([b["magnitude"] for b in batch], dim=0),
            "ep_idx": torch.tensor([_scalar(b["ep_idx"]) for b in batch], dtype=torch.long),
            "frame": torch.tensor([_scalar(b["frame"]) for b in batch], dtype=torch.long),
            "hand": torch.tensor([_scalar(b["hand"]) for b in batch], dtype=torch.long),
        }


class LeRobotF6ChunkDataset(Dataset):
    """Small sanity-check loader for T-Rex LeRobot/HF roots.

    This path is intentionally lightweight. It exists so a subset downloaded
    without videos can run smoke training when the `lerobot` package is present.
    """

    def __init__(
        self,
        root: str,
        repo_id: str,
        source_window: int = 64,
        input_window: int = 16,
        subsample_stride: int = 4,
        stats: Optional[TacF6Stats] = None,
        episodes: Optional[List[int]] = None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = root
        self.repo_id = repo_id
        self.source_window = int(source_window)
        self.input_window = int(input_window)
        self.subsample_stride = int(subsample_stride)
        self.subsample_idx = _subsample_indices(
            self.source_window, self.input_window, self.subsample_stride)
        self.key = self._detect_tactile_key(root)
        fps = self._read_fps(root)
        offsets = [(i - (self.source_window - 1)) / fps for i in range(self.source_window)]
        self.ds = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps={self.key: offsets},
        )
        self.stats = stats if stats is not None else self._build_stats_from_subset()

    @staticmethod
    def _read_fps(root: str) -> float:
        info_path = os.path.join(root, "meta", "info.json")
        if not os.path.isfile(info_path):
            return 30.0
        with open(info_path, "r") as f:
            return float(json.load(f).get("fps", 30.0))

    @staticmethod
    def _detect_tactile_key(root: str) -> str:
        info_path = os.path.join(root, "meta", "info.json")
        if os.path.isfile(info_path):
            with open(info_path, "r") as f:
                feats = json.load(f).get("features", {})
            for key in ("observation.tactile_f6", "observation.tactile_force"):
                if key in feats:
                    return key
        return "observation.tactile_f6"

    def _build_stats_from_subset(self, max_items: int = 512) -> TacF6Stats:
        sidecar = TacF6Stats.from_lerobot_root(self.root)
        if sidecar is not None:
            return sidecar
        samples = []
        n = min(len(self.ds), max_items)
        for i in range(n):
            arr = _as_full_f6(np.asarray(self.ds[i][self.key], dtype=np.float32))
            samples.append(arr[-1])
        return TacF6Stats.from_samples(np.stack(samples, axis=0))

    def __len__(self) -> int:
        return len(self.ds) * 2

    def __getitem__(self, idx: int) -> Dict:
        hand = idx // len(self.ds)
        item = self.ds[idx % len(self.ds)]
        raw_full = _as_full_f6(np.asarray(item[self.key], dtype=np.float32))  # [64,10,6]
        raw_hand = raw_full[self.subsample_idx, hand * 5:(hand + 1) * 5, :]
        f6_normed = self.stats.normalize_hand(raw_hand, hand).astype(np.float32, copy=False)
        return {
            "f6": torch.from_numpy(f6_normed),
            "raw_f6": torch.from_numpy(raw_hand.astype(np.float32, copy=False)),
            "magnitude": torch.tensor(float(np.linalg.norm(raw_hand)), dtype=torch.float32),
            "ep_idx": torch.tensor(-1, dtype=torch.long),
            "frame": torch.tensor(idx % len(self.ds), dtype=torch.long),
            "hand": torch.tensor(hand, dtype=torch.long),
        }

    collate_fn = staticmethod(F6ChunkDataset.collate_fn)


class ParquetF6ChunkDataset(Dataset):
    """Local no-video LeRobot parquet fallback for sanity training.

    Reads downloaded `data/**/*.parquet` files directly and builds the same
    hand-wise [16, 5, 6] windows without relying on the `lerobot` package.
    """

    def __init__(
        self,
        root: str,
        source_window: int = 64,
        input_window: int = 16,
        subsample_stride: int = 4,
        stride: int = 4,
        stats: Optional[TacF6Stats] = None,
        episodes: Optional[List[int]] = None,
    ):
        import pandas as pd

        self.root = root
        self.source_window = int(source_window)
        self.input_window = int(input_window)
        self.subsample_stride = int(subsample_stride)
        self.stride = max(1, int(stride))
        self.subsample_idx = _subsample_indices(
            self.source_window, self.input_window, self.subsample_stride)
        self.stats = stats if stats is not None else (TacF6Stats.from_lerobot_root(root) or self._stats_from_data())

        files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No parquet files under {root}/data")
        frames = []
        for path in files:
            df = pd.read_parquet(path, columns=[
                "observation.tactile_force",
                "episode_index",
                "frame_index",
            ])
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        if episodes is not None:
            df = df[df["episode_index"].isin(episodes)]
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        self._episodes: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for ep, group in df.groupby("episode_index", sort=True):
            f6 = np.stack(group["observation.tactile_force"].to_numpy()).astype(np.float32)
            f6 = _as_full_f6(f6)
            frames_idx = group["frame_index"].to_numpy(dtype=np.int64)
            if f6.shape[0] >= self.source_window:
                self._episodes.append((int(ep), f6, frames_idx))
        if not self._episodes:
            raise RuntimeError(f"No episodes with at least {self.source_window} frames in {root}")

        windows_per_ep = [
            max(0, (f6.shape[0] - self.source_window) // self.stride + 1)
            for _, f6, _ in self._episodes
        ]
        self._windows_per_ep = np.array(windows_per_ep, dtype=np.int64)
        self._cum_windows = np.cumsum(self._windows_per_ep)
        self._total_windows_per_hand = int(self._cum_windows[-1])
        self._total = self._total_windows_per_hand * 2

    def _stats_from_data(self, max_frames: int = 20000) -> TacF6Stats:
        import pandas as pd

        files = sorted(glob.glob(os.path.join(self.root, "data", "**", "*.parquet"), recursive=True))
        samples = []
        remaining = max_frames
        for path in files:
            df = pd.read_parquet(path, columns=["observation.tactile_force"])
            take = min(len(df), remaining)
            samples.append(np.stack(df["observation.tactile_force"].iloc[:take].to_numpy()))
            remaining -= take
            if remaining <= 0:
                break
        return TacF6Stats.from_samples(np.concatenate(samples, axis=0))

    def __len__(self) -> int:
        return self._total

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def _decode_idx(self, idx: int) -> Tuple[int, int, int]:
        hand = idx // self._total_windows_per_hand
        within = idx % self._total_windows_per_hand
        ep_idx = int(np.searchsorted(self._cum_windows, within, side="right"))
        prev = int(self._cum_windows[ep_idx - 1]) if ep_idx > 0 else 0
        frame_start = (within - prev) * self.stride
        return ep_idx, frame_start, hand

    def __getitem__(self, idx: int) -> Dict:
        ep_idx, frame_start, hand = self._decode_idx(idx)
        ep_id, f6, frames_idx = self._episodes[ep_idx]
        raw_chunk = f6[frame_start: frame_start + self.source_window]
        raw_hand_64 = raw_chunk[:, hand * 5:(hand + 1) * 5, :]
        raw_hand = raw_hand_64[self.subsample_idx]
        f6_normed = self.stats.normalize_hand(raw_hand, hand).astype(np.float32, copy=False)
        return {
            "f6": torch.from_numpy(f6_normed),
            "raw_f6": torch.from_numpy(raw_hand.astype(np.float32, copy=False)),
            "magnitude": torch.tensor(float(np.linalg.norm(raw_hand)), dtype=torch.float32),
            "ep_idx": int(ep_id),
            "frame": int(frames_idx[frame_start]),
            "hand": hand,
        }

    collate_fn = staticmethod(F6ChunkDataset.collate_fn)


def build_train_val_datasets(
    data_root: str,
    source_window: int = 64,
    input_window: int = 16,
    subsample_stride: int = 4,
    stride: int = 4,
    val_ratio: float = 0.02,
    seed: int = 42,
    stats: Optional[TacF6Stats] = None,
) -> Tuple[F6ChunkDataset, F6ChunkDataset, TacF6Stats]:
    if stats is None:
        stats = TacF6Stats.from_data_root(data_root)

    ep_dirs, n_frames = _scan_episodes(data_root)
    n_eps = len(ep_dirs)
    if n_eps < 2:
        raise RuntimeError(f"Need >=2 episodes for train/val split; have {n_eps}.")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(round(n_eps * val_ratio)))
    val_idx = sorted(perm[:n_val].tolist())
    tr_idx = sorted(perm[n_val:].tolist())
    val_eps = ([ep_dirs[i] for i in val_idx], [n_frames[i] for i in val_idx])
    tr_eps = ([ep_dirs[i] for i in tr_idx], [n_frames[i] for i in tr_idx])

    train_ds = F6ChunkDataset(
        data_root=data_root,
        source_window=source_window,
        input_window=input_window,
        subsample_stride=subsample_stride,
        stride=stride,
        stats=stats,
        episodes=tr_eps,
    )
    val_ds = F6ChunkDataset(
        data_root=data_root,
        source_window=source_window,
        input_window=input_window,
        subsample_stride=subsample_stride,
        stride=stride,
        stats=stats,
        episodes=val_eps,
    )
    return train_ds, val_ds, stats
