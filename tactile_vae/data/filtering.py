"""Utilities for visualizing tactile-force filtering decisions.

The training dataset treats a bimanual 64-frame tactile window as two
hand-wise samples. This module keeps the same unit: all filter decisions are
made on one hand, five fingers, over a contiguous raw-frame window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


FILTER_METHODS = ("any_finger", "mean5", "sustained_any", "window_mean5")
HAND_NAMES = ("left", "right")


@dataclass(frozen=True)
class WindowFilterResult:
    """Filter scores and pass/fail values for one hand window."""

    threshold: float
    scores: Mapping[str, float]
    passes: Mapping[str, bool]
    active_any_frames: int
    active_mean5_frames: int
    consecutive_any_frames: int


@dataclass(frozen=True)
class WindowRecord:
    """One threshold/method decision for a hand-wise training candidate."""

    episode_index: int
    frame_start: int
    hand: int
    threshold: float
    method: str
    score: float
    passed: bool
    active_any_frames: int
    active_mean5_frames: int
    consecutive_any_frames: int


def as_full_f6(arr: np.ndarray) -> np.ndarray:
    """Return tactile F6 as ``[T, 10, 6]`` from ``[T, 10, 6]`` or ``[T, 60]``."""

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (10, 6):
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 60:
        return arr.reshape(arr.shape[0], 10, 6)
    raise ValueError(f"Expected tactile F6 [T,10,6] or [T,60], got {arr.shape}")


def finger_force_norms(full_f6: np.ndarray) -> np.ndarray:
    """Compute per-finger 3D force norms ``[T, 10]``.

    Only ``Fx, Fy, Fz`` are used. Moment channels are intentionally ignored for
    thresholding so thresholds have force semantics.
    """

    full = as_full_f6(full_f6)
    return np.linalg.norm(full[:, :, :3], axis=-1).astype(np.float32, copy=False)


def split_hand_forces(force_norms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split bimanual finger-force norms into left and right ``[T, 5]`` arrays."""

    force_norms = np.asarray(force_norms, dtype=np.float32)
    if force_norms.ndim != 2 or force_norms.shape[1] != 10:
        raise ValueError(f"Expected force norms [T,10], got {force_norms.shape}")
    return force_norms[:, :5], force_norms[:, 5:10]


def window_start_indices(n_frames: int, source_window: int = 64, stride: int = 4) -> np.ndarray:
    """Return valid contiguous window starts for the training-style sampler."""

    n_frames = int(n_frames)
    source_window = int(source_window)
    stride = max(1, int(stride))
    if n_frames < source_window:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, n_frames - source_window + 1, stride, dtype=np.int64)


def max_true_run(mask: np.ndarray) -> int:
    """Length of the longest contiguous True run in a 1D boolean mask."""

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def evaluate_hand_window(
    hand_forces: np.ndarray,
    threshold: float,
    min_active_frames: int = 1,
    min_consecutive_frames: int = 8,
) -> WindowFilterResult:
    """Evaluate all filter methods for one ``[source_window, 5]`` hand window."""

    hand = np.asarray(hand_forces, dtype=np.float32)
    if hand.ndim != 2 or hand.shape[1] != 5:
        raise ValueError(f"Expected hand forces [T,5], got {hand.shape}")

    threshold = float(threshold)
    min_active_frames = max(1, int(min_active_frames))
    min_consecutive_frames = max(1, int(min_consecutive_frames))

    per_frame_any = hand.max(axis=1)
    per_frame_mean = hand.mean(axis=1)
    any_mask = per_frame_any >= threshold
    mean_mask = per_frame_mean >= threshold

    active_any_frames = int(any_mask.sum())
    active_mean5_frames = int(mean_mask.sum())
    consecutive_any_frames = max_true_run(any_mask)

    scores: Dict[str, float] = {
        "any_finger": float(per_frame_any.max(initial=0.0)),
        "mean5": float(per_frame_mean.max(initial=0.0)),
        "sustained_any": float(consecutive_any_frames),
        "window_mean5": float(hand.mean()) if hand.size else 0.0,
    }
    passes = {
        "any_finger": active_any_frames >= min_active_frames,
        "mean5": active_mean5_frames >= min_active_frames,
        "sustained_any": consecutive_any_frames >= min_consecutive_frames,
        "window_mean5": scores["window_mean5"] >= threshold,
    }
    return WindowFilterResult(
        threshold=threshold,
        scores=scores,
        passes=passes,
        active_any_frames=active_any_frames,
        active_mean5_frames=active_mean5_frames,
        consecutive_any_frames=consecutive_any_frames,
    )


def summarize_episode_windows(
    episode_index: int,
    full_f6: np.ndarray,
    thresholds: Sequence[float],
    source_window: int = 64,
    stride: int = 4,
    min_active_frames: int = 1,
    min_consecutive_frames: int = 8,
) -> List[WindowRecord]:
    """Return threshold/method decisions for one episode, split by hand."""

    force_norms = finger_force_norms(full_f6)
    hands = split_hand_forces(force_norms)
    starts = window_start_indices(force_norms.shape[0], source_window, stride)
    records: List[WindowRecord] = []

    for frame_start in starts:
        frame_start_int = int(frame_start)
        for hand_idx, hand_forces in enumerate(hands):
            window = hand_forces[frame_start_int: frame_start_int + source_window]
            for threshold in thresholds:
                result = evaluate_hand_window(
                    window,
                    threshold=threshold,
                    min_active_frames=min_active_frames,
                    min_consecutive_frames=min_consecutive_frames,
                )
                for method in FILTER_METHODS:
                    records.append(
                        WindowRecord(
                            episode_index=int(episode_index),
                            frame_start=frame_start_int,
                            hand=hand_idx,
                            threshold=float(threshold),
                            method=method,
                            score=float(result.scores[method]),
                            passed=bool(result.passes[method]),
                            active_any_frames=result.active_any_frames,
                            active_mean5_frames=result.active_mean5_frames,
                            consecutive_any_frames=result.consecutive_any_frames,
                        )
                    )
    return records


def iter_hand_windows(
    full_f6: np.ndarray,
    source_window: int = 64,
    stride: int = 4,
) -> Iterable[Tuple[int, int, np.ndarray]]:
    """Yield ``(frame_start, hand, hand_window)`` for all train-style windows."""

    force_norms = finger_force_norms(full_f6)
    hands = split_hand_forces(force_norms)
    for frame_start in window_start_indices(force_norms.shape[0], source_window, stride):
        frame_start_int = int(frame_start)
        for hand_idx, hand_forces in enumerate(hands):
            yield (
                frame_start_int,
                hand_idx,
                hand_forces[frame_start_int: frame_start_int + source_window],
            )
