"""Visualize tactile-force filtering methods for hand-wise VAE training."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tactile_vae.data.filtering import (  # noqa: E402
    FILTER_METHODS,
    HAND_NAMES,
    as_full_f6,
    evaluate_hand_window,
    finger_force_norms,
    iter_hand_windows,
    window_start_indices,
)


TACTILE_COLUMN = "observation.tactile_force"
EPISODE_COLUMN = "episode_index"
FRAME_COLUMN = "frame_index"
SCORE_NAMES = ("any_finger", "mean5", "window_mean5")


def _parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _threshold_label(threshold: float) -> str:
    text = f"{threshold:.8g}"
    return re.sub(r"[^0-9A-Za-z]+", "p", text).strip("p") or "0"


def _parquet_files(data_root: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_root}/data")
    return files


def iter_parquet_episodes(
    data_root: str,
    max_episodes: int = 0,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield full tactile arrays episode-wise without loading all files at once."""

    import pandas as pd

    pending_episode: Optional[int] = None
    pending_forces: List[np.ndarray] = []
    pending_frames: List[np.ndarray] = []
    yielded = 0

    def flush_pending() -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        nonlocal pending_episode, pending_forces, pending_frames
        if pending_episode is None:
            return None
        forces = as_full_f6(np.concatenate(pending_forces, axis=0))
        frames = np.concatenate(pending_frames, axis=0).astype(np.int64, copy=False)
        order = np.argsort(frames, kind="stable")
        out = int(pending_episode), forces[order], frames[order]
        pending_episode = None
        pending_forces = []
        pending_frames = []
        return out

    for path in _parquet_files(data_root):
        df = pd.read_parquet(path, columns=[TACTILE_COLUMN, EPISODE_COLUMN, FRAME_COLUMN])
        df = df.sort_values([EPISODE_COLUMN, FRAME_COLUMN])
        for episode, group in df.groupby(EPISODE_COLUMN, sort=True):
            episode = int(episode)
            forces = np.stack(group[TACTILE_COLUMN].to_numpy()).astype(np.float32, copy=False)
            frames = group[FRAME_COLUMN].to_numpy(dtype=np.int64)
            if pending_episode is None:
                pending_episode = episode
            if episode != pending_episode:
                flushed = flush_pending()
                if flushed is not None:
                    yield flushed
                    yielded += 1
                    if max_episodes > 0 and yielded >= max_episodes:
                        return
                pending_episode = episode
            pending_forces.append(forces)
            pending_frames.append(frames)

    flushed = flush_pending()
    if flushed is not None and (max_episodes <= 0 or yielded < max_episodes):
        yield flushed


def _window_scores(hand_window: np.ndarray) -> Dict[str, float]:
    per_frame_any = hand_window.max(axis=1)
    per_frame_mean = hand_window.mean(axis=1)
    return {
        "any_finger": float(per_frame_any.max(initial=0.0)),
        "mean5": float(per_frame_mean.max(initial=0.0)),
        "window_mean5": float(hand_window.mean()) if hand_window.size else 0.0,
    }


def collect_score_distributions(
    data_root: str,
    source_window: int,
    stride: int,
    max_episodes: int,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    scores = {name: [] for name in SCORE_NAMES}
    totals = {"episodes": 0, "hand_windows": 0}
    for episode_index, full_f6, _frames in iter_parquet_episodes(data_root, max_episodes=max_episodes):
        del episode_index
        totals["episodes"] += 1
        for _frame_start, _hand, hand_window in iter_hand_windows(full_f6, source_window, stride):
            totals["hand_windows"] += 1
            window_scores = _window_scores(hand_window)
            for name in SCORE_NAMES:
                scores[name].append(window_scores[name])
    return scores, totals


def threshold_candidates(
    scores: Dict[str, List[float]],
    percentiles: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    candidates: Dict[str, Dict[str, float]] = {}
    for name, values in scores.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            candidates[name] = {str(p): 0.0 for p in percentiles}
            continue
        candidates[name] = {
            str(p): float(np.percentile(arr, p))
            for p in percentiles
        }
    return candidates


def choose_auto_thresholds(candidates: Dict[str, Dict[str, float]], percentiles: Sequence[float]) -> List[float]:
    values = [candidates["any_finger"][str(p)] for p in percentiles]
    unique = sorted({round(float(v), 8) for v in values})
    return [v for v in unique if np.isfinite(v)]


def _new_counter() -> Dict[str, float]:
    return {
        "total_windows": 0,
        "kept_windows": 0,
        "score_sum": 0.0,
        "score_max": 0.0,
        "active_any_sum": 0.0,
        "active_mean5_sum": 0.0,
        "consecutive_any_sum": 0.0,
    }


def aggregate_filters(
    data_root: str,
    output_dir: Path,
    thresholds: Sequence[float],
    source_window: int,
    stride: int,
    min_active_frames: int,
    min_consecutive_frames: int,
    continuity_values: Sequence[int],
    save_window_indices: bool,
    example_episodes: int,
    seed: int,
    max_episodes: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Tuple[int, np.ndarray, np.ndarray]]]:
    counters: Dict[Tuple[str, float, int], Dict[str, float]] = defaultdict(_new_counter)
    continuity: Dict[Tuple[str, int, int], Dict[str, int]] = defaultdict(lambda: {"total_windows": 0, "kept_windows": 0})
    writers: Dict[Tuple[str, float], Tuple[object, csv.DictWriter]] = {}
    examples: List[Tuple[int, np.ndarray, np.ndarray]] = []
    rng = random.Random(seed)
    seen_episodes = 0

    try:
        for episode_index, full_f6, frames in iter_parquet_episodes(data_root, max_episodes=max_episodes):
            seen_episodes += 1
            if full_f6.shape[0] < source_window:
                continue
            if example_episodes > 0:
                if len(examples) < example_episodes:
                    examples.append((episode_index, full_f6.copy(), frames.copy()))
                else:
                    j = rng.randrange(seen_episodes)
                    if j < example_episodes:
                        examples[j] = (episode_index, full_f6.copy(), frames.copy())

            for frame_start, hand, hand_window in iter_hand_windows(full_f6, source_window, stride):
                actual_frame = int(frames[frame_start]) if frame_start < len(frames) else int(frame_start)
                for threshold in thresholds:
                    result = evaluate_hand_window(
                        hand_window,
                        threshold=threshold,
                        min_active_frames=min_active_frames,
                        min_consecutive_frames=min_consecutive_frames,
                    )
                    for method in FILTER_METHODS:
                        key = (method, float(threshold), int(hand))
                        counter = counters[key]
                        counter["total_windows"] += 1
                        counter["kept_windows"] += int(result.passes[method])
                        counter["score_sum"] += float(result.scores[method])
                        counter["score_max"] = max(counter["score_max"], float(result.scores[method]))
                        counter["active_any_sum"] += result.active_any_frames
                        counter["active_mean5_sum"] += result.active_mean5_frames
                        counter["consecutive_any_sum"] += result.consecutive_any_frames

                        if save_window_indices and result.passes[method]:
                            writer_key = (method, float(threshold))
                            if writer_key not in writers:
                                path = output_dir / f"window_index_{method}_{_threshold_label(threshold)}.csv"
                                fh = open(path, "w", newline="")
                                fieldnames = [
                                    "episode_index",
                                    "frame_start",
                                    "hand",
                                    "hand_name",
                                    "threshold",
                                    "method",
                                    "score",
                                    "active_any_frames",
                                    "active_mean5_frames",
                                    "consecutive_any_frames",
                                ]
                                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                                writer.writeheader()
                                writers[writer_key] = (fh, writer)
                            _fh, writer = writers[writer_key]
                            writer.writerow({
                                "episode_index": episode_index,
                                "frame_start": actual_frame,
                                "hand": hand,
                                "hand_name": HAND_NAMES[hand],
                                "threshold": threshold,
                                "method": method,
                                "score": result.scores[method],
                                "active_any_frames": result.active_any_frames,
                                "active_mean5_frames": result.active_mean5_frames,
                                "consecutive_any_frames": result.consecutive_any_frames,
                            })

                    if float(threshold) == float(thresholds[0]):
                        for value in continuity_values:
                            for method, kept in (
                                ("any_finger", result.active_any_frames >= value),
                                ("mean5", result.active_mean5_frames >= value),
                                ("sustained_any", result.consecutive_any_frames >= value),
                            ):
                                ckey = (method, int(value), int(hand))
                                continuity[ckey]["total_windows"] += 1
                                continuity[ckey]["kept_windows"] += int(kept)
    finally:
        for fh, _writer in writers.values():
            fh.close()

    summary_rows: List[Dict[str, object]] = []
    for (method, threshold, hand), counter in sorted(counters.items()):
        total = int(counter["total_windows"])
        kept = int(counter["kept_windows"])
        summary_rows.append({
            "method": method,
            "threshold": threshold,
            "hand": hand,
            "hand_name": HAND_NAMES[hand],
            "total_windows": total,
            "kept_windows": kept,
            "kept_ratio": kept / total if total else 0.0,
            "mean_score": counter["score_sum"] / total if total else 0.0,
            "max_score": counter["score_max"],
            "mean_active_any_frames": counter["active_any_sum"] / total if total else 0.0,
            "mean_active_mean5_frames": counter["active_mean5_sum"] / total if total else 0.0,
            "mean_consecutive_any_frames": counter["consecutive_any_sum"] / total if total else 0.0,
        })

    continuity_rows: List[Dict[str, object]] = []
    selected_threshold = float(thresholds[0]) if thresholds else 0.0
    for (method, value, hand), counter in sorted(continuity.items()):
        total = int(counter["total_windows"])
        kept = int(counter["kept_windows"])
        continuity_rows.append({
            "method": method,
            "threshold": selected_threshold,
            "continuity_value": value,
            "hand": hand,
            "hand_name": HAND_NAMES[hand],
            "total_windows": total,
            "kept_windows": kept,
            "kept_ratio": kept / total if total else 0.0,
        })
    return summary_rows, continuity_rows, examples


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_force_distributions(scores: Dict[str, List[float]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(SCORE_NAMES), 2, figsize=(12, 9), constrained_layout=True)
    for row, name in enumerate(SCORE_NAMES):
        arr = np.asarray(scores[name], dtype=np.float32)
        if arr.size == 0:
            continue
        axes[row, 0].hist(arr, bins=80, color="#4C78A8", alpha=0.85)
        axes[row, 0].set_title(f"{name} histogram")
        axes[row, 0].set_xlabel("force threshold score")
        axes[row, 0].set_ylabel("hand windows")

        sorted_arr = np.sort(arr)
        cdf = np.linspace(0.0, 1.0, sorted_arr.size, endpoint=True)
        axes[row, 1].plot(sorted_arr, cdf, color="#F58518", linewidth=2)
        axes[row, 1].set_title(f"{name} CDF")
        axes[row, 1].set_xlabel("force threshold score")
        axes[row, 1].set_ylabel("cumulative ratio")
        axes[row, 1].grid(True, alpha=0.25)
    fig.suptitle("Hand-Window Force Score Distributions")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_threshold_sweep(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for method in FILTER_METHODS:
        for hand in (0, 1):
            subset = [r for r in rows if r["method"] == method and r["hand"] == hand]
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: float(r["threshold"]))
            ax.plot(
                [float(r["threshold"]) for r in subset],
                [float(r["kept_ratio"]) for r in subset],
                marker="o",
                linewidth=1.8,
                label=f"{method} {HAND_NAMES[hand]}",
            )
    ax.set_xlabel("force threshold")
    ax.set_ylabel("kept hand-window ratio")
    ax.set_title("Threshold Sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_continuity_sweep(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for method in ("any_finger", "mean5", "sustained_any"):
        for hand in (0, 1):
            subset = [r for r in rows if r["method"] == method and r["hand"] == hand]
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: int(r["continuity_value"]))
            ax.plot(
                [int(r["continuity_value"]) for r in subset],
                [float(r["kept_ratio"]) for r in subset],
                marker="o",
                linewidth=1.8,
                label=f"{method} {HAND_NAMES[hand]}",
            )
    ax.set_xlabel("required active/consecutive raw frames")
    ax.set_ylabel("kept hand-window ratio")
    ax.set_title("Continuity Sweep at First Threshold")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_episode_examples(
    examples: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    output_dir: Path,
    threshold: float,
    method: str,
    source_window: int,
    stride: int,
    min_active_frames: int,
    min_consecutive_frames: int,
    max_frames: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for episode_index, full_f6, frames in examples:
        force_norms = finger_force_norms(full_f6)
        n_plot = min(force_norms.shape[0], max_frames) if max_frames > 0 else force_norms.shape[0]
        for hand in (0, 1):
            hand_forces = force_norms[:n_plot, hand * 5:(hand + 1) * 5]
            frame_values = frames[:n_plot] if len(frames) >= n_plot else np.arange(n_plot)
            starts = window_start_indices(n_plot, source_window, stride)
            accepted = []
            rejected = []
            for frame_start in starts:
                window = hand_forces[int(frame_start): int(frame_start) + source_window]
                result = evaluate_hand_window(
                    window,
                    threshold=threshold,
                    min_active_frames=min_active_frames,
                    min_consecutive_frames=min_consecutive_frames,
                )
                if result.passes[method]:
                    accepted.append(int(frame_start))
                else:
                    rejected.append(int(frame_start))

            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
            im = axes[0].imshow(
                hand_forces.T,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap="magma",
            )
            axes[0].set_yticks(range(5))
            axes[0].set_yticklabels(["thumb", "index", "middle", "ring", "pinky"])
            axes[0].set_ylabel("finger")
            axes[0].set_title(
                f"episode {episode_index} {HAND_NAMES[hand]} hand force norms"
            )
            fig.colorbar(im, ax=axes[0], label="3D force norm")

            axes[1].plot(hand_forces.max(axis=1), label="any finger", color="#4C78A8")
            axes[1].plot(hand_forces.mean(axis=1), label="mean5", color="#F58518")
            axes[1].axhline(threshold, color="#E45756", linestyle="--", linewidth=1.2, label="threshold")
            for start in rejected[::max(1, len(rejected) // 150 or 1)]:
                axes[1].axvspan(start, min(start + source_window, n_plot), color="#E45756", alpha=0.035)
            for start in accepted[::max(1, len(accepted) // 150 or 1)]:
                axes[1].axvspan(start, min(start + source_window, n_plot), color="#54A24B", alpha=0.06)
            axes[1].set_ylabel("force score")
            axes[1].set_xlabel(f"raw frame offset, first plotted frame_index={int(frame_values[0])}")
            axes[1].legend(loc="upper right", fontsize=8)
            axes[1].grid(True, alpha=0.25)
            fig.savefig(output_dir / f"episode_{episode_index}_{HAND_NAMES[hand]}_{method}.png", dpi=180)
            plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/data/d3/shenrui/trex_dataset_no_videos")
    parser.add_argument("--output_dir", default="tactile_vae/outputs/filter_viz/trex_default")
    parser.add_argument("--source_window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--thresholds", default="auto")
    parser.add_argument("--threshold_percentiles", default="50,60,70,80,90,95,97,99")
    parser.add_argument("--min_active_frames", type=int, default=1)
    parser.add_argument("--min_consecutive_frames", type=int, default=8)
    parser.add_argument("--continuity_values", default="1,2,4,8,16,32,64")
    parser.add_argument("--example_episodes", type=int, default=8)
    parser.add_argument("--example_method", choices=FILTER_METHODS, default="any_finger")
    parser.add_argument("--example_threshold", type=float, default=None)
    parser.add_argument("--example_max_frames", type=int, default=900)
    parser.add_argument("--save_window_indices", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episodes", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    percentiles = _parse_csv_floats(args.threshold_percentiles)
    continuity_values = _parse_csv_ints(args.continuity_values)

    scores, totals = collect_score_distributions(
        data_root=args.data_root,
        source_window=args.source_window,
        stride=args.stride,
        max_episodes=args.max_episodes,
    )
    candidates = threshold_candidates(scores, percentiles)
    if args.thresholds.strip().lower() == "auto":
        thresholds = choose_auto_thresholds(candidates, percentiles)
    else:
        thresholds = _parse_csv_floats(args.thresholds)
    if not thresholds:
        raise ValueError("No thresholds selected; provide --thresholds or non-empty --threshold_percentiles")

    candidate_payload = {
        "data_root": args.data_root,
        "score_percentiles": candidates,
        "auto_threshold_source": "any_finger",
        "selected_thresholds": thresholds,
        "totals": totals,
    }
    (output_dir / "threshold_candidates.json").write_text(json.dumps(candidate_payload, indent=2))
    plot_force_distributions(scores, output_dir / "force_distributions.png")

    summary_rows, continuity_rows, examples = aggregate_filters(
        data_root=args.data_root,
        output_dir=output_dir,
        thresholds=thresholds,
        source_window=args.source_window,
        stride=args.stride,
        min_active_frames=args.min_active_frames,
        min_consecutive_frames=args.min_consecutive_frames,
        continuity_values=continuity_values,
        save_window_indices=bool(args.save_window_indices),
        example_episodes=args.example_episodes,
        seed=args.seed,
        max_episodes=args.max_episodes,
    )

    write_csv(output_dir / "filter_summary.csv", summary_rows)
    write_csv(output_dir / "continuity_summary.csv", continuity_rows)
    plot_threshold_sweep(summary_rows, output_dir / "threshold_sweep.png")
    plot_continuity_sweep(continuity_rows, output_dir / "continuity_sweep.png")

    example_threshold = args.example_threshold
    if example_threshold is None:
        example_threshold = thresholds[min(len(thresholds) // 2, len(thresholds) - 1)]
    plot_episode_examples(
        examples=examples,
        output_dir=output_dir / "examples",
        threshold=float(example_threshold),
        method=args.example_method,
        source_window=args.source_window,
        stride=args.stride,
        min_active_frames=args.min_active_frames,
        min_consecutive_frames=args.min_consecutive_frames,
        max_frames=args.example_max_frames,
    )

    run_config = {
        "args": vars(args),
        "selected_thresholds": thresholds,
        "example_threshold": example_threshold,
        "outputs": {
            "filter_summary": str(output_dir / "filter_summary.csv"),
            "continuity_summary": str(output_dir / "continuity_summary.csv"),
            "threshold_candidates": str(output_dir / "threshold_candidates.json"),
            "force_distributions": str(output_dir / "force_distributions.png"),
            "threshold_sweep": str(output_dir / "threshold_sweep.png"),
            "continuity_sweep": str(output_dir / "continuity_sweep.png"),
            "examples": str(output_dir / "examples"),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps(run_config["outputs"], indent=2))


if __name__ == "__main__":
    main()
